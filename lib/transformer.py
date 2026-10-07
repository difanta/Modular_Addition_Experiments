from typing import Tuple

import torch
from torch.nn import Module
from torch.nn.modules import MultiheadAttention, Linear, ReLU, Sequential, Embedding
from torch.nn.functional import softmax
from torch.nn.parameter import Parameter
import math

# Constant attention

def remap_and_load_state_dict(model, state_dict_path, strict=True, weights_only=False):
    original_state_dict = torch.load(state_dict_path, weights_only=weights_only)
    
    remapped_state_dict = {}
    
    for k, v in original_state_dict.items():
        new_key = k
        
        if k.endswith('.in_proj_weight'):
            new_key = k.replace('.in_proj_weight', '.in_proj.weight')
            
        elif k.endswith('.in_proj_bias'):
            new_key = k.replace('.in_proj_bias', '.in_proj.bias')

        elif k.endswith('blocks.0.mlp.0.weight'):
            new_key = k.replace('blocks.0.mlp.0.weight', 'blocks.0.mlp_linear_in.weight')

        elif k.endswith('blocks.0.mlp.0.bias'):
            new_key = k.replace('blocks.0.mlp.0.bias', 'blocks.0.mlp_linear_in.bias')

        elif k.endswith('blocks.0.mlp.2.weight'):
            new_key = k.replace('blocks.0.mlp.2.weight', 'blocks.0.mlp_linear_out.weight')

        elif k.endswith('blocks.0.mlp.2.bias'):
            new_key = k.replace('blocks.0.mlp.2.bias', 'blocks.0.mlp_linear_out.bias')
            
        remapped_state_dict[new_key] = v

    model.load_state_dict(remapped_state_dict, strict=strict)
    
    return model

class ConstantMultiHeadAttention(Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.v_proj = Linear(embed_dim, embed_dim) # bias should be False
        self.out_proj = Linear(embed_dim, embed_dim) # Also here?
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by num_heads"

    def forward(self, query, key, value, need_weights=False, needs_pre_out_proj=False):
        B, T, H = value.shape

        value = self.v_proj(value)
        value = value.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.ones(size=(1, self.n_heads, T, T), device=value.device)

        output = scores @ value
        output = output.transpose(1, 2).contiguous().view(B, T, H)

        return (self.out_proj(output), scores if need_weights else None, None)

class MultiHeadSelfAttention(Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_proj = Linear(embed_dim, 3*embed_dim) # bias should be False
        self.out_proj = Linear(embed_dim, embed_dim) # Also here?
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by num_heads"

    def forward(self, x, _, __, need_weights=False, needs_pre_out_proj=False) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B, T, H = x.shape

        in_proj = self.in_proj(x)
        q,k,v = in_proj[..., :self.embed_dim], in_proj[..., self.embed_dim:2*self.embed_dim], in_proj[..., 2*self.embed_dim:3*self.embed_dim]

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1))/math.sqrt(self.head_dim)
        scores = torch.softmax(logits, dim=-1) #(1/logits.size(-1))*(1 + logits - torch.mean(logits, dim=-1, keepdim=True))
        
        pre_out_proj = torch.stack((scores[:, :, 0, 0].unsqueeze(-1) * v[:, :, 0], scores[:, :, 0, 1].unsqueeze(-1) * v[:, :, 1]), dim=2)
        output = scores @ v
        output = output.transpose(1, 2).contiguous().view(B, T, H)

        return (self.out_proj(output), scores if need_weights else None, pre_out_proj if needs_pre_out_proj else None)


class PositionalEncoding(Module):
    def __init__(self, max_seq_length, d_model):
        super().__init__()
        self.register_parameter("pos_encoding", Parameter(data=torch.randn((1, max_seq_length, d_model))))

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1)] # type: ignore

class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, mlp_dim_multiplier, constant_attn):
        super().__init__()
        self.mha = ConstantMultiHeadAttention(d_model, n_heads) if constant_attn else MultiHeadSelfAttention(d_model, n_heads)
        self.mlp_linear_in = Linear(d_model, mlp_dim_multiplier*d_model)
        self.mlp_activation = ReLU()
        self.mlp_linear_out = Linear(d_model*mlp_dim_multiplier, d_model)

    def forward(self, x):
        self.mha_out, self.attn_weights, self.attn_pre_out = self.mha(x, x, x, need_weights=True, needs_pre_out_proj=True)
        x = self.mlp_in = x + self.mha_out

        self.mlp_pre_act = self.mlp_linear_in(x)
        self.mlp_post_act = self.mlp_activation(self.mlp_pre_act)
        self.mlp_out = self.mlp_linear_out(self.mlp_post_act)

        return x + self.mlp_out
            
class Transformer(Module):
    def __init__(self, d_model, n_heads, mlp_dim_multiplier, vocab_size, n_layers, constant_attn):
        super().__init__()
        assert d_model % n_heads == 0
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(3, d_model)
        self.blocks = Sequential()
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model, n_heads, mlp_dim_multiplier, constant_attn))
        self.unembed = Linear(d_model, vocab_size)

    def forward(self, a, b, module = None):
        x = None
        if module is not None:
            x = torch.stack((self.embedding(a), self.embedding(b), self.embedding(module)), dim=1)
        else:
            x = torch.stack((self.embedding(a), self.embedding(b)), dim=1)
        
        x = self.pos_encoding_out = self.pos_encoding(x)

        x = self.blocks_out = self.blocks(x)

        # use the last token for classification
        self.logits = self.unembed(x[:, -1])

        return self.logits