from lib.dataset import createModularAdditionData, randomSplit, CustomDataset
from lib.transformer import Transformer, remap_and_load_state_dict
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
from lib.train import test
from lib.visualization import plot_learning_acc_and_loss, visualize_2D_matrix
from lib.metrics import fraction_variance_explained
from torch.nn import CrossEntropyLoss
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

module = 113

# Dataset

data = createModularAdditionData(module)
train_dataset, val_dataset, test_dataset = (CustomDataset(data, indices) for indices in randomSplit(data.size(0), fractions=(0.0, 1.0, 0), device=data.device))

val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True)

# Model Load saved

model = Transformer(d_model=128, n_heads=4, mlp_dim_multiplier=4, vocab_size=module, n_layers=1, constant_attn=False).to(device)

PATH = "./data/transformer_clock"

remap_and_load_state_dict(model, PATH+".pt", weights_only=True, strict=True)
model.eval()

results = None
with open(PATH+".json", 'r') as f:
    results = json.load(f)

a, b, label = (i.to(device) for i in next(iter(val_loader)))
model(a, b)


# Metrics


test(model, CrossEntropyLoss(), val_loader, device)


#plot_learning_acc_and_loss(loss_tr=results["loss_train"], acc_tr=results["acc_train"], loss_val=results["loss_val"], acc_val=results["acc_val"], norm=results["norm"], gradient_norm=results["grad_norm"], block=False)

# Test the impact of certain activations on the output

embedding_out_1 = model.pos_encoding_out[:, 0]
embedding_out_2 = model.pos_encoding_out[:, 1]
mha_out_1 = model.blocks.get_submodule("0").mha_out[:, 0] # type: ignore
mha_out_2 = model.blocks.get_submodule("0").mha_out[:, 1] # type: ignore
mlp_in = model.blocks.get_submodule("0").mlp_in[:, -1] # type: ignore
mlp_out = model.blocks.get_submodule("0").mlp_out[:, -1] # type: ignore
block_out = model.blocks_out[:, -1]
logits = model.logits
attn_weights = model.blocks.get_submodule("0").attn_weights.data.detach() # type: ignore
#attn_pre_out = model.blocks.get_submodule("0").attn_pre_out.data.detach() # type: ignore
#attn_logits = torch.log(attn_weights) - torch.mean(torch.log(attn_weights), dim=-1, keepdim=True) # type: ignore
print(attn_weights[0, 0])
print(attn_weights[0, 1])
print(attn_weights[0, 2])
print(attn_weights[0, 3])

print("\nFVE of mlp input:")
print("mha input tok 1 ->", torch.mean(fraction_variance_explained(embedding_out_1, mlp_in)).item()*100, "%")
print("mha output tok 1 ->", torch.mean(fraction_variance_explained(mha_out_1, mlp_in)).item()*100, "%")
print("mha input tok 2 ->", torch.mean(fraction_variance_explained(embedding_out_2, mlp_in)).item()*100, "%")
print("mha output tok 2 ->", torch.mean(fraction_variance_explained(mha_out_2, mlp_in)).item()*100, "%")

print("\nFVE of transformer blocks' output:")
print("mlp input ->", torch.mean(fraction_variance_explained(mlp_in, block_out)).item()*100, "%")
print("mlp output ->", torch.mean(fraction_variance_explained(mlp_out, block_out)).item()*100, "%")


# Look at the features at each point of the model and plot them as well as their Fourier transform

a, b, label = (i.to(device) for i in next(iter(val_loader)))
model(a, b)
label = label.to(dtype=torch.long)

logit_matrix = torch.empty(size=(module, module))
attn_logit_matrix_a_to_b = torch.empty(size=(module, module))
attn_logit_matrix_a_to_a = torch.empty(size=(module, module))
attn_output_from_b_to_a = torch.empty(size=(module, module, 32))
attn_output_from_a_to_a = torch.empty(size=(module, module, 32))
attn_output_a = torch.empty(size=(module, module, 32))

pos_encoding_out_b = torch.empty(size=(module, module, 128))
mha_output_b = torch.empty(size=(module, module, 128))
mlp_in_b = torch.empty(size=(module, module, 128))
mlp_pre_act_b = torch.empty(size=(module, module, 128*4))
mlp_post_act_b = torch.empty(size=(module, module, 128*4))
mlp_output_b = torch.empty(size=(module, module, 128))
logits_b = torch.empty(size=(module, module, 113))

for idx in range(len(a)):
    _a, _b, _label = a[idx], b[idx], label[idx]
    logits = model.logits[idx].detach()
    
    correct_logit = logits[_label]
    logit_matrix[(_a-_b)%module, (_a+_b)%module] = correct_logit

    attn_weights = model.blocks.get_submodule("0").attn_weights[idx].detach() # type: ignore    
    attn_logits = torch.log(attn_weights) - torch.mean(torch.log(attn_weights), dim=-1, keepdim=True)
    attn_logit_matrix_a_to_b[_a, _b] = attn_weights[1, 1, 0]
    attn_logit_matrix_a_to_a[_a, _b] = attn_weights[1, 0, 0]

    attn_pre_out = model.blocks.get_submodule("0").attn_pre_out[idx].detach() # type: ignore
    attn_output_from_b_to_a[_a, _b] = attn_pre_out[1, 1]
    attn_output_from_a_to_a[_a, _b] = attn_pre_out[1, 0]
    attn_output_a[_a, _b] = attn_pre_out[1, 1] + attn_pre_out[1, 0]

    pos_encoding_out_b[_a, _b] = model.pos_encoding_out[idx, 1].detach() # type: ignore

    mha_output_b[_a, _b] = model.blocks.get_submodule("0").mha_out[idx, 1].detach() # type: ignore

    mlp_in_b[_a, _b] = model.blocks.get_submodule("0").mlp_in[idx, 1].detach() # type: ignore
    mlp_pre_act_b[_a, _b] = model.blocks.get_submodule("0").mlp_pre_act[idx, 1].detach() # type: ignore
    mlp_post_act_b[_a, _b] = model.blocks.get_submodule("0").mlp_post_act[idx, 1].detach() # type: ignore
    mlp_output_b[_a, _b] = model.blocks.get_submodule("0").mlp_out[idx, 1].detach() # type:ignore
    logits_b[_a, _b] = model.logits[idx].detach()


#visualize_2D_matrix(torch.mean(pos_encoding_out_b, dim=-1).cpu().numpy(), "pre-attention of token b")
#pos_encoding_out_b_2d_fft = torch.fft.fft2(pos_encoding_out_b.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(pos_encoding_out_b_2d_fft), dim=0).cpu().numpy(), "pre-attention of token b, fourier 2D")

#visualize_2D_matrix(torch.mean(mha_output_b, dim=-1).cpu().numpy(), "multiheadattention output of token b")
#mha_output_b_2d_fft = torch.fft.fft2(mha_output_b.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(mha_output_b_2d_fft), dim=0).cpu().numpy(), "multiheadattention output of token b, fourier 2D")

#visualize_2D_matrix(torch.mean(mlp_in_b, dim=-1).cpu().numpy(), "mlp input of token b")
#mlp_in_b_2d_fft = torch.fft.fft2(mlp_in_b.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(mlp_in_b_2d_fft), dim=0).cpu().numpy(), "mlp input of token b, fourier 2D")

#visualize_2D_matrix(torch.mean(mlp_pre_act_b, dim=-1).cpu().numpy(), "mlp preactivations of token b")
#mlp_pre_act_b_2d_fft = torch.fft.fft2(mlp_pre_act_b.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(mlp_pre_act_b_2d_fft), dim=0).cpu().numpy(), "mlp preactivations of token b, fourier 2D")

#visualize_2D_matrix(torch.mean(mlp_post_act_b, dim=-1).cpu().numpy(), "mlp activations of token b")
#mlp_post_act_b_2d_fft = torch.fft.fft2(mlp_post_act_b.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(mlp_post_act_b_2d_fft), dim=0).cpu().numpy(), "mlp activations of token b, fourier 2D")

#visualize_2D_matrix(torch.mean(mlp_output_b, dim=-1).cpu().numpy(), "mlp output of token b")
#mlp_output_b_2d_fft = torch.fft.fft2(mlp_output_b.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(mlp_output_b_2d_fft), dim=0).cpu().numpy(), "mlp output of token b, fourier 2D")

#visualize_2D_matrix(torch.mean(logits_b, dim=-1).cpu().numpy(), "logits of token b")
#logits_b_2d_fft = torch.fft.fft2(logits_b.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(logits_b_2d_fft), dim=0).cpu().numpy(), "logits of token b, fourier 2D")

#visualize_2D_matrix(logit_matrix.cpu().numpy(), "Correct logit dependence on sum and difference of inputs")

#visualize_2D_matrix(attn_logit_matrix_a_to_b.cpu().numpy(), "Attention given to token a by token b")
#visualize_2D_matrix(attn_logit_matrix_a_to_a.cpu().numpy(), "Attention given to token a by token a")
#visualize_2D_matrix(torch.abs(torch.fft.fft2(attn_logit_matrix_a_to_b)).cpu().numpy(), "Attention given to a by b, 2D transform")
#visualize_2D_matrix(torch.abs(torch.fft.fft2(attn_logit_matrix_a_to_a)).cpu().numpy(), "Attention given to a by a, 2D transform")

#visualize_2D_matrix(torch.mean(attn_output_from_b_to_a, dim=-1).cpu().numpy(), "Attention output of token a with respect to b")
#visualize_2D_matrix(torch.mean(attn_output_from_a_to_a, dim=-1).cpu().numpy(), "Attention output of token a with respect to a")
#visualize_2D_matrix(torch.mean(attn_output_a, dim=-1).cpu().numpy(), "Attention output of token a")
attn_output_from_b_to_a_2d_fft = torch.fft.fft2(attn_output_from_b_to_a.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(attn_output_from_b_to_a_2d_fft), dim=0).cpu().numpy(), "Attention output of token a with respect to b, fourier 2D")
attn_output_from_a_to_a_2d_fft = torch.fft.fft2(attn_output_from_a_to_a.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(attn_output_from_a_to_a_2d_fft), dim=0).cpu().numpy(), "Attention output of token a with respect to a, fourier 2D")
attn_output_a_2d_fft = torch.fft.fft2(attn_output_a.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(attn_output_a_2d_fft), dim=0).cpu().numpy(), "Attention output of token a, fourier 2D")

# Compare the predicted frequencies with the actual frequencies via their Frobenius norm difference

key_freqs = [0, 15, 19, 28, 30, 38, 56]
attn_freq = 29

output_freqs = []

# Frequencies from scores_a,b * value_b
output_freqs.extend([(0, k) for k in key_freqs if (0, k) not in output_freqs])
output_freqs.extend([(0, module-k) for k in key_freqs if (0, module-k) not in output_freqs])
output_freqs.extend([(attn_freq, (attn_freq+k) % module) for k in key_freqs if (attn_freq, (attn_freq+k) % module) not in output_freqs])
output_freqs.extend([(attn_freq, (attn_freq-k) % module) for k in key_freqs if (attn_freq, (attn_freq-k) % module) not in output_freqs])
output_freqs.extend([(2*attn_freq, k) for k in key_freqs if (2*attn_freq, k) not in output_freqs])
output_freqs.extend([(2*attn_freq, module-k) for k in key_freqs if (2*attn_freq, module-k) not in output_freqs])

# Frequencies from scores_a,a * value_a
output_freqs.extend([(k, 0) for k in key_freqs if (k, 0) not in output_freqs])
output_freqs.extend([(module-k, 0) for k in key_freqs if (module-k, 0) not in output_freqs])
output_freqs.extend([((attn_freq+k) % module, attn_freq) for k in key_freqs if ((attn_freq+k) % module, attn_freq) not in output_freqs])
output_freqs.extend([((attn_freq-k) % module, attn_freq) for k in key_freqs if ((attn_freq-k) % module, attn_freq) not in output_freqs])
output_freqs.extend([((2*attn_freq+k) % module, 0) for k in key_freqs if ((2*attn_freq+k) % module, 0) not in output_freqs])
output_freqs.extend([((2*attn_freq-k) %module, 0) for k in key_freqs if ((2*attn_freq-k) % module, 0) not in output_freqs])
print(output_freqs)

def frobenius_compare(tensor, indices):
    full = torch.linalg.norm(tensor, ord="fro").item()

    valid_indices = [(i,j) for i,j in indices if i != 113 and j != 113]
    rows, cols = zip(*valid_indices)
    subset = tensor[rows, cols]

    selected = torch.linalg.norm(subset, ord=2).item() # because subset is now a 1D vector after selecting

    print(full, selected)
    print(math.fabs(full-selected)*100/full, "%")

attn_output_norm_a_2d_fft = torch.linalg.norm(attn_output_a_2d_fft, dim=0, ord=2) # collapse the first dimension which is the feature dimension (D, N, N)
frobenius_compare(attn_output_norm_a_2d_fft, output_freqs)

plt.show()