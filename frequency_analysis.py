from lib.dataset import createModularAdditionData, randomSplit, CustomDataset
from lib.transformer import Transformer, remap_and_load_state_dict
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
from lib.train import test
from lib.visualization import plot_learning_acc_and_loss, visualize_2D_matrix
from lib.metrics import fraction_variance_explained, compare_frequency_full_vs_subset
from torch.nn import CrossEntropyLoss
import math

device = torch.device(torch.accelerator.current_accelerator())
print(device)

module = 113

# Dataset

data = createModularAdditionData(module)
train_dataset, val_dataset, test_dataset = (CustomDataset(data, indices) for indices in randomSplit(data.size(0), fractions=(0.0, 1.0, 0), device=data.device))

val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True)

# Model Load saved

model = Transformer(d_model=128, n_heads=4, mlp_dim_multiplier=4, vocab_size=module, n_layers=1, constant_attn=False).to(device)

PATH = "./data/transformer_clock"

remap_and_load_state_dict(model, PATH+".pt", device, weights_only=True, strict=False)
model.eval()

results = None
with open(PATH+".json", 'r') as f:
    results = json.load(f)

test(model, CrossEntropyLoss(), val_loader, device)

a, b, label = (i.to(device) for i in next(iter(val_loader)))
model(a, b)
label = label.to(dtype=torch.long)

logit_matrix = torch.empty(size=(module, module))
attn_logit_matrix_a_to_b = torch.empty(size=(module, module))
attn_logit_matrix_a_to_a = torch.empty(size=(module, module))
attn_output_from_b_to_a = torch.empty(size=(module, module, 32))
attn_output_from_a_to_a = torch.empty(size=(module, module, 32))
attn_output_a = torch.empty(size=(module, module, 32))

pos_encoding_out_a = torch.empty(size=(module, module, 128))

for idx in range(len(a)):
    _a, _b, _label = a[idx], b[idx], label[idx]

    pos_encoding_out_a[_a, _b] = model.pos_encoding_out[idx, 0].detach() # type: ignore

    attn_weights = model.blocks.get_submodule("0").attn_weights[idx].detach() # type: ignore    
    attn_logits = torch.log(attn_weights) - torch.mean(torch.log(attn_weights), dim=-1, keepdim=True)
    attn_logit_matrix_a_to_b[_a, _b] = attn_weights[1, 1, 0]
    attn_logit_matrix_a_to_a[_a, _b] = attn_weights[1, 0, 0]

    attn_pre_out = model.blocks.get_submodule("0").attn_pre_out[idx].detach() # type: ignore
    attn_output_from_b_to_a[_a, _b] = attn_pre_out[1, 1]
    attn_output_from_a_to_a[_a, _b] = attn_pre_out[1, 0]
    attn_output_a[_a, _b] = attn_pre_out[1, 1] + attn_pre_out[1, 0]    


pos_encoding_out_a_fft = torch.norm(torch.fft.fft2(pos_encoding_out_a.permute(2, 0, 1), norm="ortho"), p=2, dim=0)

attn_output_from_b_to_a_2d_fft = torch.fft.fft2(attn_output_from_b_to_a.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(attn_output_from_b_to_a_2d_fft), dim=0).cpu().numpy(), "Attention output of token a with respect to b, fourier 2D")
attn_output_from_a_to_a_2d_fft = torch.fft.fft2(attn_output_from_a_to_a.permute(2, 0, 1))
#visualize_2D_matrix(torch.mean(torch.abs(attn_output_from_a_to_a_2d_fft), dim=0).cpu().numpy(), "Attention output of token a with respect to a, fourier 2D")
attn_output_a_2d_fft = torch.norm(torch.fft.fft2(attn_output_a.permute(2, 0, 1), norm="ortho"), p=2, dim=0)
#visualize_2D_matrix(torch.mean(torch.abs(attn_output_a_2d_fft), dim=0).cpu().numpy(), "Attention output of token a, fourier 2D")

# calculate embedding coefficients

W_E = model.embedding.weight.data.T

_W_E = torch.fft.ifft(W_E, dim=1, norm="forward")

# calculate embedding predicted frequencies and coefficients

embedding_key_freqs = [0, 15, 19, 28, 29, 30, 55, 56]

embedding_output_freqs = [(k, 0, _W_E[:, k]) for k in embedding_key_freqs]
embedding_output_freqs.extend([(module-k, 0, _W_E[:, k].conj()) for k in embedding_key_freqs if k!=0])

# compare the norm of the chosen frequencies with the norm of the full frequency matrix

unique_freqs = []
for i,j,_ in embedding_output_freqs:
    if (i,j) not in unique_freqs:
        unique_freqs.append((i,j))

print("\nEmbedding Frequencies Residual Norm")
compare_frequency_full_vs_subset(pos_encoding_out_a_fft, unique_freqs)

# compare the predicted coefficients with the observed ones

print("\nEmbedding Frequencies Prediction")
for i,j,_ in embedding_output_freqs:
    predicted = torch.linalg.norm(sum([coeff for _i,_j,coeff in embedding_output_freqs if i==_i and j==_j]), dim=0, ord=2).item()
    observed = pos_encoding_out_a_fft[i,j].item()

    print(i,j)
    print(f"Observed: {observed:.2f}, Predicted: {predicted:.2f}, Relative Accuracy: {100*(1.0 - math.fabs(observed-predicted)/observed):.1f}%\n")


# calculate attention head 1 predicted frequencies and coefficients

attn_proj = model.blocks.get_submodule("0.mha").in_proj.weight.data # type: ignore
Q,K,V = attn_proj[:128], attn_proj[128:256], attn_proj[256:384] # type: ignore
Q,K,V = Q[32:64], K[32:64], V[32:64]

_W_Q, _W_K, _W_V = torch.fft.ifft(Q @ W_E, dim=1, norm="forward"), torch.fft.ifft(K @ W_E, dim=1, norm="forward"), torch.fft.ifft(V @ W_E, dim=1, norm="forward")

Z = 1.0/(2.0*module*math.sqrt(32)) # 32 = D/4 from softmax normalization
B = 1.0/math.sqrt(module)

A       = B * (1 + Z*((_W_Q.T @ _W_K.conj())[29, 29] + (_W_Q.conj().T @ _W_K)[29, 29])) * _W_V
A_      = B * (1 - Z*((_W_Q.T @ _W_K.conj())[29, 29] + (_W_Q.conj().T @ _W_K)[29, 29])) * _W_V
QKV     = B * Z * (_W_Q.T @ _W_K)[29, 29] * _W_V
Q_KV    = B * Z * (_W_Q.T @ _W_K.conj())[29, 29] * _W_V
_QKV    = B * Z * (_W_Q.conj().T @ _W_K)[29, 29] * _W_V
_Q_KV   = B * Z * (_W_Q.conj().T @ _W_K.conj())[29, 29] * _W_V


value_freqs = [0, 15, 19, 28, 30, 38, 56]
query_key_freqs = [29]

attention_output_freqs = []

for _k in value_freqs:
    # Frequencies from scores_a,b * value_b
    attention_output_freqs.append((0, _k, A_[:, _k]))

    # Frequencies from scores_a,a * value_a
    attention_output_freqs.append((_k, 0, A[:, _k]))

    for k in query_key_freqs:
        # Frequencies from scores_a,b * value_b
        attention_output_freqs.append((k, (k+_k) % module, QKV[:, _k]))

        attention_output_freqs.append((k, (k-_k) % module, _Q_KV[:, _k]))

        attention_output_freqs.append((k, (module - (k-_k)) % module, Q_KV[:, _k]))

        attention_output_freqs.append((k, (module - (k+_k)) % module, _QKV[:, _k]))

        attention_output_freqs.append((2*k, _k, -QKV[:, _k]))

        attention_output_freqs.append((2*k, (module - _k) % module, -_Q_KV[:, _k]))

        if _k!=0:
            attention_output_freqs.append((0, (module-_k) % module, A_.conj()[:, _k]))
            attention_output_freqs.append((module-k, (module - (k+_k)) % module, QKV.conj()[:, _k]))
            attention_output_freqs.append((module-k, (module - (k-_k)) % module, _Q_KV.conj()[:, _k]))
            attention_output_freqs.append((module-k, (k-_k) % module, Q_KV.conj()[:, _k]))
            attention_output_freqs.append((module-k, (k+_k) % module, _QKV.conj()[:, _k]))
            attention_output_freqs.append(((module - 2*k) % module, (module - _k) % module, -QKV.conj()[:, _k]))
            attention_output_freqs.append(((module - 2*k) % module, _k, -_Q_KV.conj()[:, _k]))


        # Frequencies from scores_a,a * value_a
        attention_output_freqs.append(((k+_k) % module, k, QKV[:, _k]))

        attention_output_freqs.append(((k-_k) % module, k, _Q_KV[:, _k]))

        attention_output_freqs.append(((module - (k-_k)) % module, k, Q_KV[:, _k]))

        attention_output_freqs.append(((module - (k+_k)) % module, k, _QKV[:, _k]))

        attention_output_freqs.append(((2*k+_k) % module, 0, -QKV[:, _k]))
        attention_output_freqs.append(((2*k-_k) % module, 0, -_Q_KV[:, _k]))

        if _k!=0:
            attention_output_freqs.append(((module-_k) % module, 0, A.conj()[:, _k]))
            attention_output_freqs.append(((module - (k+_k)) % module, module-k, QKV.conj()[:, _k]))
            attention_output_freqs.append(((module - (k-_k)) % module, module-k, _Q_KV.conj()[:, _k]))
            attention_output_freqs.append(((k-_k) % module, module-k, Q_KV.conj()[:, _k]))
            attention_output_freqs.append(((k+_k) % module, module-k, _QKV.conj()[:, _k]))
            attention_output_freqs.append(((module - (2*k+_k)) % module, 0, -QKV.conj()[:, _k]))
            attention_output_freqs.append(((module - (2*k-_k)) % module, 0, -_Q_KV.conj()[:, _k]))


print(len(attention_output_freqs))

# compare the norm of the chosen frequencies with the norm of the full frequency matrix

unique_freqs = []
for i,j,_ in attention_output_freqs:
    if (i,j) not in unique_freqs:
        unique_freqs.append((i,j))

print("\nAttention Frequencies Residual Norm")
compare_frequency_full_vs_subset(attn_output_a_2d_fft, unique_freqs)

# compare the predicted coefficients with the observed ones

print("\nAttention Frequencies Prediction")
for i,j,_ in attention_output_freqs:
    predicted = torch.linalg.norm(sum([coeff for _i,_j,coeff in attention_output_freqs if i==_i and j==_j]), dim=0, ord=2).item()
    observed = attn_output_a_2d_fft[i,j].item()

    print(i,j)
    print(f"Observed: {observed:.2f}, Predicted: {predicted:.2f}, Relative Accuracy: {100*(1.0 - math.fabs(observed-predicted)/observed):.1f}%\n")