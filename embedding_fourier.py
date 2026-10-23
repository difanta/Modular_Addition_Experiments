from lib.train import get_best_accelerator
from lib.transformer import Transformer, remap_and_load_state_dict
import torch
import matplotlib.pyplot as plt
import numpy as np
from lib.visualization import visualize_2D_matrix
import math

def hoyer_sparsity(matrix):
    matrix = matrix.reshape(-1)
    N = matrix.size(0)
    return (math.sqrt(N) - (torch.linalg.vector_norm(matrix, ord=1)/torch.linalg.vector_norm(matrix, ord=2))) / (math.sqrt(N) - 1)

device = get_best_accelerator()

print(device)

module=113

model = Transformer(d_model=128, n_heads=4, mlp_dim_multiplier=4, vocab_size=module, n_layers=1, constant_attn=False).to(device)

PATH = "./data/transformer_clock"

remap_and_load_state_dict(model, PATH+".pt", device, weights_only=True, strict=True)
model.eval()

# Plot embedding and unembedding fourier components
fig, (ax1, ax2) = plt.subplots(2, 1)

W_E = model.embedding.weight.data.T
fourier_E = torch.fft.ifft(W_E, dim=1, norm="forward")
magnitudes_E = torch.linalg.norm(fourier_E, dim=0, ord=2).cpu().numpy()

ax1.bar(np.arange(len(magnitudes_E)), magnitudes_E, color='blue')
ax1.set_xlabel('Frequencies')
ax1.set_ylabel('Norm')
ax1.set_title('Embedding Fourier Components')

W_U = model.unembed.weight.data
fourier_U = torch.fft.fft(W_U, dim=0, norm="forward")
magnitudes_U = torch.linalg.norm(fourier_U, dim=1, ord=2).cpu().numpy()

ax2.bar(np.arange(len(magnitudes_U)), magnitudes_U, color='blue')
ax2.set_xlabel('Frequencies')
ax2.set_ylabel('Norm')
ax2.set_title('Unembedding Fourier Components')

plt.tight_layout()
plt.show(block=False)

# Plot positional encodings

positional_encodings = model.get_parameter("pos_encoding.pos_encoding")[0, :2].detach()
#visualize_2D_matrix(positional_encodings.cpu().numpy(), "Positional encodings")

# Plot embedding and unembedding 2D fourier components

print(W_E.shape)
print(W_U.shape)

attn_proj = model.blocks.get_submodule("0.mha").in_proj.weight.data # type: ignore
q, k, v = attn_proj[:128], attn_proj[128:256], attn_proj[256:384] # type: ignore
print(attn_proj.shape)

fourier_1d_E = torch.fft.ifft(W_E, dim=1, norm="ortho") # type: ignore
fourier_2d_E = torch.fft.fft(fourier_1d_E, dim=0, norm="ortho")
visualize_2D_matrix(torch.abs(fourier_1d_E).cpu().numpy(), "Embedding 1D Fourier Magnitudes")

fourier_1d_q_E = torch.fft.ifft(q @ W_E, dim=1, norm="ortho")
fourier_2d_q_E = torch.fft.fft(q @ W_E, dim=0, norm="ortho")
visualize_2D_matrix(torch.abs(fourier_1d_q_E).cpu().numpy(), "Embedding x Query 1D Fourier Magnitudes")
#visualize_2D_matrix(torch.abs(fourier_2d_q_E).cpu().numpy(), "Embedding x Query 2D Fourier Magnitudes")
#visualize_2D_matrix(torch.abs(torch.fft.ifft(k @ W_E, dim=1, norm="ortho")).cpu().numpy(), "Embedding x Key 1D Fourier Magnitudes")
#visualize_2D_matrix(torch.abs(torch.fft.ifft(v @ W_E, dim=1, norm="ortho")).cpu().numpy(), "Embedding x Value 1D Fourier Magnitudes")
#visualize_2D_matrix(torch.abs(fourier_2d_E).cpu().numpy(), "Embedding 2D Fourier Magnitudes")

fourier_1d_q = torch.fft.ifft(q, dim=1, norm="ortho")
fourier_2d_q = torch.fft.fft(fourier_1d_q, dim=0, norm="ortho")
visualize_2D_matrix(torch.abs(q).cpu().numpy(), "Query 1D Fourier Magnitudes")
#visualize_2D_matrix(torch.abs(fourier_2d_q).cpu().numpy(), "Query 2D Fourier Magnitudes")

print("p=1: output space sparsity \np=inf: input space sparsity ")

we_overall = torch.linalg.matrix_norm(fourier_1d_E, ord='fro')
print("W_E overall", hoyer_sparsity(fourier_1d_E))
print("W_E p=1", torch.linalg.matrix_norm(fourier_1d_E, ord=1) / we_overall)
print("W_E p=inf", torch.linalg.matrix_norm(fourier_1d_E, ord=float('inf')) / we_overall)

q_we_overall = torch.linalg.matrix_norm(fourier_1d_q_E, ord='fro')
print("q x W_E overall", hoyer_sparsity(fourier_1d_q_E))
print("q x W_E p=1", torch.linalg.matrix_norm(fourier_1d_q_E, ord=1) / q_we_overall)
print("q x W_E p=inf", torch.linalg.matrix_norm(fourier_1d_q_E, ord=float('inf')) / q_we_overall)

q_overall = torch.linalg.matrix_norm(q, ord='fro')
print("q overall", hoyer_sparsity(q))
print("q p=1", torch.linalg.matrix_norm(q, ord=1) / q_overall)
print("q p=inf", torch.linalg.matrix_norm(q, ord=float('inf')) / q_overall)

fourier_1d_U = torch.fft.fft(W_U, dim=0, norm="ortho")
fourier_2d_op_U = torch.fft.ifft(fourier_1d_U, dim=1, norm="ortho")
#visualize_2D_matrix(torch.abs(fourier_1d_U.T).cpu().numpy(), "Unembedding Fourier 1D Magnitudes")
#visualize_2D_matrix(torch.abs(fourier_2d_op_U).cpu().numpy(), "Unembedding Fourier 2D Magnitudes")

#visualize_2D_matrix(torch.abs(torch.fft.ifft(q @ W_E, dim=1, norm="ortho").T @ torch.fft.ifft(k @ W_E, dim=1, norm="ortho")).cpu().numpy(), "Query.T @ Key")

attn_out_proj = model.blocks.get_submodule("0.mha").out_proj.weight.data.detach() @ W_E # type: ignore
fourier_1d_O = torch.fft.ifft(attn_out_proj, dim=1, norm="ortho")
#visualize_2D_matrix(torch.abs(fourier_1d_O).cpu().numpy(), "Embedding x Out Fourier 1D Magnitudes")


plt.show()