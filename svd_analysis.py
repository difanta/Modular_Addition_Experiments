import json
import torch
from torch.utils.data import DataLoader
from lib.dataset import CustomDataset, createModularAdditionData, randomSplit
from lib.utility import get_best_accelerator
from lib.transformer import Transformer, remap_and_load_state_dict
from lib.visualization import visualize_2D_matrix
import matplotlib.pyplot as plt
from lib.utility import get_dft_basis, get_dht_basis

def match_lowrank_disordered_basis_with_known_ordered_one(known, lowrank, name=None):
    rows_energy = torch.linalg.vector_norm(known.mH @ lowrank, dim=1, ord=2) ** 2
    ordered_indices = torch.argsort(rows_energy, descending=True)

    k = lowrank.size(1)
    if name is not None:
        print(f"\n{name} basis top-{k} matched components index and energy: \n\t", ordered_indices[:k], rows_energy[ordered_indices[:k]])
    return known[:, ordered_indices[:k]]

def calc_fraction_energy_explained(S, k):
    return torch.sum((S[:k]**2) / torch.sum(S**2)).item()

def k_from_energy_ratio(S, min_fraction_energy_explained=0.95):
    N = S.size(0)
    cumulated_energy = torch.tensor(0.0, device=S.device, dtype=S.dtype)
    total_energy = torch.sum(S**2)

    for i in range(N):
        cumulated_energy += (S[i] ** 2) / total_energy
        if cumulated_energy >= min_fraction_energy_explained:
            return i+1, cumulated_energy.item()
    return N, cumulated_energy.item()

def effective_rank(S):
    probs = (S**2)/torch.sum(S**2)
    entropy = torch.distributions.Categorical(probs=probs).entropy()
    eff_rank = torch.exp(entropy)
    return torch.ceil(eff_rank).to(dtype=torch.long).item()

def print_singular_values(S, k, all=False):
    N = S.size(0)
    for i in range(N if all else min(k+1, N)):
        print(f"{i}: {S[i]:.4f}")
        if i==k-1:
            print("---")

def select_lowrank(U,S,Vh,k):
    return U[:, :k], S[:k], Vh[:k, :]

def basis_similarity(left, right):
    k = float(min(left.size(1), right.size(1)))
    return ((torch.linalg.matrix_norm(left.mH @ right, ord='fro') ** 2) / k).item()

def lowrank_svd(matrix):
    U,S,Vh = torch.linalg.svd(matrix)
    k = effective_rank(S)

    U_k,S_k,Vh_k = select_lowrank(U,S,Vh,k)
    return U_k, S_k, Vh_k, k, calc_fraction_energy_explained(S, k)

device = get_best_accelerator()
print(device)

module = 113

# Dataset

data = createModularAdditionData(module)
train_dataset, val_dataset, test_dataset = (CustomDataset(data, indices) for indices in randomSplit(data.size(0), fractions=(0.0, 1.0, 0), device=data.device))

val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True)

# Model Load saved

model = Transformer(d_model=128, n_heads=4, mlp_dim_multiplier=4, vocab_size=module, n_layers=1, constant_attn=False).to(device)

PATH = "./data/transformer_clock"

remap_and_load_state_dict(model, PATH+".pt", device, weights_only=True, strict=True)
model.eval()

results = None
with open(PATH+".json", 'r') as f:
    results = json.load(f)

a, b, label = (i.to(device) for i in next(iter(val_loader)))
model(a, b)

W_E = model.embedding.weight.data.T
W_U = model.unembed.weight.data
print(W_E.shape, W_U.shape)
attn_in_proj = model.blocks.get_submodule("0.mha").in_proj.weight.data # type: ignore
q, k, v = attn_in_proj[:128], attn_in_proj[128:256], attn_in_proj[256:384] # type: ignore

attn_out_proj = model.blocks.get_submodule("0.mha").out_proj.weight.data # type: ignore

U_we,S_we,Vh_we,k_we, fract_we = lowrank_svd(W_E)
U_q,S_q,Vh_q,k_q, fract_q = lowrank_svd(q)
U_k,S_k,Vh_k,k_k, fract_k = lowrank_svd(k)
U_v,S_v,Vh_v,k_v, fract_v = lowrank_svd(v)
U_o,S_o,Vh_o,k_o, fract_o = lowrank_svd(attn_out_proj)
U_wu,S_wu,Vh_wu,k_wu, fract_wu = lowrank_svd(W_U)

print("\nSingular values - matrix, effective rank k, fve, first k singular values:")
print("\nW_E", k_we, fract_we, S_we)
print("\nQuery", k_q, fract_q, S_q)
print("\nKey", k_k, fract_k, S_k)
print("\nValue", k_v, fract_v, S_v)
print("\nOut Proj", k_o, fract_o, S_o)
print("\nW_U", k_wu, fract_wu, S_wu)

matched_dft_Vh = match_lowrank_disordered_basis_with_known_ordered_one(get_dft_basis(Vh_wu.mH.size(0), device=Vh_wu.device), Vh_wu.mH.to(dtype=torch.complex64), name="Fourier")
matched_dft_U = match_lowrank_disordered_basis_with_known_ordered_one(get_dft_basis(module, device=U_wu.device), U_wu.to(dtype=torch.complex64), name="Fourier")

matched_dht_Vh = match_lowrank_disordered_basis_with_known_ordered_one(get_dht_basis(Vh_wu.mH.size(0), device=Vh_wu.device), Vh_wu.mH, name="Hartley")
matched_dht_U = match_lowrank_disordered_basis_with_known_ordered_one(get_dht_basis(module, device=U_wu.device), U_wu, name="Hartley")

matched_canonical_Vh = match_lowrank_disordered_basis_with_known_ordered_one(torch.eye(Vh_wu.mH.size(0), device=Vh_wu.device, dtype=Vh_wu.dtype), Vh_wu.mH, name="Canonical")
matched_canonical_U = match_lowrank_disordered_basis_with_known_ordered_one(torch.eye(module, device=U_wu.device, dtype=U_wu.dtype), U_wu, name="Canonical")

print("\nSimilarity: Fourier Transform, Vh of W_U", basis_similarity(matched_dft_Vh, Vh_wu.mH.to(dtype=torch.complex64)))
print("Similarity: Fourier Transform, U of W_U", basis_similarity(matched_dft_U, U_wu.to(dtype=torch.complex64)))

print("\nSimilarity: Hartley Transform, Vh of W_U", basis_similarity(matched_dht_Vh, Vh_wu.mH))
print("Similarity: Hartley Transform, U of W_U", basis_similarity(matched_dht_U, U_wu))

print("\nSimilarity: Canonical, Vh of W_U", basis_similarity(matched_canonical_Vh, Vh_wu.mH))
print("Similarity: Canonical, U of W_U", basis_similarity(matched_canonical_U, U_wu))

print("\nSimilarity: W_E, q", basis_similarity(U_we, Vh_q.mH))
print("Similarity: W_E, k", basis_similarity(U_we, Vh_k.mH))
print("Similarity: W_E, v", basis_similarity(U_we, Vh_v.mH))

print("\nSimilarity: q, k", basis_similarity(Vh_q.mH, Vh_k.mH))
print("Similarity: q, v", basis_similarity(Vh_q.mH, Vh_v.mH))
print("Similarity: k, v", basis_similarity(Vh_k.mH, Vh_v.mH))

print("\nSimilarity: W_E, o", basis_similarity(U_we, Vh_o.mH))
print("Similarity: o, W_U", basis_similarity(U_o, Vh_wu.mH))
print("Similarity: W_E, W_U input vs output", basis_similarity(Vh_we.mH, U_wu))
print("Similarity: W_E, W_U output vs input", basis_similarity(U_we, Vh_wu.mH))

visualize_2D_matrix(matched_dht_Vh.cpu().numpy(), "\nDHT matched with Vh of W_E, the input space basis")
#visualize_2D_matrix(matched_dht_U.cpu().numpy(), "\nDHT matched with U of W_E, the input space basis")

#visualize_2D_matrix(matched_dft_Vh.abs().cpu().numpy(), "\nDFT matched with Vh of W_E, the input space basis")
#visualize_2D_matrix(matched_dft_U.abs().cpu().numpy(), "\nDFT matched with U of W_E, the input space basis")

visualize_2D_matrix(matched_canonical_Vh.cpu().numpy(), "\nCanonical matched with Vh of W_E, the input space basis")
#visualize_2D_matrix(matched_canonical_U.cpu().numpy(), "\nCanonical matched with U of W_E, the input space basis")

visualize_2D_matrix(Vh_we.mH.cpu().numpy(), "\nVh of W_E, the input space basis")
#visualize_2D_matrix(U_we.cpu().numpy(), "U of W_E, the output space basis")

#visualize_2D_matrix(Vh_wu.mH.cpu().numpy(), "\nVh of W_U, the input space basis")
#visualize_2D_matrix(U_wu.cpu().numpy(), "U of W_U, the output space basis")

plt.show()