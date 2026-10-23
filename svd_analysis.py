import json
import torch
from torch.utils.data import DataLoader
from lib.dataset import CustomDataset, createModularAdditionData, randomSplit
from lib.train import get_best_accelerator
from lib.transformer import Transformer, remap_and_load_state_dict

def svd(matrix):
    return torch.linalg.svd(matrix)

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
    print(eff_rank)
    return torch.ceil(eff_rank).to(dtype=torch.long).item()

def print_singular_values(S, k, all=False):
    N = S.size(0)
    for i in range(N if all else min(k+1, N)):
        print(f"{i}: {S[i]:.4f}")
        if i==k-1:
            print("---")

def select_lowrank(U,S,Vh,k):
    return U[:, :k], S[:k], Vh[:k, :]

def basis_alignment(left, right):
    k_left, k_right = left.size(1), right.size(1)
    return ((torch.linalg.matrix_norm(left.mH @ right, ord="fro") ** 2) / float(min(k_left, k_right))).item()

def lowrank_svd(matrix):
    U,S,Vh = svd(matrix)
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
attn_proj = model.blocks.get_submodule("0.mha").in_proj.weight.data # type: ignore
q, k, v = attn_proj[:128], attn_proj[128:256], attn_proj[256:384] # type: ignore

U_we,S_we,Vh_we,k_we, fract_we = lowrank_svd(W_E)
U_q,S_q,Vh_q,k_q, fract_q = lowrank_svd(q)
U_k,S_k,Vh_k,k_k, fract_k = lowrank_svd(k)
U_v,S_v,Vh_v,k_v, fract_v = lowrank_svd(v)

print("W_E", k_we, fract_we, S_we)
print("Query", k_q, fract_q, S_q)
print("Key", k_k, fract_k, S_k)
print("Value", k_v, fract_v, S_v)

print("W_E, q", basis_alignment(U_we, Vh_q.mH))
print("W_E, k", basis_alignment(U_we, Vh_k.mH))
print("W_E, v", basis_alignment(U_we, Vh_v.mH))

print("q, k", basis_alignment(Vh_q.mH, Vh_k.mH))
print("q, v", basis_alignment(Vh_q.mH, Vh_v.mH))
