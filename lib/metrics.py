from torch.nn import Module
import torch 
import math

def fraction_variance_explained(source, target):
    dot_prod = (source * target).sum(dim=-1, keepdim=True)
    target_sq_norm = (target * target).sum(dim=-1, keepdim=True)
    proj = (dot_prod / target_sq_norm) * target

    return torch.linalg.norm(proj, dim=-1)**2 / torch.linalg.norm(target, dim=-1)**2

@torch.no_grad()
def calc_l2_norm(model: Module):
  """
  calculate the total l2 norm
  """
  return math.sqrt(sum((p ** 2).sum().item() for p in model.parameters()))


@torch.no_grad()
def calc_grad_norm(model: Module, name_exploded_gradients=False):
  """
  calculate the total grad norm, to call between loss.backwards() and optmizer.step()
  """
  total_norm = 0.0
  for name, p in model.named_parameters():
      if p.grad is not None:
          grad_norm = p.grad.data.norm(p=2).item() ** 2
          total_norm += grad_norm
          if name_exploded_gradients and grad_norm > 1.0:
            print(f"{name}: {grad_norm:.6f}")
  return total_norm ** 0.5