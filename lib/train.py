from typing import Any, List, Optional
import torch
from torch.nn import Module, MSELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot, log_softmax
from lib.metrics import calc_l2_norm, calc_grad_norm

def train(
    model: Module, 
    optimizer: Optimizer, 
    criterion: Any, 
    dataloader_train: DataLoader, 
    dataloader_val: DataLoader, 
    epochs: int, 
    device: torch.device, 
    lr_scheduler: Optional[LRScheduler] = None,
    output_stats_every: int | None = 1,
    ):
  loss_train: List[float] = []
  loss_val: List[float] = []
  acc_train: List[float] = []
  acc_val: List[float] = []
  norm: List[float] = []
  grad_norm: List[float] = []

  try:
    for epoch in range(epochs):
      model.train()
      n_train_batches, total_loss_train, n_correct_train, total_n_train, total_grad_norm = 0, 0, 0, 0, 0

      for a, b, label in dataloader_train:
        a, b = a.to(device), b.to(device)
        optimizer.zero_grad()
        output = model(a, b)
        label = label.to(device=device, dtype=torch.long)
        loss = criterion(output, label)
        total_loss_train += loss.item() # type: ignore
        n_train_batches += 1

        n_correct_train += (output.argmax(-1) == label).sum().item()
        total_n_train += a.size(0)

        loss.backward()
        total_grad_norm += calc_grad_norm(model, name_exploded_gradients=True)
        optimizer.step()

      avg_loss_train = total_loss_train/n_train_batches
      loss_train.append(avg_loss_train) 
      total_acc_train = 100.0 * n_correct_train / total_n_train
      acc_train.append(total_acc_train)
      total_norm = calc_l2_norm(model)
      norm.append(total_norm)
      avg_grad_norm = total_grad_norm/n_train_batches
      grad_norm.append(avg_grad_norm)

      n_val_batches, total_loss_val, n_correct_val, total_n_val = 0, 0, 0, 0
      with torch.no_grad():
        model.eval()
        for a, b, label in dataloader_val:
          a, b = a.to(device), b.to(device)
          output = model(a, b)
          label = label.to(device=device, dtype=torch.long)
          loss = criterion(output, label)
          total_loss_val += loss.item() # type: ignore
          n_val_batches += 1

          n_correct_val += (output.argmax(-1) == label).sum().item()
          total_n_val += a.size(0)

      # loss and acc, norm statistics
      avg_loss_val = total_loss_val/n_val_batches
      loss_val.append(avg_loss_val) 
      total_acc_val = 100.0 * n_correct_val / total_n_val
      acc_val.append(total_acc_val)

      # print statistics every 'output_stats_every' epochs
      if output_stats_every is not None and epoch % output_stats_every == 0:
        print(f"epoch: {epoch+1} lr: {optimizer.param_groups[0]['lr']:.0e} -> Loss: {avg_loss_train:.8f}, Acc: {total_acc_train:.2f}%",end=" ---------------- ")
        print(f"Val_Loss: {avg_loss_val:.8f}, Val_Acc: {total_acc_val:.2f}%, Norm: {total_norm:.2f}, Grad Norm: {avg_grad_norm:.6f}")

      if lr_scheduler:
        lr_scheduler.step()

    return loss_train, loss_val, acc_train, acc_val, norm, grad_norm
  except KeyboardInterrupt:
    return loss_train, loss_val, acc_train, acc_val, norm, grad_norm

@torch.no_grad()
def test(
  model: Module, 
  criterion: Any, 
  dataloader_test: DataLoader,
  device: torch.device, 
  output_stats: bool = True
  ):
  model.eval()
  n_batches_test, loss, n_correct,  total_n = 0, 0, 0, 0
  for a, b, label in dataloader_test:
    a, b = a.to(device), b.to(device)
    output = model(a, b)
    label = label.to(device=device, dtype=torch.long)
    loss += criterion(output, label)
    n_batches_test += 1

    n_correct += (output.argmax(-1) == label).sum().item()
    total_n += a.size(0)

  loss_test = loss.item()/n_batches_test # type: ignore
  acc = 100.0 * n_correct / total_n
  if output_stats:
    print(f"Test Loss: {loss_test:.8f}, Test Acc: {acc:.2f}%")
  return loss_test, acc

# https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/helpers.py#L105
def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = log_softmax(logits.to(torch.float32), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss