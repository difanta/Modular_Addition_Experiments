from lib.dataset import createModularAdditionData, randomSplit, CustomDataset
from lib.transformer import Transformer
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW
from lib.train import train, cross_entropy_high_precision
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

module = 113

# Dataset

data = createModularAdditionData(module)
train_dataset, val_dataset, test_dataset = (CustomDataset(data, indices) for indices in randomSplit(data.size(0), fractions=(0.3, 0.7, 0.0), device=data.device))

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset)+1, shuffle=False, pin_memory=True)

# Model

model = Transformer(d_model=128, n_heads=4, mlp_dim_multiplier=4, vocab_size=module, n_layers=1, constant_attn=False).to(device)

a, b, label = (i.to(device) for i in next(iter(train_loader)))
summary(model, input_data=(a, b))

# Training

criterion = cross_entropy_high_precision
optimizer = AdamW(lr=0.001, params=model.parameters(), weight_decay=1.0)
epochs = 30000

loss_train, loss_val, acc_train, acc_val, norm, grad_norm = train(model=model, optimizer=optimizer, criterion=criterion, dataloader_train=train_loader, dataloader_val=val_loader, epochs=epochs, device=device, output_stats_every=1)

PATH = "./data/transformer_clock_1"
choice = input(f"wish to save to '{PATH}'? y for yes, default no: ")

if choice.lower() == "y":
    torch.save(model.state_dict(), PATH+".pt")

    with open(PATH+".json", 'w') as f:
        json.dump({"loss_train": loss_train, "acc_train": acc_train, "loss_val": loss_val, "acc_val": acc_val, "norm": norm, "grad_norm": grad_norm}, f, indent=4)

    print(f"Saved to {PATH+".pt"} and {PATH+".json"}")
else:
    print("Not saved")