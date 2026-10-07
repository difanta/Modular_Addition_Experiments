import torch
from lib.visualization import visualize_2D_matrix
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

module = 113

one_hot_matrix = torch.eye(module)
fourier_1d = torch.fft.fft(one_hot_matrix, dim=0)
fourier_2d_op = torch.fft.ifft(fourier_1d, dim=1)

fourier_2d = torch.fft.fft2(one_hot_matrix)

visualize_2D_matrix(torch.abs(one_hot_matrix).cpu().numpy(), "One Hot Matrix")
visualize_2D_matrix(torch.abs(fourier_1d).cpu().numpy(), "Fourier 1D")
visualize_2D_matrix(torch.abs(fourier_2d_op).cpu().numpy(), "Fourier 2D")
plt.show()