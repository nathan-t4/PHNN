import torch
import numpy as np
import matplotlib.pyplot as plt
from portHamiltonian.datasets.inverter.inverter import InverterDataset

# Load dataset
device = "cuda" # or "cuda" if you want
data_dir = "/store/nt9637/portHamiltonian"
subsample = 1
dataset = InverterDataset(data_dir, device, training=False, traj_scale=None, subsample=subsample)

# Get full trajectory
x0, t, x, u, _ = dataset[1]  # x: [T, 2], u: [T, 2]

# Estimate dx/dt using finite differences
dt = torch.diff(t)
print("Min dt:", dt.min().item(), "Max dt:", dt.max().item())
print("Any zero dt?", (dt == 0).any().item())

# Un-normalize if needed for system identification and plotting
traj_scale = dataset.traj_scale if hasattr(dataset, 'traj_scale') else 1.0
traj_mean = dataset.traj_mean if hasattr(dataset, 'traj_mean') else 0.0

# x and u are normalized, so for system ID, keep them as is
# For reporting and plotting, un-normalize

# Compute dx/dt
dx = (x[1:] - x[:-1]) / dt.unsqueeze(-1)  # [T-1, 2]
x_mid = x[:-1]  # [T-1, 2]
u_mid = u[:-1]  # [T-1, 2]

# Stack for least squares: dx = [A B] [x; u]
XU = torch.cat([x_mid, u_mid], dim=1)  # [T-1, 4]
DX = dx  # [T-1, 2]

# Solve least squares: theta = (XU^T XU)^{-1} XU^T DX
theta = torch.linalg.lstsq(XU, DX).solution # [4, 2]
A = theta[:2]  # [2, 2]
B = theta[2:]  # [2, 2]

print("Identified A:\n", A)
print("Identified B:\n", B)

# Predict x_hat using the identified model
x_hat = [x0]
# length = len(u) - 1
length = 1000 - 1
for i in range(length):
    dx_hat = A @ x_hat[-1] + B @ u[i]
    x_hat.append(x_hat[-1] + dx_hat * dt[i])
x_hat = torch.stack(x_hat)[:length]

# Un-normalize for MSE and plotting
if isinstance(traj_scale, torch.Tensor):
    traj_scale = traj_scale.to(x.device)
x_true = x * traj_scale
x_hat_true = x_hat * traj_scale

t = t[:length]
x_true = x_true.squeeze()[:length]

# Compute MSE in original coordinates
mse = torch.mean((x_hat_true - x_true)**2)
print("MSE(x, x_hat):", mse.item())

# Plot
plt.figure(figsize=(10,5))
plt.plot(t.cpu(), x_true.squeeze().cpu(), label="true")
plt.plot(t.cpu(), x_hat_true.squeeze().cpu(), label="pred", linestyle='--')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.title('System Identification: True vs Predicted States')
plt.savefig("sys_id.png")
plt.close()

