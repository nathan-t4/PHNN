import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from portHamiltonian.datasets.inverter.inverter import InverterDataset

torch.set_default_dtype(torch.double)

# Load dataset
device = "cuda" # or "cuda" if you want
data_dir = "/store/nt9637/portHamiltonian"
subsample = 1
sequence_length = 25600
dataset = InverterDataset(data_dir, device, training=True, traj_scale=None, min_sequence_length=sequence_length, max_sequence_length=sequence_length, subsample=subsample)

# Create validation dataset
val_dataset = InverterDataset(data_dir, device, training=False, traj_scale=None, min_sequence_length=sequence_length, max_sequence_length=sequence_length, subsample=subsample)

# Create dataloaders
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Debug: Check dataset size
print(f"Dataset length: {len(dataset)}")
print(f"Number of batches per epoch: {len(dataloader)}")

# Un-normalize if needed for system identification and plotting
traj_scale = dataset.traj_scale if hasattr(dataset, 'traj_scale') else 1.0
traj_mean = dataset.traj_mean if hasattr(dataset, 'traj_mean') else 0.0

# x and u are normalized, so for system ID, keep them as is
# For reporting and plotting, un-normalize

# Prediction Error Method (PEM)
# Define A and B as parameters with A = S + K constraint
n_x = 2
n_u = 2

# Parameterize K (anti-symmetric), and d (diagonal for K)
K_param = torch.randn(n_x, n_x, device=device, requires_grad=True)
K_diag = torch.randn(n_x, device=device, requires_grad=True)
B = torch.randn(n_x, n_u, device=device, requires_grad=True)

optimizer = torch.optim.Adam([K_param, K_diag, B], lr=1e-2)
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x0, t, x, u) in enumerate(dataloader):
        # Construct A = K (anti-symmetric + diagonal)
        A = 0.5 * (K_param - K_param.T) + torch.diag(K_diag)

        batch_size_actual = x0.shape[0]
        sequence_length = x.shape[1]
        
        # Initialize predictions with x0
        x_hat = [x0]  # [batch_size, n_x]
        
        # Predict for each time step
        for i in range(sequence_length - 1):
            dt = t[:, i+1] - t[:, i]  # [batch_size]
            dx_hat = torch.bmm(A.unsqueeze(0).expand(batch_size_actual, -1, -1), 
                              x_hat[-1].unsqueeze(-1)).squeeze(-1) + \
                     torch.bmm(B.unsqueeze(0).expand(batch_size_actual, -1, -1), 
                              u[:, i].unsqueeze(-1)).squeeze(-1)
            x_hat.append(x_hat[-1] + dx_hat * dt.unsqueeze(-1))
        
        x_hat = torch.stack(x_hat, dim=1)  # [batch_size, sequence_length, n_x]
        
        # Un-normalize for loss
        if isinstance(traj_scale, torch.Tensor):
            traj_scale = traj_scale.to(x.device)
        x_true = x * traj_scale
        x_hat_true = x_hat * traj_scale
        
        batch_loss = torch.mean((x_hat_true - x_true)**2)
        epoch_loss += batch_loss.item()
        num_batches += 1
        
        # Backward pass
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if (epoch+1) % 1 == 0 or epoch == 0:
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.12f}")

# Final A matrix
A = 0.5 * (K_param - K_param.T) + torch.diag(K_diag)

print("Identified A (PEM):\n", A.data)
print("Identified B (PEM):\n", B.data)

# Plot a sample trajectory
with torch.no_grad():
    # Get a single trajectory from validation dataloader for plotting
    val_batch = next(iter(val_dataloader))
    x0, t, x, u = val_batch
    
    # Construct A matrix for prediction
    A = 0.5 * (K_param - K_param.T) + torch.diag(K_diag)
    
    # Predict
    x_hat = [x0]
    for i in range(x.shape[1] - 1):
        dt = t[:, i+1] - t[:, i]
        dx_hat = torch.bmm(A.unsqueeze(0).expand(x0.shape[0], -1, -1), x_hat[-1].unsqueeze(-1)).squeeze(-1) + \
                 torch.bmm(B.unsqueeze(0).expand(x0.shape[0], -1, -1), u[:, i].unsqueeze(-1)).squeeze(-1)
        x_hat.append(x_hat[-1] + dx_hat * dt.unsqueeze(-1))
    x_hat = torch.stack(x_hat, dim=1)
    
    # Un-normalize for plotting
    x_true = x * traj_scale
    x_hat_true = x_hat * traj_scale
    
    # Plot first trajectory in the batch
    plt.figure(figsize=(10,5))
    plt.plot(t[0].cpu(), x_true[0].cpu(), label="true")
    plt.plot(t[0].cpu(), x_hat_true[0].cpu(), label="pred", linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.title('System Identification (PEM): True vs Predicted States (Validation)')
    plt.savefig("sys_id_pem.png")
    plt.close()

