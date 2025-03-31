import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PortHamiltonianDataset(Dataset):
    def __init__(self, dir, device, training=True, min_sequence_length=32, max_sequence_length=128):
        """
        Modified to handle full trajectories for ODE integration
        """
        self.dir = dir
        self.device = device
        self.training = training
        df = pd.read_pickle(self.dir)

        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        
        num_timesteps, num_trajs = df.shape
        num_states = 2
        
        # Store full trajectories
        self.trajectories = torch.zeros((num_trajs, num_timesteps, num_states), device=self.device)
        
        # Store time points
        self.times = torch.tensor(df["T"].to_numpy(), device=self.device)

        df = df.drop(["T"], axis=1)
        
        # Store each trajectory
        for i, col in enumerate(df):
            self.trajectories[i] = torch.tensor(df[col].tolist(), device=self.device)
        
        # Normalize if needed
        with torch.no_grad():
            self.traj_mean = self.trajectories.mean(dim=(0,1), keepdim=True)
            self.traj_std = self.trajectories.std(dim=(0,1), keepdim=True)
        #     self.trajectories = (self.trajectories - self.traj_mean) / self.traj_std

        print("Data statistics:")
        print(f"Mean: {self.traj_mean}")
        print(f"Std: {self.traj_std}")
            
    def __len__(self):
        if self.training:
            return len(self.trajectories)
        else:
            return 1  # For validation, return full trajectory
    
    def __getitem__(self, idx):
        """
        Returns:
            x0: initial condition
            t: time points for integration
            y: target trajectory
        """
        if self.training:
            # Random sequence length for training
            # sequence_length = torch.randint(
            #     self.min_sequence_length,
            #     self.max_sequence_length,
            #     (),
            #     device=self.device
            # )
            sequence_length = self.min_sequence_length
            sequence_length = min(sequence_length + 1, len(self.times))
            start_idx = torch.randint(0, len(self.times) - sequence_length, (), device=self.device)
            end_idx = start_idx + sequence_length
            
            x0 = self.trajectories[idx, start_idx].clone()
            t = self.times[start_idx:end_idx] - self.times[start_idx]
            y = self.trajectories[idx, start_idx:end_idx]
            
            return x0, t, y
        else:
            # For validation, return full trajectory
            x0 = self.trajectories[idx,0].clone()
            t = self.times - self.times[0]
            y = self.trajectories[idx]
            
            return x0, t, y