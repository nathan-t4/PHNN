import torch
import pandas as pd
from torch.utils.data import Dataset

class PortHamiltonianDataset(Dataset):
    def __init__(self, dir, device, dataset="closed", training=True, min_sequence_length=1, max_sequence_length=1, traj_scale=None):
        """
        Modified to handle full trajectories for ODE integration
        """
        self.dir = dir
        self.device = device
        self.dataset = dataset
        self.training = training
        df = pd.read_pickle(self.dir)

        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.traj_scale = traj_scale
        
        if dataset == "open":
            num_timesteps, num_trajs = df.shape
            control_norm = 15.0 / 35.0
        else:
            num_timesteps = min([len(data) for data in df["data"].values()])
            num_trajs = len(df["data"].keys())
            control_norm = 0.0
        
        num_states = 2
        num_controls = 1

        # Store full trajectories
        self.trajectories = torch.zeros((num_trajs, num_timesteps, num_states), device=self.device)
        self.control = control_norm * torch.ones((num_trajs, num_timesteps, num_controls), device=self.device)

        if self.dataset == "open":            
            # Store time points
            self.times = torch.tensor(df["T"].to_numpy(), device=self.device)
            self.t_scale = self.times[-1] - self.times[0]

            df = df.drop(["T"], axis=1)
            
            # Store each trajectory
            for i, col in enumerate(df):
                self.trajectories[i] = torch.tensor(df[col].tolist(), device=self.device)
            
            
        elif self.dataset == "closed":
            for i, data in enumerate(df["data"].values()):
                self.trajectories[i] = torch.tensor(data[:num_timesteps,:num_states], device=self.device)
                self.control[i] = torch.tensor(data[:num_timesteps,num_states:num_states+num_controls], device=self.device)
            
            self.times = torch.zeros((num_trajs, num_timesteps), device=self.device)
            for i, time in enumerate(df["T"].values()):
                self.times[i] = torch.tensor(time[:num_timesteps], device=self.device)
            
            self.control = 1 - self.control # Required due to data format
            self.t_scale = (self.times[:,-1] - self.times[:,0]).mean(dim=0)
        
        # Normalize if needed
        with torch.no_grad():
            self.traj_mean = self.trajectories.mean(dim=(0,1), keepdim=True)
            self.traj_std = self.trajectories.std(dim=(0,1), keepdim=True)
            if traj_scale is None:
                self.traj_scale = self.traj_std
            self.trajectories = self.trajectories / self.traj_scale

        print("Data statistics:")
        print(f"Mean: {self.traj_mean}")
        print(f"Std: {self.traj_std}")
        print(f"Scale: {self.traj_scale}")
            
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
        times = self.times if self.dataset == "open" else self.times[idx]

        if self.training:
            # Random sequence length for training
            # sequence_length = torch.randint(
            #     self.min_sequence_length,
            #     self.max_sequence_length,
            #     (),
            #     device=self.device
            # )
            sequence_length = self.min_sequence_length
            sequence_length = min(sequence_length + 1, len(times))
            start_idx = torch.randint(0, len(times) - sequence_length, (), device=self.device)
            end_idx = start_idx + sequence_length
            
            x0 = self.trajectories[idx, start_idx].clone()
            t = times[start_idx:end_idx] - times[start_idx]
            y = self.trajectories[idx, start_idx:end_idx]
            u = self.control[idx, start_idx:end_idx]
            
            return x0, t, y, u
        else:
            # For validation, return full trajectory
            x0 = self.trajectories[idx,0].clone()
            t = times - times[0]
            y = self.trajectories[idx]
            u = self.control[idx]
            
            return x0, t, y, u