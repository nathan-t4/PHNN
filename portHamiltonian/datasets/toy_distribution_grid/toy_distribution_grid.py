import torch
import pandas as pd
from torch.utils.data import Dataset

class ToyDistributionGridDataset(Dataset):
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

        self.num_timesteps, self.num_states = df.shape

        self.times = torch.tensor(df["Time_hr"].to_numpy(), device=self.device)
        df.drop(["Time_hr"], axis=1, inplace=True)
        self.trajectories = torch.tensor(df.to_numpy(), device=self.device)

        self.t_scale = self.times[1] - self.times[0]
        self.traj_mean = self.trajectories.mean(dim=1)
        self.traj_std = self.trajectories.std(dim=1)

        print(self.t_scale, self.traj_mean, self.traj_std)
        
        # Normalize if needed
        with torch.no_grad():
            if traj_scale is None:
                self.traj_scale = self.traj_std
            self.trajectories = self.trajectories / self.traj_scale

        print("Data statistics:")
        print(f"Mean: {self.traj_mean}")
        print(f"Std: {self.traj_std}")
        print(f"Scale: {self.traj_scale}")
            
    def __len__(self):
        if self.training:
            return self.num_timesteps - self.max_sequence_length
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
            start_idx = torch.randint(0, idx, (), device=self.device)
            end_idx = start_idx + sequence_length
            
            x0 = self.trajectories[start_idx].clone()
            t = times[start_idx:end_idx] - times[start_idx]
            y = self.trajectories[start_idx:end_idx]
            # u = self.control[start_idx:end_idx]
            
            return x0, t, y #, u
        else:
            # For validation, return full trajectory
            x0 = self.trajectories[0].clone()
            t = times - times[0]
            y = self.trajectories
            # u = self.control[idx]
            
            return x0, t, y #, u