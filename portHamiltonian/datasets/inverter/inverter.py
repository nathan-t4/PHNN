import os
import torch
import numpy as np
from torch.utils.data import Dataset

class InverterDataset(Dataset):
    def __init__(self, dir, device, dataset="closed", training=True, min_sequence_length=1, max_sequence_length=1, traj_scale=None, subsample=100):
        """
        Dataset for inverter data with stacked cases
        """
        self.dir = dir
        self.device = device
        self.dataset = dataset
        self.training = training
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.traj_scale = traj_scale

        # Load the stacked data
        data = np.load(os.path.join(dir, "filtered_data_stacked.npy"), allow_pickle=True).item()
        
        # Convert to torch tensors
        self.times = torch.tensor(data['time'], device=device) # [num_trajs, T]
        self.w = torch.tensor(data['w'], device=device)
        self.p = torch.tensor(data['p'], device=device)
        self.d = torch.tensor(data['d'], device=device)
        self.pref = torch.tensor(data['pref'], device=device)

        # Stack states and controls
        self.trajectories = torch.stack([self.w, self.p], dim=2)[:, ::subsample, :] # [num_trajs, T, num_states]
        self.controls = torch.stack([self.d, self.pref], dim=2)[:, ::subsample, :] # [num_trajs, T, num_controls]
        self.times = self.times[:, ::subsample] # [num_trajs, T]]

        self.num_trajs, self.num_timesteps, self.num_states = self.trajectories.shape

        print("Times shape", self.times.shape)
        print("Trajectories shape", self.trajectories.shape)
        print("Controls shape", self.controls.shape)
        

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
            return self.num_trajs * (self.num_timesteps - self.max_sequence_length) # number of possible trajectories of length 
        else:
            return 1  # For validation, return full trajectory
    
    def __getitem__(self, idx):
        """
        Returns:
            x0: initial condition
            t: time points for integration
            y: target trajectory
            u: control inputs
            info: additional information (duty cycle)
        """
        if self.training:
            # Map idx to trajectory and starting position
            # idx = trajectory_idx * (num_timesteps - max_sequence_length) + start_pos
            max_sequences_per_traj = self.num_timesteps - self.max_sequence_length
            trajectory_idx = idx // max_sequences_per_traj
            start_pos = idx % max_sequences_per_traj
            
            # Ensure we don't go out of bounds
            trajectory_idx = min(trajectory_idx, self.num_trajs - 1)
            start_pos = min(start_pos, self.num_timesteps - self.max_sequence_length)
            
            # Use fixed sequence length for batching
            sequence_length = self.max_sequence_length
            end_pos = start_pos + sequence_length
            
            x0 = self.trajectories[trajectory_idx, start_pos].clone()
            t = self.times[trajectory_idx, start_pos:end_pos] - self.times[trajectory_idx, start_pos]
            y = self.trajectories[trajectory_idx, start_pos:end_pos]
            u = self.controls[trajectory_idx, start_pos:end_pos]
            # info = []
            
            return x0, t, y, u # , info
        else:
            # For validation, return full trajectory
            x0 = self.trajectories[idx][0].clone()
            t = self.times[idx] - self.times[idx][0]
            y = self.trajectories[idx]
            u = self.controls[idx]
            # info = None
            
            return x0, t, y, u #, info