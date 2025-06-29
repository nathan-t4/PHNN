import torch
import numpy as np
from torch.utils.data import Dataset

class MassSpringDataset(Dataset):
    def __init__(self, dir, device, training, min_sequence_length=1, max_sequence_length=1, traj_scale=None, dataset=None):
        self.dir = dir
        self.device = device
        self.training = training
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

        data = np.load(dir, allow_pickle=True)
        self.timesteps = torch.as_tensor(data["timesteps"], device=self.device)
        self.trajectories = torch.as_tensor(data["state_trajectories"], device=self.device)
        self.control_inputs = torch.as_tensor(data["control_inputs"], device=self.device)

        with torch.no_grad():
            self.traj_mean = self.trajectories.mean(dim=(0,1), keepdim=True)
            self.traj_std = self.trajectories.std(dim=(0,1), keepdim=True)
            self.traj_scale = self.traj_std
            if traj_scale is None:
                self.trajectories = self.trajectories / self.traj_scale
        
        print("Data statistics:")
        print(f"Mean: {self.traj_mean}")
        print(f"Std: {self.traj_std}")
        print(f"Scale: {self.traj_scale}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        if self.training:
            sequence_length = self.min_sequence_length
            sequence_length = min(sequence_length + 1, len(self.timesteps))
            start_idx = torch.randint(0, len(self.timesteps) - sequence_length, (), device=self.device)
            end_idx = start_idx + sequence_length

            x0 = self.trajectories[idx, start_idx].clone()
            t = self.timesteps[start_idx:end_idx] - self.timesteps[start_idx]
            y = self.trajectories[idx, start_idx:end_idx]
            u = self.control_inputs[idx, start_idx:end_idx]
            info = torch.tensor([])
            return x0, t, y, u, info

        else:
            x0 = self.trajectories[idx,0].clone()
            t = self.timesteps - self.timesteps[0]
            y = self.trajectories[idx]
            u = self.control_inputs[idx]
            info = torch.tensor([])
            return x0, t, y, u, info