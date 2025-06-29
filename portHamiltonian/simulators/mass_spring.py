from portHamiltonian.simulators.base import Simulator

import numpy as np
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import pickle
import os

class MassSpringSimulator(Simulator):
    def __init__(self, parameters: dict, seed):
        self.N = parameters["N"]
        self.m = np.array(parameters["m"])
        self.k = np.array(parameters["k"])
        self.b = np.array(parameters["b"])
        self.initial_condition_ranges = np.array(parameters["initial_condition_ranges"])

        assert len(self.m) == len(self.k) == len(self.b) == self.N
        assert len(self.initial_condition_ranges) == 2

        self._buildPHMatrices()
        self.rng = np.random.default_rng(seed)
    
    def _buildPHMatrices(self):
        idx = np.arange(2 * self.N - 1)
        self.J = np.zeros((2 * self.N, 2 * self.N))
        self.J[idx, idx+1] = 1 * np.ones(self.N-1) if self.N > 1 else 1
        self.J[idx+1, idx] = -1 * np.ones(self.N-1) if self.N > 1 else -1

        def get_R(x):
            R = np.zeros(2 * self.N)
            R[::2] = [b * (x[2*i+1]**2) / (m**2) for i,(b,m) in enumerate(zip(self.b, self.m))]
            return R

        self.R = get_R

        self.B = np.zeros((2 * self.N, self.N))
        for n in range(self.N):
            self.B[2*n + 1, n] = 1
        
        print("J", self.J)
        print("B", self.B)

        def H(x):
            hamiltonian = 0.0
            for n in range(len(x) // 2):
                hamiltonian += (x[2*n]**2 * self.k[n]) / 2 + (x[2*n+1]**2) / (2 * self.m[n])
            return hamiltonian

        def dH(x):
            x = torch.tensor(x, requires_grad=True)
            return torch.autograd.grad(self.H(x), x)[0].reshape(-1,1).numpy()

        self.H = H
        self.dH = dH
    
    def plot(self, data, save):
        trajs = data['state_trajectories']
        timesteps = data['timesteps']
        num_trajs = trajs.shape[0]

        for i in range(num_trajs):
            for n in range(self.N):
                plt.plot(timesteps, trajs[i,:,2*n], label=f"$q_{n}$")
                plt.plot(timesteps, trajs[i,:,2*n+1], label=f"$p_{n}$")
        
        plt.tight_layout()        
        plt.savefig(os.path.join(save, "mass_spring_trajs.png"))

    def getOneTrajectory(self, x0, T, dt, noise_std=0.0):
        timesteps = np.arange(T) * dt
        F_ext = lambda x : np.reshape([b * (x[2*i+1] / m)**3 for i,(b,m) in enumerate(zip(self.b, self.m))], (-1,1)) # F_ext = b * q_dot**3
        dynamics_function = lambda x, t : ((self.J - self.R(x)) @ self.dH(x) + self.B @ F_ext(x)).squeeze()
        noise = noise_std * self.rng.random((T, 2 * self.N))

        state_trajectory = odeint(dynamics_function, x0, timesteps) + noise
        control_input = np.array([F_ext(x) for x in state_trajectory])

        return state_trajectory, control_input

    def generateDataset(self, num_trajs, T, dt, noise_std=0.0, save=None, train_val_test_split=[0.7,0.2,0.1]):
        trajs = []
        control_inputs = []

        valid_trajs = 0

        while valid_trajs <= num_trajs:
            rngs = self.rng.random(2 * self.N)
            x0 = []
            for n in range(self.N):
                idx = np.arange(2*n, 2*(n+1))
                x0.extend([rngs[i] * (r[1] - r[0]) + r[0] for i,r in zip(idx,self.initial_condition_ranges)])
            
            state_trajectory, control_input = self.getOneTrajectory(x0, T, dt, noise_std)

            if not np.any(np.abs(state_trajectory) > 1e2):
                valid_trajs = valid_trajs + 1
                trajs.append(state_trajectory)
                control_inputs.append(control_input)
        
        trajs = np.array(trajs)
        timesteps = np.arange(T) * dt
        control_inputs = np.array(control_inputs)

        train_ind, test_ind = train_test_split(np.arange(num_trajs), test_size=train_val_test_split[1]+train_val_test_split[2])

        val_ind, test_ind = train_test_split(test_ind, test_size=train_val_test_split[2] / (train_val_test_split[1] + train_val_test_split[2]))
        
        train_data = {
            'state_trajectories': trajs[train_ind],
            'timesteps': timesteps,
            'control_inputs': control_inputs[train_ind],
            'm': self.m,
            'b': self.b,
            'k': self.k,
            'N': self.N
        }

        val_data = {
            'state_trajectories': trajs[val_ind],
            'timesteps': timesteps,
            'control_inputs': control_inputs[val_ind],
            'm': self.m,
            'b': self.b,
            'k': self.k,
            'N': self.N
        }

        test_data = {
            'state_trajectories': trajs[test_ind],
            'timesteps': timesteps,
            'control_inputs': control_inputs[test_ind],
            'm': self.m,
            'b': self.b,
            'k': self.k,
            'N': self.N
        }

        if save is not None:
            os.makedirs(save, exist_ok=True)
            filename = os.path.join(save, "train_data.pkl")
            with open(filename, 'wb') as file:
                pickle.dump(train_data, file)

            filename = os.path.join(save, "val_data.pkl")
            with open(filename, 'wb') as file:
                pickle.dump(val_data, file)
            
            filename = os.path.join(save, "test_data.pkl")
            with open(filename, 'wb') as file:
                pickle.dump(test_data, file)
        
        return train_data
        

if __name__ == "__main__":
    seed = 42
    # parameters = {
    #     'N': 2,
    #     'm': [1.0, 1.0],
    #     'k': [1.2, 1.5],
    #     'b': [1.7, 1.7],
    #     'initial_condition_ranges': [[-1.0, 1.0], [-1.0, 1.0]],
    # }

    parameters = {
        'N': 1,
        'm': [1.0],
        'k': [1.2],
        'b': [1.7],
        'initial_condition_ranges': [[-1.0, 1.0], [-1.0, 1.0]],
    }

    num_trajs = 30
    T = 100
    dt = 0.01
    noise_std = 0.0

    save_path = "../data/SingleMassSpring" if parameters["N"] == 1 else "../data/DoubleMassSpring"

    sim = MassSpringSimulator(parameters, seed)
    train_data = sim.generateDataset(num_trajs, T, dt, noise_std, save=save_path)
    sim.plot(train_data, save_path)

    data_file = os.path.join(save_path, "train_data.pkl")
    train_data = np.load(data_file, allow_pickle=True)

    data_file = os.path.join(save_path, "val_data.pkl")
    val_data = np.load(data_file, allow_pickle=True)

    data_file = os.path.join(save_path, "test_data.pkl")
    test_data = np.load(data_file, allow_pickle=True)


    print(train_data['state_trajectories'].shape, val_data['state_trajectories'].shape, test_data['state_trajectories'].shape)
    