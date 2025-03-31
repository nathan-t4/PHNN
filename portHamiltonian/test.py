import itertools
import pandas as pd
import torch
import os

dir = os.path.join(os.path.abspath(os.curdir), "data", "processed_data.pkl")
df = pd.read_pickle(dir)
        
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a,b))

num_timesteps, num_trajs = df.shape
num_trajs -= 1 
num_states = 2

# ( T, x_{t}, x_{t_next} )
# ( num_timesteps, num_trajs, 2, 2 )
data = torch.zeros(num_timesteps-1, num_trajs, num_states, num_states)
T = torch.tensor(df["T"], dtype=torch.float32)

df = df.drop(["T"], axis=1)
for i,col in enumerate(df):
    pairs = torch.tensor(pairwise(df[col]))
    data[:,i] = pairs

torch.set_printoptions(precision=8)
print(data[0,0], T[0], T[1])

print(data.flatten(0,1).shape)

