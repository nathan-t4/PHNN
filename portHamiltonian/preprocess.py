import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

trajs_path = os.path.abspath(os.path.join(os.curdir, "data", "Trajectories"))

data_df = pd.DataFrame()
T = None

def filter(df: pd.DataFrame):
    """ Filter by dt """
    to_remove = []
    last_time = 0
    min_dt = 1e-7
    max_timesteps = 1.5e4

    for i,t in enumerate(df):
        if abs(t-last_time) < min_dt:
            to_remove.append(i)
        else:
            last_time = t
        
        if i > max_timesteps:
            to_remove.append(i)
    
    last_idx = len(df) - 1
    if last_idx > max_timesteps:
        to_remove.append(last_idx)
    
    return to_remove


for i,path in enumerate(os.listdir(trajs_path)):
    idx = re.findall(r'\d+', path)[-1]
    charge_path = os.path.join(trajs_path, path, f"charge_{idx}.csv")
    flux_path = os.path.join(trajs_path, path, f"flux_{idx}.csv")

    q = pd.read_csv(rf"{charge_path}")
    phi = pd.read_csv(rf"{flux_path}")

    t_phi = phi.iloc[:,0]
    t_q = q.iloc[:,0]

    phi = phi.iloc[:,1]
    q = q.iloc[:,1]

    if i == 0:
        T = t_q        

    assert(t_q.equals(t_phi) and t_q.equals(T)), f"Time indices mismatch for {path}"

    # (x1, x2) = (flux, charge)
    data_df[f"{i}"] = list(zip(phi, q))

old_size = data_df.shape
to_remove = filter(T)
data_df = data_df.drop(to_remove, axis=0)

print("After removing redundant indices", old_size, "-->", data_df.shape)

# 0.7, 0.2, 0.1 split
data_df = data_df.T
train_df, val_df = train_test_split(data_df, test_size=0.3, shuffle=False)
val_df, test_df = train_test_split(val_df, test_size=0.33, shuffle=False)

train_df = train_df.T
val_df = val_df.T
test_df = test_df.T

train_df["T"] = T
val_df["T"] = T
test_df["T"] = T

print(f"train {train_df.shape}, val {val_df.shape}, test {test_df.shape}")

save_path = os.path.join(os.path.abspath(os.curdir), "data")

def plot(df: pd.DataFrame, log=False):
    for i,col in enumerate(df):
        if col != "T":
            if log:
                phi = np.log(df[col].str[0])
                q = np.log(df[col].str[1])
            else:
                phi = df[col].str[0]
                q = df[col].str[1]
            plt.plot(df["T"], phi, label=r"$\phi$")
            plt.plot(df["T"], q, label=r"$q$")
            plt.legend()
    plt.savefig(os.path.join(save_path,f"test_plot.png"))
    plt.clf()
    plt.close()
    
plot(train_df)

train_df.to_pickle(os.path.join(save_path, "train_data.pkl"))
val_df.to_pickle(os.path.join(save_path, "val_data.pkl"))
test_df.to_pickle(os.path.join(save_path, "test_data.pkl"))
