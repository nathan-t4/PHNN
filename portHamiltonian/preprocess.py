import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def openLoop():
    trajs_path = os.path.abspath(os.path.join(os.curdir, "data", "OpenLoop", "Trajectories"))

    data_df = pd.DataFrame()
    T = None

    def filter(df: pd.DataFrame):
        """ Filter by dt """
        to_remove = []
        last_time = 0
        min_dt = 1e-7
        max_timesteps = 3e4

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
        
        return np.array(to_remove)


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
    train_df = data_df
    # train_df, val_df = train_test_split(data_df, test_size=0.3, shuffle=False)
    # val_df, test_df = train_test_split(val_df, test_size=0.33, shuffle=False)

    train_df = train_df.T
    # val_df = val_df.T
    # test_df = test_df.T

    train_df["T"] = T
    # val_df["T"] = T
    # test_df["T"] = T

    # print(f"train {train_df.shape}, val {val_df.shape}, test {test_df.shape}")

    save_path = os.path.join(os.path.abspath(os.curdir), "data")
    save_path = os.path.join(trajs_path, os.path.pardir)

    print(f"Save path: {save_path}")

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
    # val_df.to_pickle(os.path.join(save_path, "val_data.pkl"))
    # test_df.to_pickle(os.path.join(save_path, "test_data.pkl"))

def closedLoop():
    trajs_path = os.path.abspath(os.path.join(os.curdir, "data", "ClosedLoop", "Trajectories"))

    data_df = {}
    T_df = {}

    def filter(df: pd.DataFrame):
        """ Filter by dt """
        filter_idx = []
        last_time = 0
        min_dt = 1e-7
        max_time = 6e-2

        for i,t in enumerate(df):
            if t > max_time:
                break
                
            if abs(t-last_time) < min_dt:
                filter_idx.append(i)
            else:
                last_time = t
        
        return filter_idx

    for i,path in enumerate(os.listdir(trajs_path)):
        idx = re.findall(r'\d+', path)[-1]
        charge_path = os.path.join(trajs_path, path, f"charge{idx}.csv")
        flux_path = os.path.join(trajs_path, path, f"flux{idx}.csv")
        duty_path = os.path.join(trajs_path, path, f"duty{idx}.csv")

        q = pd.read_csv(rf"{charge_path}")
        phi = pd.read_csv(rf"{flux_path}")
        u = pd.read_csv(rf"{duty_path}")

        t_phi = phi.iloc[:,0]
        t_q = q.iloc[:,0]
        t_u = u.iloc[:,0]

        phi = phi.iloc[:,1]
        q = q.iloc[:,1]
        u = u.iloc[:,1]    

        assert(t_q.equals(t_phi) and t_q.equals(t_u)), f"Time indices mismatch for {path}"

        # (x1, x2) = (flux, charge, u)
        old_size = t_q.shape
        filter_idx = filter(t_q)
        data_df[i] = np.array(list(zip(phi, q, u)))[filter_idx]
        T_df[i] = np.array(t_q)[filter_idx]
        print("After removing redundant indices", old_size, "-->", T_df[i].shape)


    # 0.7, 0.2, 0.1 split
    keys = list(data_df.keys())
    train_keys, val_keys = train_test_split(keys, test_size=0.3, shuffle=False)
    val_keys, test_keys = train_test_split(val_keys, test_size=0.33, shuffle=False)

    print(train_keys, val_keys, test_keys)
    print(data_df.keys())

    def get_dataset(keys):
        def get_items(dict, keys):
            return {k : dict[k] for k in keys}
        return {"data": get_items(data_df, keys), "T": get_items(T_df, keys)}

    train_df = get_dataset(train_keys)
    val_df = get_dataset(val_keys)
    test_df = get_dataset(test_keys)

    save_path = os.path.join(os.path.abspath(os.curdir), "data")
    save_path = os.path.join(trajs_path, os.path.pardir)

    print(f"Save path: {save_path}")

    def plot(df: dict, log=False):
        assert(df["data"].keys() == df["T"].keys()), "Keys mismatch"
        for k in df["data"].keys():
            if log:
                phi = np.log(df["data"][k][:,0])
                q = np.log(df["data"][k][:,1])
            else:
                phi = df["data"][k][:,0]
                q = df["data"][k][:,1]
            
            print(k, phi.shape, df["T"][k].shape)

            plt.plot(df["T"][k], phi, label=r"$\phi$")
            plt.plot(df["T"][k], q, label=r"$q$")
            plt.legend()

        plt.savefig(os.path.join(save_path,f"test_plot.png"))
        plt.clf()
        plt.close()
        
    plot(train_df)

    import pickle
    pickle.dump(train_df, open(os.path.join(save_path, "train_data.pkl"), "wb"))
    pickle.dump(val_df, open(os.path.join(save_path, "val_data.pkl"), "wb"))
    pickle.dump(test_df, open(os.path.join(save_path, "test_data.pkl"), "wb"))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="open")

    args = parser.parse_args()

    assert(args.mode in ["open", "closed"])

    if args.mode == "open":
        openLoop()
    else:
        closedLoop()