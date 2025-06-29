import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/store/nt9637/portHamiltonian"
data_dir = os.path.abspath(data_dir)

cases = ["Case_9", "Case_10", "Case_11"]

state_w_path = "del_w.csv"
state_p_path = "p.csv"
input_d_path = "d.csv"
input_pref_path = "pref.csv"

paths = [state_w_path, state_p_path, input_d_path, input_pref_path]

filter_params = {
    'min_time': 0.2,
    'min_dt': 1e-7, # filter out dt=0
    'max_length': 6000000, # make all trajs same length to stack
}

def filter_time_indices(time: np.ndarray, filter_params: dict):
    valid_time_indices = np.arange(len(time))
    if 'min_time' in filter_params.keys():
        valid_time_indices = valid_time_indices[time[valid_time_indices] >= filter_params['min_time']]
    # Filter by min_dt
    if 'min_dt' in filter_params.keys():
        dt = np.diff(time[valid_time_indices])
        invalid_indices = np.where(dt <= filter_params['min_dt'])[0] + 1  # +1 to refer to the second of the pair
        valid_time_indices = np.delete(valid_time_indices, invalid_indices)
    if 'max_length' in filter_params.keys():
        valid_time_indices = valid_time_indices[:filter_params['max_length']]

    return valid_time_indices

# Dictionary to store filtered data for each case
all_cases_data = {}

for case in cases:
    case_data_dir = os.path.join(data_dir, case)
    
    data = []
    times = []
    for i,p in enumerate(paths):
        data.append(
            pd.read_csv(os.path.join(case_data_dir, p))
        )
        times.append(data[i]['Time'])

    assert (times[0].equals(times[1]))
    assert (times[2].equals(times[3]))
    assert (times[0].equals(times[2]))

    w = data[0][data[0].columns.tolist()[1]]
    p = data[1][data[1].columns.tolist()[1]]
    d = data[2][data[2].columns.tolist()[1]]
    pref = data[3][data[3].columns.tolist()[1]]

    time = times[0]
    filtered_time_indices = filter_time_indices(time.to_numpy(), filter_params)

    w_f = w[filtered_time_indices]
    p_f = p[filtered_time_indices]
    d_f = d[filtered_time_indices]
    pref_f = pref[filtered_time_indices]
    t_f = time[filtered_time_indices]

    filtered_data = {
        'time': t_f,
        'w': w_f,
        'p': p_f,
        'd': d_f,
        'pref': pref_f,
    }
    
    all_cases_data[case] = filtered_data

# Stack the data from all cases
stacked_data = {
    'time': np.stack([all_cases_data[case]['time'] for case in cases]), # [trajs, T]
    'w': np.stack([all_cases_data[case]['w'] for case in cases]),
    'p': np.stack([all_cases_data[case]['p'] for case in cases]),
    'd': np.stack([all_cases_data[case]['d'] for case in cases]),
    'pref': np.stack([all_cases_data[case]['pref'] for case in cases]),
}

print("STACKED W SHAPE", stacked_data['w'].shape)

plt.figure(figsize=(10,5))

# Plot w and p for all cases
for i, case in enumerate(cases):
    w = stacked_data['w'][i]
    p = stacked_data['p'][i]
    t = stacked_data['time'][i]
    plt.plot(t, w, label=f'w{i}')
    plt.plot(t, p, label=f'p{i}')

# plt.ylim([-2,2])
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.tight_layout()
plt.savefig(f'preprocess_inverter.png')
plt.close()

# Save both individual case data and stacked data
for case in cases:
    save_file = f"filtered_data_{case}.npy"
    np.save(os.path.join(data_dir, case, save_file), all_cases_data[case])

# Save stacked data
stacked_save_file = "filtered_data_stacked.npy"
np.save(os.path.join(data_dir, stacked_save_file), stacked_data)