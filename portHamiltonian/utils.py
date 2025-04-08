import torch

def input_interp(t, timepoints, input_vals):
    """Custom forward interpolating function without numpy"""
    
    assert len(input_vals) == len(timepoints)

    # Get list indicating if t <= tp fpr all tp in timepoints
    # t_smaller_time = [1 if t <= tp else 0 for tp in timepoints]
    t_smaller_time = torch.where(t <= timepoints)[0]

    # Return value corresponding to first tp that fulfills t <= tp
    if torch.numel(t_smaller_time) != 0:
        # idx_last_value = t_smaller_time.index(1)
        val_interp = input_vals[t_smaller_time[0]]
    # Return last value if there is no tp that fulfills t <= tp
    else:
        val_interp = input_vals[len(input_vals)-1]
    
    return val_interp