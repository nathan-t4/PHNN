import os
import torch as th
from ml_collections import ConfigDict

def get_toy_distribution_grid_config():
    config = ConfigDict()
    config.system_name = "toy_distribution_grid"
    config.experiment_name = "OpenLoop"

    config.data = data = ConfigDict()
    data.mode = "open" if config.experiment_name == "OpenLoop" else "closed"
    data.dir = os.path.join(os.path.abspath(os.curdir), "data", config.experiment_name)
    data.sequence_length = 32
    data.scale = True

    config.net_cfg = net_cfg = ConfigDict()
    net_cfg.model_name = "PHNODE"
    net_cfg.hidden_dim = 16
    net_cfg.in_dim = 2
    net_cfg.control_dim = 1

    config.training = training = ConfigDict()
    training.total_epochs = 5000
    training.batch_size = 128
    training.val_interval = 10
    training.learning_rate = 1e-3
    training.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    r = 30
    config.system_matrices = matrices = ConfigDict()
    matrices.R = [[0, 0], [0, 1/r]]
    matrices.B = [[1], [0]]

    return config