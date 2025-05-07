import os
import torch as th
from ml_collections import ConfigDict

def get_boost_converter_config():
    config = ConfigDict()
    config.system_name = "boost_converter"
    config.experiment_name = "OpenLoop"

    config.data = data = ConfigDict()
    data.mode = "open" if config.experiment_name == "OpenLoop" else "closed"
    data.dir = os.path.join(os.path.abspath(os.curdir), "data", config.experiment_name)
    data.sequence_length = 32
    data.scale = True

    config.net_cfg = net_cfg = ConfigDict()
    net_cfg.model_name = "PHNODE"
    net_cfg.in_dim = 2
    net_cfg.hidden_dim = 16
    net_cfg.control_dim = 1

    config.training = training = ConfigDict()
    training.total_epochs = 50000
    training.batch_size = 128
    training.val_interval = 10
    training.learning_rate = 3e-4
    training.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    r = 30
    config.system_matrices = matrices = ConfigDict()
    matrices.J = None
    matrices.R = [[0, 0], [0, 1/r]]
    matrices.B = [[1], [0]]

    matrices.J_cfg = J_cfg = ConfigDict()
    J_cfg.learn_matrix = True
    J_cfg.constant_parameter = False
    J_cfg.net_cfg = J_net_cfg = ConfigDict()
    J_net_cfg.in_dim = 1
    J_net_cfg.hidden_dim = 16

    matrices.R_cfg = R_cfg = ConfigDict()
    R_cfg.learn_matrix = True
    R_cfg.constant_parameter = True
    R_cfg.net_cfg = J_net_cfg

    matrices.B_cfg = B_cfg = ConfigDict()
    B_cfg.learn_matrix = True
    B_cfg.constant_parameter = True
    B_cfg.net_cfg = J_net_cfg

    # Parameters for learning the system matrices. 
    # Only used when matrices.learn_matrices = True
    matrices.matrix_net_cfg = matrix_net_cfg = ConfigDict()
    matrix_net_cfg.in_dim = 1
    matrix_net_cfg.hidden_dim = 16

    return config