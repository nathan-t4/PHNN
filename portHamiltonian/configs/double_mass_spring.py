import os
import torch as th
from ml_collections import ConfigDict

def get_double_mass_spring_config():
    config = ConfigDict()
    config.system_name = "2_mass_spring"
    config.experiment_name = "DoubleMassSpring"

    config.data = data = ConfigDict()
    data.mode = None
    data.dir = os.path.join(os.path.abspath(os.curdir), "data", config.experiment_name)
    data.sequence_length = 32
    data.scale = True

    config.net_cfg = net_cfg = ConfigDict()
    net_cfg.model_name = "PHNODE"
    net_cfg.hidden_dim = 16
    net_cfg.control_dim = 1

    config.training = training = ConfigDict()
    training.total_epochs = 5000
    training.batch_size = 128
    training.val_interval = 10
    training.learning_rate = 3e-4
    training.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    b1 = 1.7
    m1 = 1.0
    b2 = 1.7
    m2 = 1.0
    config.system_matrices = matrices = ConfigDict()
    matrices.J = [[0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0]]
    matrices.R = lambda x : [[0, 0, 0, 0], [0, (b1 * (x[1]**2)) / (m1**2), 0, 0], [0, 0, 0, 0], [0, 0, 0, (b2 * (x[3]**2)) / (m2**2)]]
    matrices.B = [[0, 0], [1, 0], [0, 0], [0, 1]]

    matrices.J_cfg = J_cfg = ConfigDict()
    J_cfg.learn_matrix = False
    J_cfg.constant_parameter = True
    J_cfg.net_cfg = J_net_cfg = ConfigDict()
    J_net_cfg.in_dim = 1
    J_net_cfg.hidden_dim = 16

    matrices.R_cfg = R_cfg = ConfigDict()
    R_cfg.learn_matrix = False
    R_cfg.constant_parameter = False # TODO
    R_cfg.net_cfg = J_net_cfg

    matrices.B_cfg = B_cfg = ConfigDict()
    B_cfg.learn_matrix = False
    B_cfg.constant_parameter = True
    B_cfg.net_cfg = J_net_cfg

    # Parameters for learning the system matrices. 
    # Only used when matrices.learn_matrices = True
    matrices.matrix_net_cfg = matrix_net_cfg = ConfigDict()
    matrix_net_cfg.in_dim = 1
    matrix_net_cfg.hidden_dim = 16

    return config