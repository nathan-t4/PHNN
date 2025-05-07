
from models.node import NODE
from models.phnode import PHNODE
from models.latent_phnode import LatentPHNODE

models = {
    "NODE": NODE,
    "PHNODE": PHNODE,
    "LatentPHNODE": LatentPHNODE,
}

def get_model(name):
    assert name in models
    return models[name]