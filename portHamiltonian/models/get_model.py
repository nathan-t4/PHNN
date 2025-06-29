
from models.node import NODE
from models.nsde import NSDE
from models.phnode import PHNODE
from models.phnsde import PHNSDE
from models.latent_phnode import LatentPHNODE

models = {
    "NODE": NODE,
    "PHNODE": PHNODE,
    "LatentPHNODE": LatentPHNODE,
    "NSDE": NSDE,
    "PHNDSE": PHNSDE,
}

def get_model(name):
    assert name in models
    return models[name]