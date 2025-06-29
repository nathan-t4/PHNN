from portHamiltonian.models.get_model import *

BASELINE_MODELS = ["NODE", "NSDE"]

def is_baseline_model(model):
    return any([isinstance(model, baseline) for baseline in [NODE, NSDE]])