from configs.boost_converter import get_boost_converter_config
from configs.toy_distribution_grid import get_toy_distribution_grid_config

configs = {
    "boost_converter": get_boost_converter_config(),
    "toy_distribution_grid": get_toy_distribution_grid_config(),
}

def get_config(name):
    assert name in configs, f"There is no config for {name}"
    return configs[name]