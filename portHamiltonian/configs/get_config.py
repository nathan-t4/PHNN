from portHamiltonian.configs.boost_converter import get_boost_converter_config

configs = {
    "boost_converter": get_boost_converter_config()
}

def get_config(name):
    assert name in configs, f"There is no config for {name}"
    return configs[name]