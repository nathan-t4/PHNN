from portHamiltonian.datasets.boost_converter.boost_converter import BoostConverterDataset
from portHamiltonian.datasets.toy_distribution_grid.toy_distribution_grid import ToyDistributionGridDataset

datasets = {
    "boost_converter": BoostConverterDataset,
    "toy_distribution_grid": ToyDistributionGridDataset,
}

def get_dataset(name):
    assert name in datasets
    return datasets[name]