from portHamiltonian.datasets.boost_converter.boost_converter import BoostConverterDataset
from portHamiltonian.datasets.toy_distribution_grid.toy_distribution_grid import ToyDistributionGridDataset
from portHamiltonian.datasets.mass_spring.mass_spring import MassSpringDataset
from portHamiltonian.datasets.inverter.inverter import InverterDataset

datasets = {
    "boost_converter": BoostConverterDataset,
    "toy_distribution_grid": ToyDistributionGridDataset,
    "inverter": InverterDataset,
    "mass_spring": MassSpringDataset,
}

def get_dataset(name):
    assert name in datasets
    return datasets[name]