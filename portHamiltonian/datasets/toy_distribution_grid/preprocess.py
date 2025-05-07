import os
import pandas as pd

def toy_distribution_grid():
    data_path = os.path.abspath(os.path.join(os.curdir, "..", "..", "data", "ToyDistributionGrid", "qsts_gridconnected_fullresults.csv"))
    data_df = pd.read_csv(data_path)

    print(data_df.head())

    # Filter and split train val test data

if __name__ == "__main__":
    toy_distribution_grid()