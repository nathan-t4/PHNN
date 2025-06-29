import os
import pandas as pd
from sklearn.model_selection import train_test_split

def toy_distribution_grid():
    data_dir = os.path.abspath(os.path.join(os.curdir, "data", "ToyDistributionGrid"))
    data_path = os.path.join(data_dir, "qsts_gridconnected_fullresults.csv")
    df = pd.read_csv(data_path)

    # Total load power (sum of all three loads)
    df["Total_Load_P_kW"] = df["Load1_P_kW"] + df["Load2_P_kW"] + df["Load3_P_kW"]
    # Total supply power (PV + ESS)
    df["Total_Supply_P_kW"] = df["PV_P_kW"] + df["ESS_P_kW"]
    # Total reactive power from loads
    df["Total_Load_Q_kVAR"] = df["Load1_Q_kVAR"] + df["Load2_Q_kVAR"] + df["Load3_Q_kVAR"]
    # Total reactive power from PV and ESS
    df["Total_Supply_Q_kVAR"] = df["PV_Q_kVAR"] + df["ESS_Q_kVAR"]

    print(df.head())
    # Freq_Hz is known but constant at 60 so excluding
    known_quantities = ["Time_hr", "Bus1_V_pu", "Bus2_V_pu", "Bus3_V_pu", "PV_P_kW", "ESS_P_kW", "PV_Q_kVAR", "ESS_Q_kVAR"]
    unknown_quantities = [col for col in df.columns if col not in known_quantities]

    print("Known quantities:", known_quantities)
    print("Unknown quantities:", unknown_quantities)

    df.drop(unknown_quantities, axis=1, inplace=True)

    # Split train, test, val sets
    train_df, val_df = train_test_split(df, test_size=0.3, shuffle=False)
    val_df, test_df = train_test_split(df, test_size=0.33, shuffle=False)

    save_path = data_dir
    print(f"Save path: {save_path}")
    train_df.to_pickle(os.path.join(save_path, "train_data.pkl"))
    val_df.to_pickle(os.path.join(save_path, "val_data.pkl"))
    test_df.to_pickle(os.path.join(save_path, "test_data.pkl"))
    

    # === 8. Compare Total Load vs Total Supply (PV + ESS) ===
    
    # plt.figure(figsize=(14, 5))
    # plt.plot(df["Total_hr"], df["Total_Load_P_kW"], label="Total Load Power (kW)", color='black')
    # plt.plot(df["Total_hr"], df["Total_Supply_P_kW"], label="Total PV + ESS Supply (kW)", color='orange')
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Power (kW)")
    # plt.title("Total Load vs PV+ESS Supplied Active Power over 7 Days")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("Plots/plot_total_load_vs_supply.png", dpi=300)
    # plt.close()

    # === 9. Compare Total Reactive Load vs Total Q Supplied (PV + ESS) ===

    # plt.figure(figsize=(14, 5))
    # plt.plot(df["Total_hr"], df["Total_Load_Q_kVAR"], label="Total Load Reactive Power (kVAR)", color='blue')
    # plt.plot(df["Total_hr"], df["Total_Supply_Q_kVAR"], label="Total PV + ESS Reactive Supply (kVAR)", color='green')
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Reactive Power (kVAR)")
    # plt.title("Total Reactive Load vs PV+ESS Supplied Q over 7 Days")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("Plots/plot_total_q_load_vs_supply.png", dpi=300)
    # plt.close()

if __name__ == "__main__":
    toy_distribution_grid()

    import numpy as np
    save_path = os.path.abspath(os.path.join(os.curdir, "data", "ToyDistributionGrid"))
    train_data = np.load(os.path.join(save_path, "train_data.pkl"), allow_pickle=True)

    print("Training data")
    print(train_data)