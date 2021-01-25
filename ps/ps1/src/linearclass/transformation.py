import pandas as pd
import numpy as np


def main(csv_path, save_path):
    df = pd.read_csv(csv_path)

    df_log = log_transformation(df)
    df_log.to_csv(save_path.replace("TYPE", "log"))


def log_transformation(df):
    df["x_2"] = np.log(df["x_2"])
    return df


if __name__ == "__main__":
    main(csv_path="ds1_train.csv", save_path="ds1_train_TYPE.csv")
    main(csv_path="ds1_valid.csv", save_path="ds1_valid_TYPE.csv")

