import os
import torch
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler 


def prepare_dataframes(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Prepares two DataFrames from multiple csv files. 
    

    Args:
        root (str): The path to the directory where all the csv files are.
    
    Returns:
        out (tuple[DataFrame, DataFrame]): The first DataFrame contains training data. The second DataFrame contains testing data.
    '''
    csv_file_names = [f for f in os.listdir(ATTACK_DATA_ROOT_PATH) if f.endswith(".csv")]
    csv_file_paths = [os.path.join(ATTACK_DATA_ROOT_PATH, f) for f in csv_file_names]
    dataframes = {os.path.basename(p).split(".")[0]: pd.read_csv(p, index_col=0) for p in csv_file_paths}
    train_dataframe = pd.concat([dataframes[k] for k in dataframes.keys() 
                                 if k.startswith("shadow")], axis=0)
    test_dataframe = pd.concat([dataframes[k] for k in dataframes.keys() 
                                if k.startswith("target")], axis=0)
    
    return train_dataframe, test_dataframe


def get_rnn_dataloaders(root: str, num_metrics: int, num_snapshots: int, batch_size: int=1, 
                        scaler_save_path=None) -> tuple[DataLoader, DataLoader]:
    df_train, df_test = prepare_dataframes(root)

    X_train = df_train.drop("member", axis=1).values
    y_train = df_train["member"].values

    X_test = df_test.drop("member", axis=1).values
    y_test = df_test["member"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if not scaler_save_path is None:
        joblib.dump(scaler, scaler_save_path)

    X_train_reshaped = X_train_scaled.reshape(-1, num_snapshots, num_metrics)
    X_test_reshaped = X_test_scaled.reshape(-1, num_snapshots, num_metrics)

    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).unsqueeze(1)

    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_test = TensorDataset(X_test_tensor, y_test_tensor)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=batch_size,
                                 shuffle=False)

    return dataloader_train, dataloader_test


