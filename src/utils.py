import os
import pandas as pd
from datetime import datetime

def _read_file(filepath, **kwargs):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, **kwargs)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath, **kwargs)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath, **kwargs)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath, **kwargs)
    elif filepath.endswith('.feather'):
        return pd.read_feather(filepath, **kwargs)
    elif filepath.endswith('.hdf'):
        return pd.read_hdf(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format : {filepath}")
    
def read_seed(filepath, x_label, y_label, **kwargs):
    df = _read_file(filepath, **kwargs)
    if x_label in df.columns and y_label in df.columns:
        df = df.rename(columns={x_label:'x_label',y_label:'y_label'})
        return df
    else:
        raise ValueError(f"Incorrect x_label/y_label column name")
    
def read_unlabeled(filepath, x_label, **kwargs):
    df = _read_file(filepath, **kwargs)
    if x_label in df.columns:
        df = df.rename(columns={x_label:'x_label'})
        return df
    else:
        raise ValueError(f"Incorrect x_label column name")
    
def read_from_folder(folder_name, dir = 'cache'):
    dir = os.path.join(dir, folder_name)
    seed = os.path.join(dir, 'seed.parquet')
    unlabeled = os.path.join(dir, 'unlabeled.parquet')
    return _read_file(seed), _read_file(unlabeled)
    
def save_to_folder(seed, unlabeled, dir = 'cache', folder_name = None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dir_name = f"{timestamp}" if folder_name == None else f"{folder_name}"
    unique_dir = os.path.join(dir, dir_name)
    print(f'Saved to {unique_dir}')
    os.makedirs(unique_dir, exist_ok=True)

    # Save the dataframes in parquet format
    seed.to_parquet(os.path.join(unique_dir, 'seed.parquet'), index=False)
    unlabeled.to_parquet(os.path.join(unique_dir, 'unlabeled.parquet'), index=False)
    