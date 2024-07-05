import os
import pandas as pd
from datasets import load_dataset
import pyarrow.ipc as ipc

def download_ms_marco(data_dir):
    dataset = load_dataset("microsoft/ms_marco", "v2.1", ignore_verifications=True)
    #this is deprecate - should use verification_mode=no_checks
    dataset.save_to_disk(data_dir)

def read_arrow_file(file_path):
    with ipc.open_file(file_path) as reader:
        table = reader.read_all()
        df = table.to_pandas()
    return df

def load_data(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith('.arrow'):
            file_path = os.path.join(directory, filename)
            df = read_arrow_file(file_path)
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df