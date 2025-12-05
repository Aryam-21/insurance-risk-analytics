

import pandas as pd
import numpy as np



def cleaning_data_robust(data, gender_col):
    threshold = 0.8
    data = data[data.columns[data.isnull().mean() < threshold]].copy()
    num_columns = data.select_dtypes(include=np.number).columns
    non_num_columns = data.columns.difference(num_columns)
    if not num_columns.empty:
        data[num_columns] = data[num_columns].fillna(data[num_columns].median())
    numeric_like_cols = ['mmcode', 'UnderwrittenCoverID', 'PolicyID']
    for col in numeric_like_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce') 
            if data[col].dtype.kind in 'if': 
                data[col] = data[col].fillna(data[col].median())
            else: 
                 data[col] = data[col].fillna('Unknown')
        for col in non_num_columns:
            if col in data.columns:
                if col == gender_col:
                    data[col] = data[col].fillna('Not specified')
                else:
                    data[col] = data[col].fillna('Unknown')
        print("--- Missing Values After Cleaning ---")
        print(data.isnull().sum().sort_values(ascending=False).head())
        
        return data 

