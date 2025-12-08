import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder

class Processor:
    def __init__(self):
        pass
    def encoder(self, data, columns_label=[], columns_onehot=[]):
        df = data.copy()

        for col in columns_label:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        if columns_onehot:
            for col in columns_onehot:
                df[col] = df[col].astype(str)
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = ohe.fit_transform(df[columns_onehot])
            encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(columns_onehot))
            df = df.drop(columns=columns_onehot)
            df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

        return df
    def scaler(self, method, df, columns):
        df = df.copy()
        cols = [c for c in columns if c in df.columns]
        if len(cols) == 0:
            print("⚠ No numeric columns found to scale.")
            return df
        if method == "standardScaler":
            scaler = StandardScaler()
        elif method == "minMaxScaler":
            scaler = MinMaxScaler()
        elif method == "npLog":
            df[columns] = np.log(df[columns] + 1)
            return df
        else:
            return df

        df[columns] = scaler.fit_transform(df[columns])
        return df
    def convert_dates(self, data, date_columns):
        df = data.copy()
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
            df[col] = (df[col] - pd.Timestamp("1970-01-01")).dt.days
        return df