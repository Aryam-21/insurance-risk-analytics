import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder

class Processor:
    def __init__(self):
        pass
    def label_encoder(self, df, columns):
        data = df.copy()
        for col in columns:
            le = LabelEncoder()
            data[col]= le.fit_transform(data[col].astype(str))
        return data
    def one_hot_encoder(self, df, columns):
        data = df.copy()
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded= ohe.fit_transform(data[columns])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(columns))
        data = data.drop(columns, axis=1)
        data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)

        return data
    def scaler(self, method, df, columns):
        df = df.copy()

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