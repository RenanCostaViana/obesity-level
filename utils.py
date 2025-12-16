# Importando as bibliotecas
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE

# Classe de OneHotEnconding
class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=['Gender', 'family_history', 'FAVC','SMOKE','SCC','MTRANS']):
        self.OneHotEncoding = OneHotEncoding
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, df, y=None):
        self.encoder.fit(df[self.OneHotEncoding])
        return self

    def transform(self, df):
        df = df.copy()
        encoded = self.encoder.transform(df[self.OneHotEncoding])
        feature_names = self.encoder.get_feature_names_out(self.OneHotEncoding)
        df_encoded = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        df_rest = df.drop(columns=self.OneHotEncoding)
        return pd.concat([df_encoded, df_rest], axis=1)
    
# Classe de OrdinalFeature
class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_feature=None):
        if ordinal_feature is None:
            self.ordinal_feature = ['FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE', 'CALC', 'Obesity']
        else:
            self.ordinal_feature = ordinal_feature
        self.ordinal_encoder = OrdinalEncoder()

    def fit(self, df, y=None):
        self.ordinal_encoder.fit(df[self.ordinal_feature])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.ordinal_feature] = self.ordinal_encoder.transform(df[self.ordinal_feature])
        return df

# Classe de MinMax
class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=['Age', 'Height', 'Weight']):
        self.min_max_scaler = min_max_scaler
        self.scaler = MinMaxScaler()

    def fit(self, df, y=None):
        self.scaler.fit(df[self.min_max_scaler])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.min_max_scaler] = self.scaler.transform(df[self.min_max_scaler])
        return df