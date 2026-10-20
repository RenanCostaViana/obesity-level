"""Transformadores do pipeline de classificação de nível de obesidade.

Ficam em um módulo separado (e não no notebook) para que o modelo salvo
com joblib possa ser carregado pelo app do Streamlit.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

# Variáveis binárias e nominais: viram colunas 0/1
COLUNAS_ONE_HOT = ['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC', 'MTRANS']

# Variáveis ordinais: a ordem das categorias é definida explicitamente
# (o OrdinalEncoder, sem isso, ordenaria em ordem alfabética)
ORDEM_ORDINAIS = {
    'FCVC': ['Raramente', 'Às Vezes', 'Sempre'],
    'NCP':  ['Uma', 'Duas', 'Três', 'Quatro ou mais'],
    'CAEC': ['Não', 'Às Vezes', 'Frequentemente', 'Sempre'],
    'CH2O': ['<1 L/dia', '1-2 L/dia', '> 2 L/dia'],
    'FAF':  ['Nenhuma', '1-2x/semana', '3-4x/semana', '5-7x/semana'],
    'TUE':  ['0-2h/dia', '3-5h/dia', 'mais que 5h/dia'],
    'CALC': ['Não Bebe', 'Às Vezes', 'Frequentemente', 'Sempre'],
}

# Variáveis numéricas contínuas
COLUNAS_MIN_MAX = ['Age', 'Height', 'Weight']

# Ordem das classes do alvo, da mais leve para a mais grave
ORDEM_CLASSES = ['Abaixo do Peso', 'Peso Normal', 'Sobrepeso I', 'Sobrepeso II',
                 'Obesidade I', 'Obesidade II', 'Obesidade III']


class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=None):
        self.OneHotEncoding = OneHotEncoding

    def fit(self, df, y=None):
        self.colunas_ = self.OneHotEncoding or COLUNAS_ONE_HOT
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder_.fit(df[self.colunas_])
        return self

    def transform(self, df):
        df = df.copy()
        encoded = self.encoder_.transform(df[self.colunas_])
        nomes = self.encoder_.get_feature_names_out(self.colunas_)
        df_encoded = pd.DataFrame(encoded, columns=nomes, index=df.index)
        return pd.concat([df_encoded, df.drop(columns=self.colunas_)], axis=1)


class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_feature=None):
        self.ordinal_feature = ordinal_feature

    def fit(self, df, y=None):
        ordem = self.ordinal_feature or ORDEM_ORDINAIS
        self.colunas_ = list(ordem.keys())
        self.encoder_ = OrdinalEncoder(categories=[ordem[c] for c in self.colunas_])
        self.encoder_.fit(df[self.colunas_])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.colunas_] = self.encoder_.transform(df[self.colunas_])
        return df


class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=None):
        self.min_max_scaler = min_max_scaler

    def fit(self, df, y=None):
        self.colunas_ = self.min_max_scaler or COLUNAS_MIN_MAX
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(df[self.colunas_])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.colunas_] = self.scaler_.transform(df[self.colunas_])
        return df
