# Importando as bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import OneHotEncodingNames, OrdinalFeature, MinMax
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

st.write('# Avaliação de nível de obesidade do paciente')

# Carregando os dados
dados = pd.read_csv(
    'https://raw.githubusercontent.com/RenanCostaViana/PosTech-DataAnalytics/refs/heads/main/obesity-level/df_clean.csv'
)

# Entradas do usuário
st.write("### Gênero")
input_genero = st.radio('Qual é o seu gênero biológico?', ['Masculino', 'Feminino'])

st.write("### Idade")
input_idade = int(st.number_input('Digite a idade do paciente', 0))

st.write("### Altura")
input_altura= float(st.number_input('Digite a altura do paciente', 0.01))

st.write("### Peso")
input_peso = float(st.number_input('Digite o peso do paciente', 0.1))

st.write("### Histórico Familiar")
input_historico= st.radio('Há histórico de obesidade na família?', ['Sim', 'Não'])

st.write("### FAVC")
input_favc = st.radio('Consome alimentos altamente calóricos com frequência?', ['Sim', 'Não'])

st.write("### FCVC")
input_fcvc = st.selectbox('Frequência de consumo de vegetais', dados['FCVC'].unique())

st.write("### NCP")
input_ncp = st.selectbox('Número de refeições principais', dados['NCP'].unique())

st.write("### CAEC")
input_caec = st.selectbox('Consumo de comida entre refeições', dados['CAEC'].unique())

st.write("### SMOKE")
input_smoke = st.radio('O paciente fuma?', ['Sim', 'Não'])

st.write("### CH2O")
input_ch2o= st.selectbox('Consumo de água diário', dados['CH2O'].unique())

st.write("### SCC")
input_scc = st.radio('O paciente monitora a ingestão diária de calorias?', ['Sim', 'Não'])

st.write("### FAF")
input_faf = st.selectbox('Frequência semanal de atividade física', dados['FAF'].unique())

st.write("### TUE")
input_tue = st.selectbox('Tempo diário usando dispositivos eletrônicos', dados['TUE'].unique())

st.write("### CALC")
input_calc = st.selectbox('Consumo de bebida alcoólica', dados['CALC'].unique())

st.write("### MTRANS")
input_mtrans = st.selectbox('Meio de transporte habitual', dados['MTRANS'].unique())

# Lista de todas as variáveis:
novo_cliente = [input_genero, # Gender
                input_idade, # Age
                input_altura, # Height
                input_peso, # Weight
                input_historico, # family_history
                input_favc, # FAVC
                input_fcvc,  # FCVC
                input_ncp,  # NCP
                input_caec, # CAEC
                input_smoke, # SMOKE
                input_ch2o, # CH20
                input_scc, # SCC
                input_faf, # FAF
                input_tue, # TUE
                input_calc, # CALC
                input_mtrans, # MTRANS
                None # target (Obesity)
                ]

# Separando os dados em treino e teste
def data_split(df, test_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(dados, 0.2)

# Criando novo cliente
cliente_predict_df = pd.DataFrame([novo_cliente],columns=teste_df.columns)

# Concatenando novo cliente ao dataframe dos dados de teste
teste_novo_cliente  = pd.concat([teste_df,cliente_predict_df],ignore_index=True)

# Pipeline
def pipeline_teste(df):

    pipeline = Pipeline([
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMax()),
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

# Aplicando a pipeline
teste_novo_cliente = pipeline_teste(teste_novo_cliente)

# Retirando a coluna target
cliente_pred = teste_novo_cliente.drop(['Obesity'], axis=1)

# Dicionário de categorias após o OrdinalFeature
obesity_labels = {
    0: 'Abaixo do Peso',
    1: 'Obesidade I',
    2: 'Obesidade II',
    3: 'Obesidade III',
    4: 'Peso Normal',
    5: 'Sobrepeso I',
    6: 'Sobrepeso II'
}

# Predições
if st.button('Enviar'):
    model = joblib.load('modelo/forest.joblib')
    final_pred = model.predict(cliente_pred)
    predicted_class = int(final_pred[-1])
    st.success(f"### O nível de obesidade previsto para o paciente é: **{obesity_labels[predicted_class]}**")