"""App de apoio à equipe médica: estimativa do nível de obesidade de um paciente."""
import joblib
import pandas as pd
import streamlit as st

from transformadores import ORDEM_ORDINAIS, ORDEM_CLASSES  # noqa: F401 (necessário para carregar o modelo)

st.set_page_config(page_title="Nível de Obesidade", page_icon="🩺", layout="centered")


@st.cache_resource
def carrega_modelo():
    return joblib.load("modelo.joblib")


modelo = carrega_modelo()

st.title("🩺 Estimativa de Nível de Obesidade")
st.write(
    "Preencha os dados do paciente para estimar o nível de obesidade. "
    "A ferramenta apoia a avaliação clínica e **não substitui o diagnóstico médico**."
)

with st.form("paciente"):
    st.subheader("Dados do paciente")
    c1, c2, c3 = st.columns(3)
    genero = c1.selectbox("Gênero", ["Feminino", "Masculino"])
    idade = c2.number_input("Idade (anos)", min_value=14, max_value=100, value=25)
    historico = c3.selectbox("Histórico familiar de excesso de peso?", ["Sim", "Não"])

    c4, c5 = st.columns(2)
    altura = c4.number_input("Altura (m)", min_value=1.20, max_value=2.20, value=1.70, step=0.01, format="%.2f")
    peso = c5.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5, format="%.1f")

    st.subheader("Alimentação")
    c6, c7 = st.columns(2)
    favc = c6.selectbox("Come alimentos muito calóricos com frequência?", ["Sim", "Não"])
    fcvc = c7.selectbox("Come vegetais nas refeições?", ORDEM_ORDINAIS["FCVC"])
    c8, c9 = st.columns(2)
    ncp = c8.selectbox("Refeições principais por dia", ORDEM_ORDINAIS["NCP"], index=2)
    caec = c9.selectbox("Come entre as refeições?", ORDEM_ORDINAIS["CAEC"], index=1)
    c10, c11 = st.columns(2)
    ch2o = c10.selectbox("Consumo de água", ORDEM_ORDINAIS["CH2O"], index=1)
    scc = c11.selectbox("Monitora as calorias ingeridas?", ["Não", "Sim"])

    st.subheader("Hábitos")
    c12, c13 = st.columns(2)
    faf = c12.selectbox("Atividade física", ORDEM_ORDINAIS["FAF"])
    tue = c13.selectbox("Tempo em dispositivos eletrônicos", ORDEM_ORDINAIS["TUE"])
    c14, c15, c16 = st.columns(3)
    calc = c14.selectbox("Consumo de álcool", ORDEM_ORDINAIS["CALC"])
    smoke = c15.selectbox("Fuma?", ["Não", "Sim"])
    mtrans = c16.selectbox("Meio de transporte", ["Transporte Público", "Carro", "A pé", "Moto", "Bicicleta"])

    enviar = st.form_submit_button("Estimar nível de obesidade", type="primary")

if enviar:
    paciente = pd.DataFrame([{
        "Gender": genero, "Age": float(idade), "Height": float(altura), "Weight": float(peso),
        "family_history": historico, "FAVC": favc, "FCVC": fcvc, "NCP": ncp, "CAEC": caec,
        "SMOKE": smoke, "CH2O": ch2o, "SCC": scc, "FAF": faf, "TUE": tue,
        "CALC": calc, "MTRANS": mtrans,
    }])

    previsao = modelo.predict(paciente)[0]
    probabilidades = pd.Series(modelo.predict_proba(paciente)[0], index=modelo.classes_).reindex(ORDEM_CLASSES)
    imc = peso / altura ** 2

    st.divider()
    st.subheader("Resultado")
    r1, r2 = st.columns(2)
    r1.metric("Nível estimado", previsao)
    r2.metric("IMC calculado", f"{imc:.1f}")

    st.write("**Probabilidade por nível**")
    st.bar_chart(probabilidades.rename("Probabilidade"), horizontal=True)

    st.caption(
        "Modelo: Gradient Boosting treinado com a base pública de níveis de obesidade (UCI). "
        "As estimativas refletem os padrões dessa base e devem ser interpretadas pela equipe médica."
    )
