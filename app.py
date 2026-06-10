from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "modelo_final.joblib"
METADATA_PATH = ROOT / "model" / "metadata.json"


def tenure_group(value: int) -> str:
    if value <= 12:
        return "0-12"
    if value <= 24:
        return "13-24"
    if value <= 36:
        return "25-36"
    return "37+"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def build_input(data: dict, columns: list[str]) -> pd.DataFrame:
    row = pd.DataFrame([data])
    return row.reindex(columns=columns)


st.set_page_config(page_title="Previsao de Churn", layout="wide")

st.title("Previsao de Churn de Clientes")

model, metadata = load_model()
threshold = float(metadata["threshold"])

with st.form("prediction_form"):
    st.subheader("Dados do cliente")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Genero", ["Female", "Male"])
        senior = st.selectbox("Cliente senior", [0, 1], format_func=lambda x: "Sim" if x else "Nao")
        partner = st.selectbox("Possui parceiro(a)", ["Yes", "No"])
        dependents = st.selectbox("Possui dependentes", ["Yes", "No"])
        tenure = st.slider("Tempo como cliente (meses)", 0, 72, 12)
        contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Fatura digital", ["Yes", "No"])

    with col2:
        phone = st.selectbox("Servico telefonico", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiplas linhas", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Servico de internet", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Seguranca online", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Backup online", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Protecao do dispositivo", ["No", "Yes", "No internet service"])

    with col3:
        tech_support = st.selectbox("Suporte tecnico", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming filmes", ["No", "Yes", "No internet service"])
        payment = st.selectbox(
            "Metodo de pagamento",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input("Cobranca mensal", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        total = st.number_input(
            "Cobranca total",
            min_value=0.0,
            max_value=10000.0,
            value=float(max(tenure, 1) * monthly),
            step=10.0,
        )

    submitted = st.form_submit_button("Executar predicao")

input_data = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": int(tenure),
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(total),
    "tenure_group": tenure_group(int(tenure)),
}

if submitted:
    input_df = build_input(input_data, metadata["input_columns"])
    probability = float(model.predict_proba(input_df)[:, 1][0])
    prediction = int(probability >= threshold)

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Probabilidade de churn", f"{probability:.1%}")
    res_col2.metric("Threshold usado", f"{threshold:.3f}")
    res_col3.metric("Resultado", "Churn" if prediction else "Nao churn")

    if prediction:
        st.warning(
            "O cliente foi classificado como propenso a cancelar. Vale priorizar uma acao de retencao."
        )
    else:
        st.success(
            "O cliente foi classificado como baixa propensao de cancelamento no limite atual."
        )

    with st.expander("Dados enviados ao modelo"):
        st.dataframe(input_df, use_container_width=True)
