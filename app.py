from fastapi import FastAPI
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import pandas as pd

app = FastAPI()

modelo = IsolationForest(
    contamination=0.02,
    random_state=42
)

@app.get("/")
async def status():
    return {"status":"ML online"}

@app.post("/analisar")
async def analisar(dados: list):

    df = pd.DataFrame(dados)

    # segurança
    if len(df) < 10:
        return {
            "erro":"mínimo 10 leituras"
        }

    ultima_temp=df.iloc[-1]["temperatura"]
    ultima_umid=df.iloc[-1]["umidade"]

    alertas=[]

    # regras rápidas
    if ultima_temp>35:
        alertas.append(
            "Temperatura crítica"
        )

    if ultima_umid>80:
        alertas.append(
            "Umidade elevada"
        )

    # IA de anomalias
    X=df[[
        "temperatura",
        "umidade"
    ]]

    modelo.fit(X)

    pred=modelo.predict(X)

    anomalia=pred[-1]==-1

    # previsão futura
    p=df[[
        "timestamp",
        "temperatura"
    ]]

    p.columns=[
        "ds",
        "y"
    ]

    m=Prophet()

    m.fit(p)

    futuro=m.make_future_dataframe(
        periods=12,
        freq='5min'
    )

    previsao=m.predict(futuro)

    temp_futura=round(
        float(
            previsao.iloc[-1]["yhat"]
        ),2
    )

    risco="baixo"

    if temp_futura>30:
        risco="medio"

    if temp_futura>35:
        risco="alto"

    return {

        "anomalia":bool(
            anomalia
        ),

        "temperatura_atual":
        float(
            ultima_temp
        ),

        "umidade_atual":
        float(
            ultima_umid
        ),

        "previsao_1h":
        temp_futura,

        "risco":
        risco,

        "alertas":
        alertas
    }
