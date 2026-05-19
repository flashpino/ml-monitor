from fastapi import FastAPI
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from influxdb_client import InfluxDBClient
import pandas as pd
import os

app = FastAPI()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

cliente = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

query_api = cliente.query_api()

modelo = IsolationForest(
    contamination=0.02,
    random_state=42
)

@app.get("/")
async def status():
    return {"status":"ML online"}

@app.get("/analisar")
async def analisar():

    query=f'''
from(bucket: "{INFLUX_BUCKET}")
|> range(start: -24h)
|> filter(fn: (r) =>
    r["_field"] == "temperatura"
    or
    r["_field"] == "umidade"
)
|> pivot(
    rowKey:["_time"],
    columnKey:["_field"],
    valueColumn:"_value"
)
'''

    df=query_api.query_data_frame(query)

    if len(df)<10:
        return {
            "erro":"Poucos dados"
        }

    df=df[[
        "_time",
        "temperatura",
        "umidade"
    ]]

    df.columns=[
        "timestamp",
        "temperatura",
        "umidade"
    ]

    X=df[[
        "temperatura",
        "umidade"
    ]]

    modelo.fit(X)

    pred=modelo.predict(X)

    anomalia=pred[-1]==-1

    p=df[[
        "timestamp",
        "temperatura"
    ]]

    p.columns=["ds","y"]

    m=Prophet()

    m.fit(p)

    futuro=m.make_future_dataframe(
        periods=12,
        freq="5min"
    )

    previsao=m.predict(futuro)

    temp_futura=round(
        float(
            previsao.iloc[-1]["yhat"]
        ),2
    )

    return {

        "anomalia":bool(anomalia),

        "temperatura_atual":
        float(
            df.iloc[-1]["temperatura"]
        ),

        "umidade_atual":
        float(
            df.iloc[-1]["umidade"]
        ),

        "previsao_1h":
        temp_futura
    }
