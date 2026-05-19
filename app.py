from fastapi import FastAPI
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from influxdb_client import InfluxDBClient
import pandas as pd
import os

app=FastAPI()

# Variáveis que vamos configurar no EasyPanel
INFLUX_URL=os.getenv("INFLUX_URL")
INFLUX_TOKEN=os.getenv("INFLUX_TOKEN")
INFLUX_ORG=os.getenv("INFLUX_ORG")
INFLUX_BUCKET=os.getenv("INFLUX_BUCKET")

cliente=InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

query_api=cliente.query_api()

modelo=IsolationForest(
    contamination=0.02,
    random_state=42
)

@app.get("/")
async def status():
    return {"status":"ML online"}

@app.get("/analisar")

async def analisar():

    query=f'''
from(bucket:"{INFLUX_BUCKET}")
|> range(start:-24h)
|> filter(fn:(r)=>r._field=="temperatura")
'''

    resultado=query_api.query_data_frame(query)

    if len(resultado)<10:
        return {
            "erro":"Poucos dados"
        }

    df=resultado[["_time","_value"]]

    df.columns=[
        "timestamp",
        "temperatura"
    ]

    df["umidade"]=60

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
      freq='5min'
    )

    prev=m.predict(futuro)

    return{
      "anomalia":bool(anomalia),
      "temperatura":
      float(df.iloc[-1]["temperatura"]),
      "previsao":
      round(
      float(prev.iloc[-1]["yhat"]),2)
    }
