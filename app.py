from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from influxdb_client import InfluxDBClient
import pandas as pd
import os

app = FastAPI()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")

modelo = IsolationForest(
    contamination=0.02,
    random_state=42
)

class Consulta(BaseModel):
    org: str
    bucket: str
    device: str


@app.get("/")
async def status():

    return {
        "status":"ML online"
    }


@app.post("/analisar")
async def analisar(req: Consulta):

    try:

        cliente = InfluxDBClient(
            url=INFLUX_URL,
            token=INFLUX_TOKEN,
            org=req.org
        )

        query_api = cliente.query_api()

        query = f'''
from(bucket:"{req.bucket}")
|> range(start:-24h)

|> filter(fn:(r)=>
    r["device_id"]=="{req.device}"
)

|> filter(fn:(r)=>
    r["_field"]=="temperatura"
    or
    r["_field"]=="umidade"
)

|> pivot(
    rowKey:["_time"],
    columnKey:["_field"],
    valueColumn:"_value"
)
'''

        df = query_api.query_data_frame(query)

        if isinstance(df, list):
            df = pd.concat(df)

        if df.empty:
            return {
                "erro":"Nenhum dado encontrado"
            }

        colunas = df.columns.tolist()

        if "temperatura" not in colunas:
            return {
                "erro":"Campo temperatura não encontrado",
                "colunas":colunas
            }

        if "umidade" not in colunas:
            return {
                "erro":"Campo umidade não encontrado",
                "colunas":colunas
            }

        df = df[[
            "_time",
            "temperatura",
            "umidade"
        ]]

        df.columns = [
            "timestamp",
            "temperatura",
            "umidade"
        ]

        df = df.dropna()

        if len(df) < 10:
            return {
                "erro":"Poucos dados",
                "quantidade":len(df)
            }

        # remove timezone
        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        ).dt.tz_localize(None)

        # -------- ANOMALIA --------

        X = df[[
            "temperatura",
            "umidade"
        ]]

        modelo.fit(X)

        pred = modelo.predict(X)

        anomalia = pred[-1] == -1

        # -------- PROPHET --------

        p = df[[
            "timestamp",
            "temperatura"
        ]]

        p.columns = [
            "ds",
            "y"
        ]

        prophet = Prophet()

        prophet.fit(p)

        futuro = prophet.make_future_dataframe(
            periods=12,
            freq="5min"
        )

        previsao = prophet.predict(
            futuro
        )

        temperatura_futura = round(
            float(
                previsao.iloc[-1]["yhat"]
            ),
            2
        )

        risco="baixo"

        if temperatura_futura > 30:
            risco="medio"

        if temperatura_futura > 35:
            risco="alto"

        return {

            "org":req.org,
            "bucket":req.bucket,
            "device":req.device,

            "anomalia":
            bool(anomalia),

            "temperatura_atual":
            round(
                float(
                    df.iloc[-1]["temperatura"]
                ),
                2
            ),

            "umidade_atual":
            round(
                float(
                    df.iloc[-1]["umidade"]
                ),
                2
            ),

            "previsao_1h":
            temperatura_futura,

            "risco":
            risco
        }

    except Exception as e:

        return {
            "erro":str(e)
        }
