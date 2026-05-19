from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from influxdb_client import InfluxDBClient
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from pathlib import Path
 
# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CPD ML Service")
 
# ── config ────────────────────────────────────────────────────────────────────
INFLUX_URL   = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
 
MODEL_DIR         = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_PATH        = MODEL_DIR / "isolation_forest.pkl"
MODEL_META_PATH   = MODEL_DIR / "isolation_forest_meta.json"
PROPHET_PATH      = MODEL_DIR / "prophet.pkl"
 
MODEL_DIR.mkdir(parents=True, exist_ok=True)
 
# ── schemas ───────────────────────────────────────────────────────────────────
class Consulta(BaseModel):
    org: str
    bucket: str
    device: str
 
# ── helpers ───────────────────────────────────────────────────────────────────
def get_influx_client(org: str) -> InfluxDBClient:
    if not INFLUX_URL or not INFLUX_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Variáveis INFLUX_URL e INFLUX_TOKEN não configuradas"
        )
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=org)
 
 
def query_influx(org: str, bucket: str, device: str, range_str: str) -> pd.DataFrame:
    """Consulta temperatura e umidade do InfluxDB para um range qualquer."""
    cliente   = get_influx_client(org)
    query_api = cliente.query_api()
 
    query = f'''
from(bucket:"{bucket}")
|> range(start:{range_str})
|> filter(fn:(r) => r["device_id"] == "{device}")
|> filter(fn:(r) =>
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
    df = query_api.query_data_frame(query)
 
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)
 
    if df.empty:
        return pd.DataFrame()
 
    colunas_necessarias = {"temperatura", "umidade", "_time"}
    faltando = colunas_necessarias - set(df.columns)
    if faltando:
        raise HTTPException(
            status_code=422,
            detail=f"Campos ausentes no InfluxDB: {faltando}"
        )
 
    df = df[["_time", "temperatura", "umidade"]].copy()
    df.columns = ["timestamp", "temperatura", "umidade"]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.dropna().reset_index(drop=True)
 
    return df
 
 
def modelo_existe() -> bool:
    return MODEL_PATH.exists() and PROPHET_PATH.exists()
 
 
def carregar_isolation() -> IsolationForest:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Modelo IsolationForest não encontrado. Chame /treinar primeiro."
        )
    return joblib.load(MODEL_PATH)
 
 
def carregar_prophet() -> Prophet:
    if not PROPHET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Modelo Prophet não encontrado. Chame /treinar primeiro."
        )
    return joblib.load(PROPHET_PATH)
 
 
def calcular_risco(temperatura: float) -> str:
    if temperatura > 35:
        return "alto"
    if temperatura > 30:
        return "medio"
    return "baixo"
 
 
# ── rotas ─────────────────────────────────────────────────────────────────────
 
@app.get("/")
async def status():
    return {
        "status": "ML online",
        "modelo_treinado": modelo_existe(),
        "model_dir": str(MODEL_DIR),
    }
 
 
@app.post("/treinar")
async def treinar(req: Consulta):
    """
    Treina IsolationForest + Prophet com 30 dias de histórico.
    Chame este endpoint 1x por semana (ex: cron no n8n às 02h de domingo).
    """
    log.info("Iniciando treinamento — range: -90d")
 
    df = query_influx(req.org, req.bucket, req.device, range_str=os.getenv("TRAINING_RANGE", "-90d"))

 
    if df.empty:
        raise HTTPException(status_code=422, detail="Nenhum dado encontrado nos últimos 90 dias.")
 
    if len(df) < 100:
        raise HTTPException(
            status_code=422,
            detail=f"Dados insuficientes para treino: {len(df)} registros (mínimo: 50)."
        )
 
    # ── IsolationForest ───────────────────────────────────────────────────────
    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(df[["temperatura", "umidade"]])
    joblib.dump(iso, MODEL_PATH)
    log.info("IsolationForest salvo em %s", MODEL_PATH)
 
    # ── Prophet ───────────────────────────────────────────────────────────────
    p = df[["timestamp", "temperatura"]].copy()
    p.columns = ["ds", "y"]
 
    prophet = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,   # 90d não é suficiente pra anual
        interval_width=0.80,
    )
    prophet.fit(p)
    joblib.dump(prophet, PROPHET_PATH)
    log.info("Prophet salvo em %s", PROPHET_PATH)
 
    # ── metadados ─────────────────────────────────────────────────────────────
    meta = {
        "treinado_em": datetime.utcnow().isoformat(),
        "amostras": len(df),
        "temperatura_min": round(float(df["temperatura"].min()), 2),
        "temperatura_max": round(float(df["temperatura"].max()), 2),
        "umidade_min": round(float(df["umidade"].min()), 2),
        "umidade_max": round(float(df["umidade"].max()), 2),
    }
 
    import json
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
 
    return {
        "status": "treinamento concluído",
        **meta,
    }
 
 
@app.post("/analisar")
async def analisar(req: Consulta):
    """
    Analisa os últimos 15 minutos usando modelos já treinados.
    Chame a cada 5 minutos no n8n — rápido e leve.
    """
    # ── carrega modelos do disco ───────────────────────────────────────────────
    iso    = carregar_isolation()
    prophet = carregar_prophet()
 
    # ── busca apenas dados recentes ───────────────────────────────────────────
    df = query_influx(req.org, req.bucket, req.device, range_str="-15m")
 
    if df.empty:
        raise HTTPException(status_code=422, detail="Nenhum dado nos últimos 15 minutos.")
 
    if len(df) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Poucos pontos para análise: {len(df)}. Verifique o intervalo de coleta."
        )
 
    # ── anomalia ──────────────────────────────────────────────────────────────
    X = df[["temperatura", "umidade"]]
    pred        = iso.predict(X)
    score       = iso.decision_function(X)          # quanto mais negativo, mais anômalo
    anomalia    = bool(pred[-1] == -1)
    score_atual = round(float(score[-1]), 4)
 
    # ── previsão 1h ───────────────────────────────────────────────────────────
    p = df[["timestamp", "temperatura"]].copy()
    p.columns = ["ds", "y"]
 
    # Prophet precisa de pelo menos alguns pontos — se tiver poucos,
    # usamos o modelo global já treinado e apenas predizemos o futuro
    futuro   = prophet.make_future_dataframe(periods=12, freq="5min")
    previsao = prophet.predict(futuro)
 
    ultima_previsao    = previsao.iloc[-1]
    temperatura_futura = round(float(ultima_previsao["yhat"]), 2)
    temp_futuro_min    = round(float(ultima_previsao["yhat_lower"]), 2)
    temp_futuro_max    = round(float(ultima_previsao["yhat_upper"]), 2)
 
    ultimo = df.iloc[-1]
 
    return {
        "org":    req.org,
        "bucket": req.bucket,
        "device": req.device,
        # leitura atual
        "temperatura_atual": round(float(ultimo["temperatura"]), 2),
        "umidade_atual":     round(float(ultimo["umidade"]), 2),
        "timestamp_leitura": ultimo["timestamp"].isoformat(),
        # anomalia
        "anomalia":     anomalia,
        "anomalia_score": score_atual,   # referência: < 0 suspeito, << 0 anômalo
        # previsão +1h
        "previsao_1h": {
            "temperatura":  temperatura_futura,
            "intervalo_min": temp_futuro_min,
            "intervalo_max": temp_futuro_max,
        },
        "risco": calcular_risco(temperatura_futura),
    }
 
 
@app.get("/modelo/info")
async def modelo_info():
    """Retorna metadados do último treinamento."""
    if not MODEL_META_PATH.exists():
        raise HTTPException(status_code=404, detail="Nenhum treinamento registrado ainda.")
 
    import json
    meta = json.loads(MODEL_META_PATH.read_text())
    return meta
