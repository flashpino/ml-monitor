from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from influxdb_client import InfluxDBClient
import pandas as pd
import joblib
import os
import json
import logging
from datetime import datetime
from pathlib import Path
 
# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CPD ML Service")
 
# ── config ────────────────────────────────────────────────────────────────────
INFLUX_URL      = os.getenv("INFLUX_URL")
INFLUX_TOKEN    = os.getenv("INFLUX_TOKEN")
TRAINING_RANGE  = os.getenv("TRAINING_RANGE", "-90d")
 
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
 
# buckets internos do influx — ignorar no discovery
BUCKETS_IGNORADOS = {"_monitoring", "_tasks", "monitoring"}
 
# ── schemas ───────────────────────────────────────────────────────────────────
class Consulta(BaseModel):
    org: str
    bucket: str
    device: str
 
# ── helpers de path ───────────────────────────────────────────────────────────
def model_key(org: str, bucket: str, device: str) -> str:
    return f"{org}__{bucket}__{device}".replace("/", "_").replace(" ", "_")
 
def isolation_path(org: str, bucket: str, device: str) -> Path:
    return MODEL_DIR / f"iso__{model_key(org, bucket, device)}.pkl"
 
def prophet_path(org: str, bucket: str, device: str) -> Path:
    return MODEL_DIR / f"prophet__{model_key(org, bucket, device)}.pkl"
 
def meta_path(org: str, bucket: str, device: str) -> Path:
    return MODEL_DIR / f"meta__{model_key(org, bucket, device)}.json"
 
def salvar_meta(org: str, bucket: str, device: str, dados: dict):
    meta_path(org, bucket, device).write_text(json.dumps(dados, indent=2))
 
def carregar_meta(org: str, bucket: str, device: str) -> dict:
    p = meta_path(org, bucket, device)
    if not p.exists():
        return {}
    return json.loads(p.read_text())
 
# ── helpers influx ────────────────────────────────────────────────────────────
def get_client(org: str) -> InfluxDBClient:
    if not INFLUX_URL or not INFLUX_TOKEN:
        raise HTTPException(500, "INFLUX_URL e INFLUX_TOKEN não configurados.")
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=org)
 
 
def listar_orgs() -> list[str]:
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN)
    orgs_api = client.organizations_api()
    return [o.name for o in orgs_api.find_organizations()]
 
 
def listar_buckets(org: str) -> list[str]:
    client = get_client(org)
    buckets_api = client.buckets_api()
    buckets = buckets_api.find_buckets().buckets or []
    return [b.name for b in buckets if b.name not in BUCKETS_IGNORADOS]
 
 
def listar_devices(org: str, bucket: str) -> list[str]:
    client    = get_client(org)
    query_api = client.query_api()
    query = f'''
import "influxdata/influxdb/schema"
schema.tagValues(
    bucket: "{bucket}",
    tag: "device_id",
    start: {TRAINING_RANGE}
)
'''
    try:
        tables  = query_api.query(query)
        devices = []
        for table in tables:
            for record in table.records:
                v = record.get_value()
                if v:
                    devices.append(v)
        return devices
    except Exception as e:
        log.warning("Erro ao listar devices em %s/%s: %s", org, bucket, e)
        return []
 
 
def query_dados(org: str, bucket: str, device: str, range_str: str) -> pd.DataFrame:
    client    = get_client(org)
    query_api = client.query_api()
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
 
    faltando = {"temperatura", "umidade", "_time"} - set(df.columns)
    if faltando:
        log.warning("Campos ausentes em %s/%s/%s: %s", org, bucket, device, faltando)
        return pd.DataFrame()
 
    df = df[["_time", "temperatura", "umidade"]].copy()
    df.columns = ["timestamp", "temperatura", "umidade"]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.dropna().reset_index(drop=True)
    return df
 
# ── treino de um device ───────────────────────────────────────────────────────
# limites razoáveis para temperatura e umidade
TEMP_MIN = float(os.getenv("TEMP_MIN", "15"))
TEMP_MAX = float(os.getenv("TEMP_MAX", "35"))
UMID_MIN = float(os.getenv("UMID_MIN", "20"))
UMID_MAX = float(os.getenv("UMID_MAX", "80"))
def treinar_device(org, bucket, device):
    df = query_dados(org, bucket, device, range_str=TRAINING_RANGE)

    # remove leituras fisicamente impossíveis
    antes = len(df)
    df = df[
        df["temperatura"].between(TEMP_MIN, TEMP_MAX) &
        df["umidade"].between(UMID_MIN, UMID_MAX)
    ]
    descartados = antes - len(df)
    if descartados > 0:
        log.warning("Descartados %d registros inválidos em %s/%s/%s", descartados, org, bucket, device)

    if len(df) < 100:
        return {
            "status": "dados_insuficientes",
            "org": org, "bucket": bucket, "device": device,
            "amostras": len(df),
        }
 
    # IsolationForest
    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(df[["temperatura", "umidade"]])
    joblib.dump(iso, isolation_path(org, bucket, device))
 
    # Prophet
    p = df[["timestamp", "temperatura"]].copy()
    p.columns = ["ds", "y"]
    prophet = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.80,
    )
    prophet.fit(p)
    joblib.dump(prophet, prophet_path(org, bucket, device))
 
    # Meta
    meta = {
        "treinado_em": datetime.utcnow().isoformat(),
        "amostras": len(df),
        "temperatura_min": round(float(df["temperatura"].min()), 2),
        "temperatura_max": round(float(df["temperatura"].max()), 2),
        "umidade_min": round(float(df["umidade"].min()), 2),
        "umidade_max": round(float(df["umidade"].max()), 2),
        "training_range": TRAINING_RANGE,
    }
    salvar_meta(org, bucket, device, meta)
 
    log.info("Concluído: %s/%s/%s — %d amostras", org, bucket, device, len(df))
    return {"status": "ok", "org": org, "bucket": bucket, "device": device, **meta}
 
# ── risco ─────────────────────────────────────────────────────────────────────
def calcular_risco(temperatura: float) -> str:
    if temperatura > 35:
        return "alto"
    if temperatura > 30:
        return "medio"
    return "baixo"
 
# ── rotas ─────────────────────────────────────────────────────────────────────
 
@app.get("/")
async def status():
    modelos = list(MODEL_DIR.glob("iso__*.pkl"))
    return {
        "status": "ML online",
        "devices_treinados": len(modelos),
        "training_range": TRAINING_RANGE,
    }
 
 
@app.post("/treinar-tudo")
async def treinar_tudo():
    """
    Descobre automaticamente todas as orgs, buckets e devices
    e treina um modelo separado pra cada um.
    Chame 1x por semana via cron no n8n.
    """
    log.info("Iniciando treino automático completo")
 
    orgs       = listar_orgs()
    resultados = []
    erros      = []
 
    for org in orgs:
        buckets = listar_buckets(org)
        log.info("Org: %s | Buckets: %s", org, buckets)
 
        for bucket in buckets:
            devices = listar_devices(org, bucket)
            log.info("Bucket: %s/%s | Devices: %s", org, bucket, devices)
 
            for device in devices:
                try:
                    resultado = treinar_device(org, bucket, device)
                    if resultado["status"] == "ok":
                        resultados.append(resultado)
                    else:
                        erros.append(resultado)
                except Exception as e:
                    log.error("Erro: %s/%s/%s — %s", org, bucket, device, e)
                    erros.append({
                        "status": "erro",
                        "org": org, "bucket": bucket, "device": device,
                        "detalhe": str(e),
                    })
 
    return {
        "treinados_com_sucesso": len(resultados),
        "com_problema":          len(erros),
        "detalhes":              resultados,
        "problemas":             erros,
    }
 
 
@app.post("/treinar")
async def treinar(req: Consulta):
    """
    Treina um device específico manualmente.
    Útil pra forçar retreino de um sensor sem esperar o cron semanal.
    """
    resultado = treinar_device(req.org, req.bucket, req.device)
 
    if resultado["status"] == "sem_dados":
        raise HTTPException(422, "Nenhum dado encontrado no período.")
    if resultado["status"] == "dados_insuficientes":
        raise HTTPException(422, f"Dados insuficientes: {resultado['amostras']} amostras (mínimo: 100).")
 
    return resultado
 
 
@app.post("/analisar")
async def analisar(req: Consulta):
    """
    Analisa os últimos 15 minutos usando o modelo treinado para este device.
    Chame a cada 5 minutos no n8n.
    """
    iso_p = isolation_path(req.org, req.bucket, req.device)
    pro_p = prophet_path(req.org, req.bucket, req.device)
 
    if not iso_p.exists() or not pro_p.exists():
        raise HTTPException(
            404,
            f"Modelo não encontrado para {req.org}/{req.bucket}/{req.device}. "
            "Chame /treinar-tudo ou /treinar primeiro."
        )
 
    iso     = joblib.load(iso_p)
    prophet = joblib.load(pro_p)
 
    df = query_dados(req.org, req.bucket, req.device, range_str="-15m")
 
    if df.empty:
        raise HTTPException(422, "Nenhum dado nos últimos 15 minutos.")
    if len(df) < 2:
        raise HTTPException(422, f"Poucos pontos: {len(df)}. Verifique o intervalo de coleta.")
 
    # Anomalia
    X           = df[["temperatura", "umidade"]]
    pred        = iso.predict(X)
    score       = iso.decision_function(X)
    anomalia    = bool(pred[-1] == -1)
    score_atual = round(float(score[-1]), 4)
 
    # Previsão 1h
    futuro   = prophet.make_future_dataframe(periods=12, freq="5min")
    previsao = prophet.predict(futuro)
 
    ultima          = previsao.iloc[-1]
    temp_futura     = round(float(ultima["yhat"]), 2)
    temp_futuro_min = round(float(ultima["yhat_lower"]), 2)
    temp_futuro_max = round(float(ultima["yhat_upper"]), 2)
 
    ultimo = df.iloc[-1]
    meta   = carregar_meta(req.org, req.bucket, req.device)
 
    return {
        "org":               req.org,
        "bucket":            req.bucket,
        "device":            req.device,
        "timestamp_leitura": ultimo["timestamp"].isoformat(),
        "temperatura_atual": round(float(ultimo["temperatura"]), 2),
        "umidade_atual":     round(float(ultimo["umidade"]), 2),
        "anomalia":          anomalia,
        "anomalia_score":    score_atual,
        "previsao_1h": {
            "temperatura":   temp_futura,
            "intervalo_min": temp_futuro_min,
            "intervalo_max": temp_futuro_max,
        },
        "risco":              calcular_risco(temp_futura),
        "modelo_treinado_em": meta.get("treinado_em", "desconhecido"),
    }
 
 
@app.get("/modelos")
async def listar_modelos():
    """Lista todos os devices que já têm modelo treinado."""
    modelos = []
    for iso_file in sorted(MODEL_DIR.glob("iso__*.pkl")):
        chave  = iso_file.stem.replace("iso__", "")
        partes = chave.split("__")
        if len(partes) != 3:
            continue
        org, bucket, device = partes
        meta = carregar_meta(org, bucket, device)
        modelos.append({
            "org": org, "bucket": bucket, "device": device,
            **meta,
        })
    return {"total": len(modelos), "modelos": modelos}
 
 
@app.delete("/modelos/{org}/{bucket}/{device}")
async def deletar_modelo(org: str, bucket: str, device: str):
    """Remove o modelo de um device específico."""
    arquivos = [
        isolation_path(org, bucket, device),
        prophet_path(org, bucket, device),
        meta_path(org, bucket, device),
    ]
    deletados = [f.name for f in arquivos if f.exists()]
    for f in arquivos:
        if f.exists():
            f.unlink()
 
    if not deletados:
        raise HTTPException(404, "Nenhum modelo encontrado para este device.")
 
    return {"deletados": deletados}
