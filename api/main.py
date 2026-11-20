"""PathSnap FastAPI backend (Laptop 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Ground-truth routes stay on this machine: they are only used to compute accuracy AFTER prediction,
and are never returned to the UI.
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import EVAL_DATA_DIR
from src.data_io import load_trip
from src.pipeline import evaluate_prediction, load_artifacts, predict_trip, route_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL, HMM_CFG = load_artifacts()
log.info("Loaded final model and HMM config %s; trips dir = %s", HMM_CFG, EVAL_DATA_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def available_trips():
    return sorted(p.name for p in EVAL_DATA_DIR.iterdir() if p.is_dir())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    return {"trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in available_trips():
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    log.info("REQUEST /predict trip_id=%s", req.trip_id)
    trip = load_trip("eval", req.trip_id, with_route=True, base_dir=EVAL_DATA_DIR)
    out = predict_trip(trip, MODEL, HMM_CFG)          # prediction: no route used
    metrics = evaluate_prediction(trip, out)           # server-side scoring only
    seq, coords = route_coordinates(out["hmm_pred"], trip)
    tr = trip["track"]
    resp = {
        "trip_id": req.trip_id,
        "n_gps_points": int(len(tr)),
        "gps_points": tr[["lat", "lon"]].round(6).values.tolist(),
        "predicted_route_latlon": coords,
        "predicted_arc_sequence_length": len(seq),
        "hmm_chain_breaks": out["n_breaks"],
        "metrics": metrics,
        "processing_seconds": round(time.time() - t0, 2),
    }
    if metrics:
        log.info("RESULT trip=%s top1(clf)=%.4f top1(hmm)=%.4f continuity(hmm)=%.4f time=%.2fs",
                 req.trip_id, metrics["classifier_only"]["top1_accuracy"], metrics["classifier_hmm"]["top1_accuracy"],
                 metrics["classifier_hmm"]["route_continuity"], resp["processing_seconds"])
    return resp
