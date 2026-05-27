from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from ..cyberbully_detector.inference import Predictor
from ..cyberbully_detector.rephrase import rephrase_to_safe
from ..cyberbully_detector.analytics import log_event, compute_dashboard, export_csv, export_pdf
from .schemas import AnalyzeRequest, AnalyzeResponse, RephraseRequest, RephraseResponse, FeedbackRequest

# Optional translation
try:
    from langdetect import detect
    from deep_translator import GoogleTranslator
except Exception:
    detect = None
    GoogleTranslator = None

app = FastAPI(title="Cyberbullying Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Predictor | None = None


@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = Predictor()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    assert predictor is not None
    text = req.text
    translated = None
    if GoogleTranslator is not None:
        try:
            lang = req.source_lang or (detect(text) if detect else "en")
            if lang and lang != "en":
                translated = GoogleTranslator(source=lang, target="en").translate(text)
            else:
                translated = text
        except Exception:
            translated = text
    else:
        translated = text

    out = predictor.predict(translated)
    exps = predictor.explain_lime(translated)

    # 🩹 Fix structure mismatch before validation
    # Ensure probs is a flat list or float
    if isinstance(out.get("probs"), list):
        if len(out["probs"]) and isinstance(out["probs"][0], list):
            out["probs"] = out["probs"][0]  # flatten [[0.99, 0.01]] → [0.99, 0.01]

    # Ensure explanation tokens are strings (not floats)
    for e in exps:
        if not isinstance(e.get("token"), str):
            e["token"] = str(e["token"])

    out_payload = AnalyzeResponse(
        **{**out,"text": text, "explanations": exps, "translated": translated if translated != text else None}
    )


    log_event(text=text, prediction=out_payload.prediction, category=out_payload.category, severity=out_payload.severity, confidence=out_payload.confidence)
    return out_payload


@app.post("/rephrase", response_model=RephraseResponse)
async def rephrase(req: RephraseRequest):
    return RephraseResponse(**rephrase_to_safe(req.text))


@app.get("/dashboard")
async def dashboard():
    return compute_dashboard()


@app.get("/report/csv")
async def report_csv():
    path = export_csv()
    return {"path": path}


@app.get("/report/pdf")
async def report_pdf():
    path = export_pdf()
    return {"path": path}


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    from ..cyberbully_detector.analytics import load_analytics_df, ANALYTICS_CSV
    df = load_analytics_df()
    if not df.empty:
        idxs = df.index[df['text'] == req.text].tolist()
        if idxs:
            idx = idxs[-1]
            df.at[idx, 'feedback'] = 'yes' if req.correct else 'no'
            df.to_csv(ANALYTICS_CSV, index=False)
    return {"ok": True}


@app.get("/models/comparison")
async def get_model_comparison():
    """Get comprehensive model comparison metrics"""
    from ..cyberbully_detector.model_comparison import load_saved_metrics, COMPARISON_DIR
    import os
    
    metrics = load_saved_metrics()
    
    # Get list of confusion matrix images
    confusion_matrices_dir = COMPARISON_DIR / "confusion_matrices"
    confusion_matrices = []
    if confusion_matrices_dir.exists():
        for file in confusion_matrices_dir.glob("*.png"):
            confusion_matrices.append({
                "filename": file.name,
                "path": str(file),
                "model": file.stem.replace("_confusion_matrix", "").replace("_normalized", "")
            })
    
    # Get comparison plots
    comparison_plots = []
    for plot_file in ["model_comparison.png", "model_comparison_grouped.png"]:
        plot_path = COMPARISON_DIR / plot_file
        if plot_path.exists():
            comparison_plots.append({
                "filename": plot_file,
                "path": str(plot_path)
            })
    
    return {
        "metrics": metrics,
        "confusion_matrices": confusion_matrices,
        "comparison_plots": comparison_plots,
        "csv_path": str(COMPARISON_DIR / "model_metrics.csv"),
        "detailed_csv_path": str(COMPARISON_DIR / "model_metrics_detailed.csv"),
        "report_path": str(COMPARISON_DIR / "comparison_report.txt")
    }


@app.get("/models/metrics/{model_name}")
async def get_model_metrics(model_name: str):
    """Get detailed metrics for a specific model"""
    from ..cyberbully_detector.model_comparison import load_saved_metrics
    
    metrics = load_saved_metrics()
    for m in metrics:
        if m["model_name"] == model_name:
            return m
    
    return {"error": "Model not found"}
