from __future__ import annotations
import csv
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from wordcloud import WordCloud
from .config import LOGS_DIR

ANALYTICS_CSV = LOGS_DIR / "analytics.csv"
REPORTS_DIR = LOGS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def log_event(text: str, prediction: str, category: str, severity: str, confidence: float, correct: bool | None = None):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "text", "prediction", "category", "severity", "confidence", "feedback"]
    row = [datetime.utcnow().isoformat(), text, prediction, category, severity, confidence, None if correct is None else ("yes" if correct else "no")]
    new_file = not ANALYTICS_CSV.exists()
    with open(ANALYTICS_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def load_analytics_df() -> pd.DataFrame:
    if ANALYTICS_CSV.exists():
        return pd.read_csv(ANALYTICS_CSV)
    return pd.DataFrame(columns=["timestamp", "text", "prediction", "category", "severity", "confidence", "feedback"])


def compute_dashboard() -> Dict[str, Any]:
    df = load_analytics_df()
    cat_counts = df["category"].value_counts(dropna=False).to_dict()
    sev_counts = df["severity"].value_counts(dropna=False).to_dict()
    bully_texts = df[df["prediction"] == "bullying"]["text"].astype(str)
    text_blob = " ".join(bully_texts.tolist())
    wc = WordCloud(width=600, height=400).generate(text_blob) if text_blob.strip() else None
    wc_path = None
    if wc is not None:
        wc_path = REPORTS_DIR / "wordcloud.png"
        wc.to_file(wc_path)
    if not df.empty:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        timeline = df.groupby("date").size().reset_index(name="count").to_dict(orient="list")
    else:
        timeline = {"date": [], "count": []}
    return {
        "category_distribution": cat_counts,
        "severity_distribution": sev_counts,
        "wordcloud_path": str(wc_path) if wc_path else None,
        "timeline": timeline
    }


def export_csv() -> str:
    df = load_analytics_df()
    out = REPORTS_DIR / f"analytics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out, index=False)
    return str(out)


def export_pdf() -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    out = REPORTS_DIR / f"analytics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, h - 72, "Cyberbullying Analytics Report")
    c.setFont("Helvetica", 12)
    y = h - 100
    dash = compute_dashboard()
    c.drawString(72, y, f"Category distribution: {dash['category_distribution']}")
    y -= 20
    c.drawString(72, y, f"Severity distribution: {dash['severity_distribution']}")
    y -= 20
    if dash.get("wordcloud_path"):
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(dash["wordcloud_path"])
            c.drawImage(img, 72, y - 200, width=400, height=200)
            y -= 220
        except Exception:
            pass
    c.showPage()
    c.save()
    return str(out)
