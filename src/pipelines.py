import io
import base64
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

from .config import (
    SENTIMENT_MODEL,
    EMOTION_MODEL,
    TOXIC_MODEL,
    NER_MODEL,
    TOPIC_MODEL,
    SUMM_MODEL,
    LANG_MODEL,
    POS_WORDS,
    NEG_WORDS,
)

# Create pipelines once
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    top_k=None,
    return_all_scores=True,
)

emotion_pipe = pipeline(
    "text-classification",
    model=EMOTION_MODEL,
    return_all_scores=True,
    top_k=None,
)

toxicity_pipe = pipeline(
    "text-classification",
    model=TOXIC_MODEL,
    return_all_scores=True,
    top_k=None,
)

ner_pipe = pipeline(
    "token-classification",
    model=NER_MODEL,
    aggregation_strategy="simple",
)

topic_pipe = pipeline(
    "zero-shot-classification",
    model=TOPIC_MODEL,
)

summ_pipe = pipeline(
    "summarization",
    model=SUMM_MODEL,
)

lang_pipe = pipeline(
    "text-classification",
    model=LANG_MODEL,
    top_k=None,
    return_all_scores=True,
)


def _bar_png(labels: List[str], values: List[float]) -> str:
    """Return a base64 encoded bar chart for use in HTML img tag."""
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.barh(labels, values)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


def _highlight_words(text: str) -> str:
    """Simple keyword based highlight for positive and negative words."""
    words = text.split()
    out = []
    for w in words:
        key = w.strip(".,!?;:").lower()
        if key in POS_WORDS:
            out.append(
                "<mark style='background:#d9fdd3;padding:0 2px;border-radius:2px;'>"
                + w
                + "</mark>"
            )
        elif key in NEG_WORDS:
            out.append(
                "<mark style='background:#ffd6d6;padding:0 2px;border-radius:2px;'>"
                + w
                + "</mark>"
            )
        else:
            out.append(w)
    return " ".join(out)


# -------- Sentiment --------

def sentiment_single(text: str, explain: bool = False) -> Tuple[str, str]:
    if not text.strip():
        return "<div class='result-card'>Please enter some text.</div>", ""

    scores = sentiment_pipe(text)[0]
    scores = [
        {
            "label": s["label"].capitalize(),
            "score": float(s["score"]),
        }
        for s in scores
    ]
    best = max(scores, key=lambda x: x["score"])

    chart = _bar_png(
        [s["label"] for s in scores],
        [s["score"] for s in scores],
    )
    explained = _highlight_words(text) if explain else text

    html = (
        f"<div class='result-card'>"
        f"Sentiment: <b>{best['label']}</b> "
        f"with confidence <b>{best['score']:.2f}</b>"
        f"</div>"
        f"<div style='margin-top:8px'>{explained}</div>"
    )
    return html, f"<img src='{chart}' style='width:100%;max-width:560px'>"


def sentiment_batch(multiline_text: str):
    rows = [r.strip() for r in multiline_text.split("\n") if r.strip()]
    if not rows:
        return pd.DataFrame(columns=[
            "text",
            "predicted_label",
            "confidence",
            "negative",
            "neutral",
            "positive",
        ]), None

    records = []
    for t in rows:
        scores = sentiment_pipe(t)[0]
        scores_norm = {s["label"].capitalize(): float(s["score"]) for s in scores}
        best_label = max(scores_norm, key=scores_norm.get)
        record = {
            "text": t,
            "predicted_label": best_label,
            "confidence": round(scores_norm[best_label], 4),
            "negative": round(scores_norm.get("Negative", 0.0), 4),
            "neutral": round(scores_norm.get("Neutral", 0.0), 4),
            "positive": round(scores_norm.get("Positive", 0.0), 4),
        }
        records.append(record)

    df = pd.DataFrame(records)
    path = "sentiment_results.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return df, path


# -------- Emotion --------

def emotion_single(text: str, explain: bool = False) -> Tuple[str, str]:
    if not text.strip():
        return "<div class='result-card'>Please enter some text.</div>", ""

    scores = emotion_pipe(text)[0]
    scores = sorted(scores, key=lambda d: d["score"], reverse=True)
    labels = [s["label"] for s in scores]
    values = [float(s["score"]) for s in scores]
    best = scores[0]

    chart = _bar_png(labels, values)
    explained = _highlight_words(text) if explain else text

    html = (
        f"<div class='result-card'>"
        f"Top emotion: <b>{best['label']}</b> "
        f"with confidence <b>{best['score']:.2f}</b>"
        f"</div>"
        f"<div style='margin-top:8px'>{explained}</div>"
    )
    return html, f"<img src='{chart}' style='width:100%;max-width:560px'>"


# -------- Toxicity --------

def toxicity_single(text: str) -> Tuple[str, str]:
    if not text.strip():
        return "<div class='result-card'>Please enter some text.</div>", ""

    scores = toxicity_pipe(text)[0]
    scores = sorted(scores, key=lambda d: d["score"], reverse=True)
    best = scores[0]
    labels = [s["label"] for s in scores]
    values = [float(s["score"]) for s in scores]
    chart = _bar_png(labels, values)

    html = (
        f"<div class='result-card'>"
        f"Toxicity label: <b>{best['label']}</b> "
        f"with confidence <b>{best['score']:.2f}</b>"
        f"</div>"
        f"<div style='margin-top:8px'>{text}</div>"
    )
    return html, f"<img src='{chart}' style='width:100%;max-width:560px'>"


# -------- Topics (zero shot) --------

def topic_classify(text: str, labels_str: str, multi_label: bool):
    if not text.strip():
        return "<div class='result-card'>Please enter some text.</div>", ""

    candidate_labels = [
        s.strip()
        for s in labels_str.split(",")
        if s.strip()
    ]
    if not candidate_labels:
        return "<div class='result-card'>Please provide at least one label.</div>", ""

    result = topic_pipe(
        text,
        candidate_labels=candidate_labels,
        multi_label=multi_label,
    )
    labels = result["labels"]
    scores = [float(s) for s in result["scores"]]
    chart = _bar_png(labels, scores)

    rows = [
        f"<li>{lab}: <b>{score:.2f}</b></li>"
        for lab, score in zip(labels, scores)
    ]
    html = (
        "<div class='result-card'>Topic probabilities:</div>"
        "<ul style='margin-top:6px'>" + "".join(rows) + "</ul>"
    )
    return html, f"<img src='{chart}' style='width:100%;max-width:560px'>"


# -------- NER --------

def ner_single(text: str) -> str:
    if not text.strip():
        return "<div class='result-card'>Please enter some text.</div>"

    entities = ner_pipe(text)
    if not entities:
        return "<div class='result-card'>No named entities found.</div>"

    spans = []
    for ent in entities:
        word = ent.get("word", ent.get("entity_group", ""))
        label = ent.get("entity_group", ent.get("entity", ""))
        score = float(ent.get("score", 0.0))
        spans.append(
            f"<mark style='background:#dbeafe;padding:0 2px;border-radius:2px;'>"
            f"{word} <small>({label} {score:.2f})</small>"
            f"</mark>"
        )

    html = (
        "<div class='result-card'>Detected entities:</div>"
        "<div style='margin-top:8px;line-height:1.7'>"
        + " ".join(spans)
        + "</div>"
    )
    return html


# -------- Summarization --------

def summarize_text(text: str, max_words: int) -> str:
    if not text.strip():
        return "<div class='result-card'>Please paste some text.</div>"

    target_tokens = max(30, int(max_words * 1.3))
    result = summ_pipe(
        text,
        max_length=min(512, target_tokens),
        min_length=25,
        do_sample=False,
    )[0]["summary_text"]

    return "<div class='result-card'>" + result + "</div>"


# -------- Language detection --------

def detect_language(text: str):
    if not text.strip():
        return "<div class='result-card'>Please enter text.</div>", ""

    scores = lang_pipe(text)[0]
    scores = sorted(scores, key=lambda d: d["score"], reverse=True)
    top = scores[0]
    labels = [s["label"] for s in scores[:8]]
    values = [float(s["score"]) for s in scores[:8]]
    chart = _bar_png(labels, values)

    html = (
        f"<div class='result-card'>"
        f"Language: <b>{top['label']}</b> "
        f"with confidence <b>{top['score']:.2f}</b>"
        f"</div>"
    )
    return html, f"<img src='{chart}' style='width:100%;max-width:560px'>"
