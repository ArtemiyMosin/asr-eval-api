"""ASR Eval API — a small service for evaluating speech recognition quality (WER/CER)."""

import logging
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_NAME = os.getenv("APP_NAME", "asr-eval-api")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "1000"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(APP_NAME)

STARTED_AT = time.time()


# --- Metrics -----------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, unify ё/е, drop punctuation and collapse whitespace."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())


def align(ref: list, hyp: list) -> dict:
    """Levenshtein alignment of two token sequences.

    Returns counts of substitutions, deletions and insertions on the cheapest path.
    Only two DP rows are kept, so memory is O(len(hyp)).
    """
    # each cell: (cost, substitutions, deletions, insertions)
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            s, d, ins = prev[j - 1], prev[j], cur[j - 1]
            cur.append(min(
                (s[0] + 1, s[1] + 1, s[2], s[3]),
                (d[0] + 1, d[1], d[2] + 1, d[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            ))
        prev = cur
    cost, subs, dels, inss = prev[-1]
    return {"errors": cost, "substitutions": subs, "deletions": dels, "insertions": inss}


def compute_metrics(reference: str, hypothesis: str) -> dict:
    ref_norm, hyp_norm = normalize(reference), normalize(hypothesis)
    if not ref_norm:
        raise HTTPException(status_code=422, detail="Reference is empty after normalization")
    ref_words, hyp_words = ref_norm.split(), hyp_norm.split()
    words = align(ref_words, hyp_words)
    chars = align(list(ref_norm), list(hyp_norm))
    return {
        "wer": round(words["errors"] / len(ref_words), 4),
        "cer": round(chars["errors"] / len(ref_norm), 4),
        "reference_words": len(ref_words),
        **words,
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm,
    }


class TextPair(BaseModel):
    reference: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, examples=["Привет, как дела?"])
    hypothesis: str = Field(..., max_length=MAX_TEXT_LENGTH, examples=["привет как дело"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting %s v%s (log level %s)", APP_NAME, APP_VERSION, LOG_LEVEL)
    yield
    logger.info("Shutting down %s", APP_NAME)


app = FastAPI(
    title="ASR Eval API",
    version=APP_VERSION,
    description="Evaluate speech recognition output against reference transcripts: WER, CER, per-model summaries.",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"service": APP_NAME, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - STARTED_AT, 1)}


@app.get("/version")
def version():
    return {"name": APP_NAME, "version": APP_VERSION}


@app.post("/metrics/wer")
def metrics_wer(pair: TextPair):
    """Compute WER/CER for a single reference/hypothesis pair without storing it."""
    return compute_metrics(pair.reference, pair.hypothesis)
