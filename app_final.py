"""ASR Eval API — a small service for evaluating speech recognition quality (WER/CER)."""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

APP_NAME = os.getenv("APP_NAME", "asr-eval-api")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(APP_NAME)

STARTED_AT = time.time()


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
