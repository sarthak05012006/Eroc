"""All tunables in one place. Override any of them with an environment variable."""
import os
from pathlib import Path

VERSION = "2.0"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
FRONTEND_DIR = ROOT / "frontend"

# ---- Ollama (can run on another machine: OLLAMA_HOST=http://192.168.1.20:11434) ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")       # qwen3.5:4b on an 8 GB machine
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")        # multilingual, handles Hindi
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "900"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))        # retries on connection errors / 5xx
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")             # keep the model in memory between questions
WARMUP = os.getenv("WARMUP", "1") == "1"                # load models at startup so question 1 is not slow

# ---- retrieval / verification ----
TOP_K = int(os.getenv("TOP_K", "8"))
MAX_EVIDENCE = int(os.getenv("MAX_EVIDENCE", "12"))
MIN_SIM = float(os.getenv("MIN_SIM", "0.35"))           # best cosine below this -> abstain before calling the LLM
QUOTE_MIN_RATIO = float(os.getenv("QUOTE_MIN_RATIO", "0.9"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1000"))
SCAN_SIM = float(os.getenv("SCAN_SIM", "0.60"))
SCAN_MAX_GROUPS = int(os.getenv("SCAN_MAX_GROUPS", "60"))

# ---- service limits ----
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "64"))         # answers kept in memory; 0 disables the cache
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def summary() -> dict:
    """Safe-to-log view of the active settings."""
    keys = ["VERSION", "OLLAMA_HOST", "LLM_MODEL", "EMBED_MODEL", "NUM_CTX", "KEEP_ALIVE", "TOP_K", "MAX_EVIDENCE",
            "MIN_SIM", "QUOTE_MIN_RATIO", "CACHE_SIZE", "MAX_UPLOAD_MB"]
    g = globals()
    return {k: g[k] for k in keys}
