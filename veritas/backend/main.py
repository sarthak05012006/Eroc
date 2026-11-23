"""Veritas web app: app factory, error handling and startup. Routes live in backend/routes/."""
import threading
from contextlib import asynccontextmanager

from . import config, logs

logs.setup(config.LOG_LEVEL)                     # before anything else, so loading the index is logged

from fastapi import FastAPI, Request             # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles      # noqa: E402

from . import llm, validation                    # noqa: E402
from .routes import ask, documents, scan, system  # noqa: E402

log = logs.get("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Veritas %s starting: %s", config.VERSION, config.summary())
    if config.WARMUP:
        threading.Thread(target=llm.warmup, daemon=True, name="warmup").start()
    yield


app = FastAPI(title="Veritas - Zero-Guessing Document AI", version=config.VERSION, lifespan=lifespan)
for module in (system, documents, ask, scan):
    app.include_router(module.router)


@app.exception_handler(validation.BadRequest)
async def bad_request(request: Request, exc: validation.BadRequest):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(llm.LLMError)
async def model_error(request: Request, exc: llm.LLMError):
    log.error("model error on %s: %s", request.url.path, exc)
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal error. See the server log for details."}, status_code=500)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
