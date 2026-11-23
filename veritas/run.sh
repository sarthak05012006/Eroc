#!/usr/bin/env bash
# Usage: OLLAMA_HOST=http://<ip-of-mac>:11434 ./run.sh
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
