"""Startup script for the arq ingestion worker — forces UTF-8 encoding."""
import sys
import os
import io

# Force UTF-8 on all standard streams before anything else
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ["PYTHONIOENCODING"] = "utf-8"

import asyncio
from arq import run_worker
from backend.pipeline.ingestion_worker import WorkerSettings

if __name__ == "__main__":
    asyncio.run(run_worker(WorkerSettings))
