#!/bin/bash
pip install -r requirements.txt
uvicorn text:app --host 0.0.0.0 --port $PORT
