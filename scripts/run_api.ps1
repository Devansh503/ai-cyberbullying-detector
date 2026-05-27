param([int]$Port=8000)
uvicorn src.api.main:app --host 0.0.0.0 --port $Port --workers 2
