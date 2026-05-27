# AI Cyberbullying Detector Startup Script
Write-Host "Starting AI Cyberbullying Detection System..." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# Set environment variable
$env:PYTHONPATH = $PWD

Write-Host "`n1. Testing the model first..." -ForegroundColor Yellow
.venv\Scripts\python.exe test_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model test failed! Please check your setup." -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Starting FastAPI backend..." -ForegroundColor Yellow
Write-Host "   Backend will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "   API documentation at: http://localhost:8000/docs" -ForegroundColor Cyan

# Start API in background
Start-Job -ScriptBlock {
    Set-Location $args[0]
    $env:PYTHONPATH = $PWD
    .venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
} -ArgumentList $PWD -Name "CyberbullyAPI"

# Wait a moment for API to start
Start-Sleep -Seconds 3

Write-Host "`n3. Starting Streamlit frontend..." -ForegroundColor Yellow
Write-Host "   Frontend will be available at: http://localhost:8501" -ForegroundColor Cyan

# Start Streamlit
.venv\Scripts\python.exe -m streamlit run src\frontend\streamlit_app.py

Write-Host "`nShutting down background services..." -ForegroundColor Yellow
Get-Job -Name "CyberbullyAPI" | Remove-Job -Force

Write-Host "All services stopped. Thank you for using AI Cyberbullying Detector!" -ForegroundColor Green