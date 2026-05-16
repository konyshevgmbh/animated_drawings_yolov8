@echo off
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
echo.
echo Done. Use .venv\Scripts\python to run scripts.
