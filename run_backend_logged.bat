@echo off
cd /d C:\Users\Ali\Downloads\etib_project\etib\backend
..\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 1>..\uvicorn.out.log 2>..\uvicorn.err.log
