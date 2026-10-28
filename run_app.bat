@echo off
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Starting Malayalam Agentic AI Diagnostic Engine...
py -X utf8 run_app.py
pause

