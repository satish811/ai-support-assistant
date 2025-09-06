# AI-Powered Communication Assistant (Hackathon Demo)
visualisation dashboard

 <img width="1920" height="2257" alt="127 0 0 1_5000-AI Communication Assistant - Demo-fpscreenshot" src="https://github.com/user-attachments/assets/c5874b0d-ef4f-42fe-acea-65a2eae3b4bf" />

This is a lightweight, end-to-end demo of the **AI-Powered Communication Assistant** tailored for a 4-day hackathon. It uses a local CSV of sample support emails (provided) instead of live IMAP access so you can run it locally without giving mail credentials.

## Features (implemented in demo)
- Import & filter emails from CSV (subject contains Support/Query/Request/Help)
- Display sender, subject, body, date/time
- Simple categorization: Sentiment (heuristic), Priority (keyword-based)
- Keyword and contact extraction (phone/email)
- RAG-style KB retrieval using TF-IDF (local `kb.md`)
- Draft reply generation using OpenAI (if `OPENAI_API_KEY` provided) else template fallback
- Dashboard with table, analytics (Chart.js), generate/edit responses, mark resolved
- SQLite storage to persist email state

## Requirements
- Python 3.9+
- Install requirements: `pip install -r requirements.txt`

## Files
- `app.py` - Flask backend (API + simple web UI)
- `templates/index.html` - Dashboard UI
- `static/app.js` - Frontend JS for interaction
- `static/style.css` - Basic styles
- `sample_kb.md` - Example knowledge base used for RAG retrieval
- `requirements.txt` - Python deps
- `run_demo.sh` - Simple run script for Linux/macOS (Windows: run `python app.py`)
- `README.md` - This file

## How to run locally (Linux / macOS / WSL)
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place the provided sample CSV in `/mnt/data/68b1acd44f393_Sample_Support_Emails_Dataset.csv` (already present in this environment for the demo).
3. (Optional) Export your OpenAI key to enable better reply generation:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
   On Windows (PowerShell):
   ```powershell
   setx OPENAI_API_KEY "sk-..."
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open browser at `http://127.0.0.1:5000` and click **Import CSV** to load demo emails.
