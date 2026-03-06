 ## Mnemo
 
 Mnemo is a lightweight, self-hostable memory middleware REST API for AI agents. It provides:
 
 - **Episodic memory**: immutable raw turns stored in SQLite
 - **Semantic memory**: bi-temporal triplet edges stored in SQLite + Qdrant vectors
 - **User profile**: persistent key/value facts stored in SQLite + cached in Redis
 
 ### Quickstart
 
 1) Create a virtualenv and install dependencies (using `uv` recommended):
 
 ```bash
 uv venv
 source .venv/bin/activate
 uv pip install -e ".[dev]"
 ```
 
 2) Copy env file and set keys:
 
 ```bash
 cp .env.example .env
 ```
 
 3) Start Redis (Docker recommended):
 
 ```bash
 docker compose up -d redis
 ```
 
 4) Run the API:
 
 ```bash
 uvicorn mnemo.app.main:app --reload
 ```
 
 5) Open docs:
 
 - `http://localhost:8000/docs`
 
 ### Notes
 
 - Qdrant runs in **embedded local** mode by default (no Docker required).
 - Extraction runs asynchronously via ARQ worker (1-turn lag by design).
