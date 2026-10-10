# AskMyData

**Upload a messy spreadsheet or database, clean it with a human approving every step, then ask it questions in plain English.**

AskMyData is a human-in-the-loop platform for reliable natural-language-to-SQL. Small businesses keep their data in Excel files with merged headers, mixed types, duplicate rows and undocumented relationships, and they can't afford a data engineer to make it queryable. AskMyData repairs the data, recovers its meaning and structure, exports it to PostgreSQL with an embedded semantic layer, and answers questions against it with charts and a live dashboard.

It is built on two CHI '26 papers: **SemTabla** (semantic enrichment and validation of data tables) and **Cocoa** (human-agent co-planning).

## How it works

```
messy source
   │
   ▼
0 Ingest ─► 1 Semantics ─► 2 Clean ─► 3 Relationships ─► 4 Export ─► 5 Ask ─► 6 Dashboard
```

Phases 0–3 end at a human checkpoint: you review and correct what the algorithms found before it reaches the database.

| Phase | What it does |
|---|---|
| **0. Ingestion** | Loads Excel (`.xlsx`/`.xlsm`), CSV/TSV, SQLite files, live PostgreSQL/MySQL connections and SQL dumps. Flattens multi-row merged headers, normalises nulls and currency, detects equivalent columns across sheets with embeddings, discovers minimal primary keys, and triages every table (clean / fixable / needs attention). |
| **1. Semantics** | Classifies each column into one of 9 types and a semantic taxonomy label (rule engine first, LLM for the ambiguous rest), profiles each table, and shows everything in an editable grid. |
| **2. Cleaning** | A LangGraph agent proposes a cleaning plan; you reorder, edit or reject steps, then approve each execution on a before/after view. State is checkpointed, so a session survives closing the browser. |
| **3. Relationships** | Detects primary keys, foreign keys and functional dependencies. Every finding comes with positive and negative sample rows and the exact SQL behind them, shown on a force-directed D3 diagram where you confirm or reject each edge. |
| **4. Export** | Materialises the cleaned tables into a per-dataset PostgreSQL schema with real `PRIMARY KEY`/`FOREIGN KEY` constraints, plus a `sem_metadata` table holding each column's type, label, description and a pgvector embedding. A portable JSON bundle is also available. |
| **5. NL2SQL** | Retrieves the relevant tables and columns, builds a prompt from schema and semantic metadata, generates SQL, validates it with `EXPLAIN` and retries with the error on failure (up to 3 attempts), then renders the result as a metric card, chart or table. |
| **6. Dashboard** | Pin any answer as a card that re-executes against the live database, keep a history of recent questions, get suggested questions, and export the dashboard to PDF. |

## Design decisions

- **The model never sees data rows.** Taxonomy labelling gets one column's name and up to 10 sample values; cleaning plans get statistical summaries; SQL generation gets schema and metadata. Computation happens in PostgreSQL, so a ten-million-row table costs the same prompt as a hundred-row one.
- **Declared schemas are trusted.** For SQLite, PostgreSQL/MySQL and SQL-dump sources, declared types, stored values and declared keys are kept as ground truth rather than re-guessed. Changing them is a reviewed cleaning step.
- **Generated SQL is defence-in-depth guarded.** A `sqlglot`-based `SELECT`-only guard runs first, and queries execute as a `semantic_readonly` PostgreSQL role that cannot write even if the guard were bypassed.
- **Low-confidence answers abstain.** A generated query below the confidence threshold is withheld rather than shown as an answer.
- **Graceful degradation.** Without an LLM key, taxonomy falls back to the rule engine, descriptions to a deterministic template and the cleaning plan to a built-in statistical planner. Asking questions needs a model.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS, D3.js (relationship diagram), Recharts, react-grid-layout |
| Backend | FastAPI, SQLAlchemy 2, pandas, openpyxl, rapidfuzz, sqlglot |
| Agent orchestration | LangGraph with PostgreSQL checkpointing |
| Storage and vectors | PostgreSQL 16 + pgvector (SQLite supported for quick local runs) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local, CPU) |
| LLM | Any model on [OpenRouter](https://openrouter.ai) through its OpenAI-compatible API |

## Getting started

**Prerequisites:** Python 3.12, Node.js, and Docker (for PostgreSQL + pgvector).

```bash
# 1. Start PostgreSQL with pgvector (port 5433) and the read-only query role
docker compose up -d db

# 2. Configure
cp .env.example .env
# set OPENROUTER_API_KEY (and optionally OPENROUTER_MODEL) in .env

# 3. Start the backend (:8000) and frontend (:5173) together
./run.sh
```

`run.sh` creates `.venv` and installs `backend/requirements.txt` and the frontend dependencies on first run. Open <http://localhost:5173>, register an account, and create a session.

To skip Docker, point `DATABASE_URL` at a SQLite file (`sqlite:///./backend/var/semanticlayer.db`). Everything works except pgvector similarity search, which falls back to JSON embeddings.


## Evaluation on BIRD

`backend/evaluation/bird/` benchmarks the full pipeline against a raw baseline on the [BIRD](https://bird-bench.github.io/) dev set, scoring by execution accuracy against the gold SQL.

- **Raw arm:** the model sees only table and column names.
- **Semantic arm:** a real AskMyData session answers the question, with the semantic layer, retrieval and saved examples.
- **Setup:** 6 BIRD dev databases (`formula_1`, `student_club`, `thrombosis_prediction`, `california_schools`, `toxicology`, `superhero`), 3 attempts per question, and `google/gemini-3.5-flash-lite` as the model in both arms. "Relaxed" accuracy tolerates up to 3 surplus result columns; "strict" requires the exact result.

| Run | Questions | Raw (strict / relaxed) | Semantic (strict / relaxed) | Gain (strict) |
|---|---|---|---|---|
| With BIRD evidence hints | 299 | 51.8% / 53.8% | 58.2% / 60.5% | +6.4 pts (McNemar p = 0.010) |
| No evidence hints | 300 | 30.0% / 32.3% | 46.3% / 48.7% | +16.3 pts |
| No evidence hints, larger sample | 858 | 39.9% / 42.8% | 56.3% / 58.9% | +16.4 pts |

The semantic layer helps most when BIRD's hand-written hints are removed, which is the situation a real user is in. Raw per-question records, configs and summaries are in `backend/evaluation/bird/results/`. These are single runs on a small model, so read the confidence intervals in each `summary.json` before quoting a number.

```bash
cd backend
python -m evaluation.bird.download          # fetch the BIRD dev set (~330 MB)
python -m evaluation.bird.selftest          # whole harness with a fake model: no network, no key, no cost
python -m evaluation.bird.run_eval --run-name bird300 --batch-size 100   # real run: spends API credit; re-run to continue
python -m evaluation.bird.report --run-name bird300                      # re-read any run's numbers, free
```

## Project structure

```
backend/
  app/
    ingestion/       Phase 0 — loaders, header repair, equivalence, key discovery
    semantics/       Phase 1 — column types, taxonomy, profiling, LLM client
    cleaning/        Phase 2 — LangGraph plan / review / execute / validate
    relationships/   Phase 3 — PK, FK, functional dependencies, evidence
    export/          Phase 4 — PostgreSQL schema, sem_metadata, quality report
    query/           Phase 5 — retrieval, SELECT-only guard, result shaping
    api/, auth/      FastAPI routes, accounts, sessions
  evaluation/bird/   BIRD benchmark harness
  tests/
frontend/src/        React app (one panel per phase)
ops/                 PostgreSQL init scripts (pgvector, read-only role)
docker-compose.yml   PostgreSQL + pgvector
run.sh               start backend and frontend
```

## References

- Jin, Z., Mi, Y., Zhu, Y., Yao, Y., Yu, C., & Xu, K. (2026). *SemTabla: A Human-in-the-Loop Framework for Semantic Enrichment and Validation of Data Tables.* CHI '26.
- Feng, K.J.K., Pu, K., Latzke, M., August, T., Siangliulue, P., Bragg, J., Weld, D.S., Zhang, A.X., & Chang, J.C. (2026). *Cocoa: Co-Planning and Co-Execution with AI Agents.* CHI '26.
