# main.py
import os
import sqlite3
import pandas as pd
import tempfile
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from llm import LLM

app = FastAPI(title="NL → SQL (local Mistral)")

# Serve frontend from static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# We'll keep track of the latest uploaded table in a temp sqlite DB
TEMP_DB = "uploaded_data.db"
TABLE_NAME = "user_table"

# lazy init LLM
llm = None
def get_llm():
    global llm
    if llm is None:
        # try to use GPU if available; set use_gpu_if_available=False if you want CPU only
        llm = LLM(use_gpu_if_available=True)
    return llm

def sanitize_table_name(name: str) -> str:
    return re.sub(r"[^\w_]", "_", name) or "user_table"

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), table: str = Form(None)):
    """
    Accepts a CSV upload and imports it into a local SQLite DB as a table.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        return JSONResponse({"error": "Could not read CSV: " + str(e)}, status_code=400)

    table_name = sanitize_table_name(table or TABLE_NAME)

    # write to sqlite
    conn = sqlite3.connect(TEMP_DB)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    # create schema description
    cols = df.dtypes.apply(lambda x: x.name).to_dict()
    schema_lines = []
    for c, dtype in cols.items():
        schema_lines.append(f"- {c}: {dtype}")
    schema_description = "\n".join(schema_lines)

    return {"message": "File uploaded", "table_name": table_name, "rows": len(df), "schema": schema_description}

@app.post("/ask")
async def ask_question(question: str = Form(...), table: str = Form(None)):
    """
    Convert natural language to SQL via local LLM, then validate and execute the SQL against the sqlite DB.
    """
    table_name = sanitize_table_name(table or TABLE_NAME)
    # read schema from sqlite
    conn = sqlite3.connect(TEMP_DB)
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name});")
        info = cur.fetchall()
        if not info:
            return JSONResponse({"error": f"Table `{table_name}` not found. Upload a CSV first."}, status_code=400)
        cols = [row[1] for row in info]
        types = [row[2] for row in info]
        schema_description = "\n".join([f"- {c}: {t}" for c, t in zip(cols, types)])
    finally:
        conn.close()

    # call LLM
    model = get_llm()
    try:
        generated_sql = model.nl_to_sql(table_name, schema_description, question)
    except Exception as e:
        return JSONResponse({"error": "LLM error: " + str(e)}, status_code=500)

    # safety checks: only allow SELECT
    if not re.match(r"(?i)^\s*SELECT\b", generated_sql):
        return JSONResponse({"error": "Generated SQL is not a SELECT statement. Aborting for safety.", "sql": generated_sql}, status_code=400)

    # strip dangerous keywords just in case
    forbidden = ["ATTACH", "DETACH", "PRAGMA", "VACUUM", "DROP", "ALTER", "INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE"]
    for kw in forbidden:
        if re.search(rf"(?i)\b{kw}\b", generated_sql) and not re.match(rf"(?i)^\s*SELECT\b", generated_sql):
            return JSONResponse({"error": f"Generated SQL contains forbidden keyword: {kw}", "sql": generated_sql}, status_code=400)

    # Ensure a LIMIT exists (to avoid huge outputs)
    if not re.search(r"(?i)\bLIMIT\b", generated_sql):
        # append LIMIT 100
        generated_sql = generated_sql.rstrip().rstrip(";") + " LIMIT 100;"

    # execute
    conn = sqlite3.connect(TEMP_DB)
    try:
        cur = conn.execute(generated_sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(200)  # fetch capped rows
    except Exception as e:
        return JSONResponse({"error": "SQL execution error: " + str(e), "sql": generated_sql}, status_code=400)
    finally:
        conn.close()

    # Return SQL and results
    results = [dict(zip(cols, r)) for r in rows]
    return {"sql": generated_sql, "columns": cols, "rows": results}

@app.get("/")
def index():
    # serve the frontend single page
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
