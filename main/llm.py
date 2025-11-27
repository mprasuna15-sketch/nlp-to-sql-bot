# llm.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

HF_TOKEN_ENV = "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"
MODEL_ID = "mistral/mistral-7b-instruct"  # official HF identifier (subject to change)

def ensure_logged_in():
    token = os.environ.get(HF_TOKEN_ENV)
    if token:
        try:
            login(token=token)
        except Exception as e:
            print("HF login failed:", e)
    else:
        print("Warning: HUGGINGFACE_HUB_TOKEN not set. If the model is private you'll need to set it.")

class LLM:
    def __init__(self, model_id=MODEL_ID, use_gpu_if_available=True):
        ensure_logged_in()
        self.model_id = model_id
        # load tokenizer and model
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        print("Loading model (this may take a while)...")
        # Auto device map: will put on GPU if available and supported
        device_map = "auto" if use_gpu_if_available else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,  # some HF community models require this
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map=device_map)

    def nl_to_sql(self, table_name, schema_description, question, max_new_tokens=256, temperature=0.0):
        """
        Returns SQL (string) for the user's question. We prompt the model to only output the SQL.
        """
        prompt = f"""
You are an expert SQL generator. The database table is named `{table_name}` with these columns:
{schema_description}

Convert the user's natural language question into a single valid SQLite SQL query. ONLY output the SQL query, and nothing else — no explanations, no extra text.

The SQL must:
- Be valid SQLite SQL
- Only use the table `{table_name}`
- Only be a SELECT statement (no INSERT/UPDATE/DELETE/ATTACH/etc)
- Limit results if appropriate (you may add "LIMIT 100" at the end if missing)

User question:
\"\"\"{question}\"\"\"
"""
        # call the model
        out = self.generator(prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
        text = out[0]["generated_text"]
        # The model often echoes the prompt; we extract the last portion after the prompt.
        # We'll attempt to extract the SQL by finding the first "SELECT" ... 
        import re
        m = re.search(r"(?i)(SELECT\b[\s\S]*)", text)
        if m:
            sql = m.group(1).strip()
        else:
            # fallback to whole generated text trimmed
            sql = text.strip()
        # ensure it ends with semicolon
        if not sql.endswith(";"):
            sql = sql + ";"
        return sql
