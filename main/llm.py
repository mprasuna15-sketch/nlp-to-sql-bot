# llm.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"   # lightweight & works offline

class LLM:
    def __init__(self, model_id=MODEL_NAME, use_gpu_if_available=True):
        device = "cuda" if torch.cuda.is_available() and use_gpu_if_available else "cpu"
        self.device = device

        print("Loading model:", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

    def nl_to_sql(self, table_name, schema_description, question):
        prompt = f"""
Convert this natural language question into a valid SQLite SELECT query.
Only output SQL.

Table name: {table_name}
Schema:
{schema_description}

Question: {question}

SQL:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            temperature=0.0,
        )

        sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # ensure SELECT only
        if not sql.lower().startswith("select"):
            sql = "SELECT * FROM {} LIMIT 10;".format(table_name)

        if not sql.endswith(";"):
            sql += ";"

        return sql
