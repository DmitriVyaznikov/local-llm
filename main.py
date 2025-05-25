from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os
import logging
from dotenv import load_dotenv
import time
import subprocess
import threading

load_dotenv()

logging.basicConfig(level=logging.INFO)
MODEL = os.getenv('MODEL_NAME', 'LLM')
PORT = int(os.getenv('PORT', 8000))
Settings.llm = Ollama(model=MODEL, temperature=0.1,request_timeout=120)
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
mcp_proc = subprocess.Popen(
    ["npx", "-y", "@upstash/context7-mcp"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)
mcp_lock = threading.Lock()


def query_context7(prompt: str) -> str:
    with mcp_lock:
        mcp_proc.stdin.write((prompt.strip() + "\n").encode("utf-8"))
        mcp_proc.stdin.flush()
        return mcp_proc.stdout.readline().decode("utf-8").strip()



@app.post("/run")
async def run_pipeline(request: Request):
    default_prompt="Ты должен отвечать строго на русском языке, если далее не будет заявлено иного."
    data = await request.json()
    query = data.get("code", "")
    query = f"{default_prompt} {query}"
    # if "use context7" in query.lower():
    #     logging.info("📡 Обогащение через context7 MCP...")
    #     context_info = query_context7(query)
    #     query = f"Контекст: {context_info.replace('use context7', '')}\n\nЗапрос: {query.replace('use context7', '').strip()}"

    logging.info(f"QUERY: {query[:100]}...")

    start = time.time()
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
    response = query_engine.query(query)
    duration = round(time.time() - start, 2)

    return {
        "output": str(response).split("</think>")[-1].strip(),
        "time": f"{duration} сек"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)



