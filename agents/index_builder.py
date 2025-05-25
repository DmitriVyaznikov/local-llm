from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from text_loader import PlainTextFileReader
import os
import logging

logging.basicConfig(level=logging.INFO)
MODEL = os.getenv('MODEL_NAME', 'LLM')
Settings.llm = Ollama(model=MODEL, temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

def build_index():
    logging.info("Строим индекс из repomix-output.txt...")
    documents = PlainTextFileReader("data/repomix-output.txt").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")
    logging.info("Индекс сохранён в папке ./storage")

if __name__ == "__main__":
    build_index()