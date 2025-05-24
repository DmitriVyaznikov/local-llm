import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "qwen3:8b"

def generate_docs(code: str) -> str:
    prompt = f"Проанализируй следующий код и сгенерируй JSDoc-комментарии, а если это модуль/компонент — начальный README.md:\n\n{code}\n\nОтвет должен содержать документацию."
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt})
    return resp.text

if __name__ == "__main__":
    import sys
    sample_code = sys.stdin.read()
    print(generate_docs(sample_code))