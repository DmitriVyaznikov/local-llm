import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "qwen3:8b"

def generate_tests(code: str) -> str:
    prompt = f"Сгенерируй unit-тесты для следующего кода с использованием Jest или Testing Library:\n\n{code}\n\nОтвет должен содержать тестовый файл или функции."
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt})
    return resp.text

if __name__ == "__main__":
    import sys
    sample_code = sys.stdin.read()
    print(generate_tests(sample_code))