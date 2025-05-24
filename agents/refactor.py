import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "qwen3:8b"

def run_refactor(code: str) -> str:
    prompt = (
            "Проанализируй и предложи улучшения для следующего кода:\n\n"
            + code +
            "\n\nОтвет дай в виде улучшенного кода с пояснениями."
    )
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt})
    return resp.text

if __name__ == "__main__":
    import sys
    sample_code = sys.stdin.read()
    print(run_refactor(sample_code))