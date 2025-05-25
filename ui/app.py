
import gradio as gr
import requests
import time
import threading
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv('API_URL', 'http://localhost:8000/run')
MODEL = os.getenv('MODEL_NAME', 'LLM')
PROJECT_NAME = os.getenv('PROJECT_NAME', 'Local LLM')
first_run = True

def run_analysis(code, mode):
    global first_run
    start = time.time()
    response_data = None

    if first_run:
        waiting_for_response = True

        def make_api_call():
            nonlocal waiting_for_response, response_data
            payload = {"code": code}
            try:
                response = requests.post(API_URL, json=payload).json()
                response_data = response
            finally:
                waiting_for_response = False

        thread = threading.Thread(target=make_api_call)
        thread.start()

        # Показываем таймер, пока ждем ответ
        while waiting_for_response:
            time_passed = round(time.time() - start, 2)
            yield f"⏳ Ожидайте ответа... ⏱️ {time_passed} сек"
            time.sleep(0.1)

        first_run = False

        # Формируем ответ после получения данных
        if response_data:
            duration = response_data.get("time", f"⏱️ {round(time.time() - start, 2)} сек")
            yield f"⏱️ Время ответа: {duration}\n---\n{response_data.get('output', '[Нет ответа]')}"
        return

    # Для последующих запросов
    payload = {"code": code}
    response = requests.post(API_URL, json=payload).json()
    duration = response.get("time", f"⏱️ {round(time.time() - start, 2)} сек")
    yield f"⏱️ Время ответа: {duration}\n---\n{response.get('output', '[Нет ответа]')}"

with gr.Blocks() as demo:
    gr.Markdown(f"# ⚙️ {MODEL.upper()} — LLM для проекта {PROJECT_NAME}")

    code_input = gr.Textbox(label="Запрос или код", lines=12, placeholder="Что делает компонент Input?")
    run_btn = gr.Button("🚀 Запустить")
    output_md = gr.Markdown()

    run_btn.click(fn=run_analysis, inputs=[code_input], outputs=output_md)

demo.launch()

