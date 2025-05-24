import gradio as gr
import requests
import time

API_URL = "http://localhost:8000/run"
MODEL = requests.get("http://localhost:8000/model").json().get("model", "LLM")
first_run = True
def run_agents(code, mode):
    global first_run
    if first_run:
        yield "⏳ Ожидайте ответа..."
        first_run = False
    payload = {"code": code}
    start = time.time()
    response = requests.post(API_URL, json=payload).json()
    duration = response.get("time", f"⏱️ {round(time.time() - start, 2)} сек")
    yield f"⏱️ Время ответа: {duration}\n---\n{response.get('output', '[Нет ответа]')}"

with gr.Blocks() as demo:
    gr.Markdown(f"# ⚙️ {MODEL.upper()} — LLM для проекта XSUD")
    # gr.Markdown("Вставь запрос — LLM выдаст ответ, используя документ `repomix-output.txt`")

    code_input = gr.Textbox(label="Запрос или код", lines=12, placeholder="Что делает компонент XsudInput.vue?")
    # mode_input = gr.Radio(["all"], label="Тип агента", value="all", visible=False)
    run_btn = gr.Button("🚀 Запустить")
    output_md = gr.Markdown()

    run_btn.click(fn=run_agents, inputs=[code_input], outputs=output_md)

demo.launch()
