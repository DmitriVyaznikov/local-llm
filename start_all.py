import subprocess

# Запуск API-сервера и ожидание запуска
api_proc = subprocess.Popen(
    ["python", "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Ждём сигнала о запуске
while True:
    line = api_proc.stdout.readline()
    print(line.strip())
    if "Application startup complete." in line:
        break

# Запуск UI
subprocess.call(["python", "ui/app.py"])