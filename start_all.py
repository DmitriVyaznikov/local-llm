import subprocess
import os
import psutil
import signal
import sys
import time
from dotenv import load_dotenv

load_dotenv()
PORT = int(os.getenv('PORT', 8005))
api_proc = None
ui_proc = None

def is_port_in_use(port):
    """Проверка использования порта и количества процессов на нём"""
    processes = []
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                processes.append(conn.pid)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    return processes

def kill_process_on_port(port):
    """Завершение всех процессов на указанном порту"""
    processes = is_port_in_use(port)
    if processes:
        print(f"Найдено {len(processes)} процессов на порту {port}")
        for pid in processes:
            try:
                process = psutil.Process(pid)
                print(f"Завершение процесса {pid} ({process.name()}) на порту {port}")
                process.terminate()
                process.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    process.kill()
                except:
                    pass
        time.sleep(1)  # Ждем освобождения порта
        return True
    return False

def cleanup():
    """Завершение всех запущенных процессов"""
    print("\nЗавершение процессов...")

    if ui_proc and ui_proc.poll() is None:
        try:
            ui_proc.terminate()
            ui_proc.wait(timeout=3)
        except:
            ui_proc.kill()

    if api_proc and api_proc.poll() is None:
        try:
            api_proc.terminate()
            api_proc.wait(timeout=3)
        except:
            api_proc.kill()

    kill_process_on_port(PORT)
    print("Все процессы завершены")

def signal_handler(signum, frame):
    """Обработчик сигнала Ctrl+C"""
    cleanup()
    sys.exit(0)

def start_api_server():
    """Запуск API сервера с проверкой успешности"""
    global api_proc

    # Проверяем и убиваем все процессы на порту
    if is_port_in_use(PORT):
        kill_process_on_port(PORT)

    # Запуск API-сервера
    api_proc = subprocess.Popen(
        ["python", "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Ждём сигнала о запуске API
    while True:
        if api_proc.poll() is not None:  # Процесс завершился
            return False

        line = api_proc.stdout.readline()
        if not line:
            return False
        print(line.strip())
        if "Application startup complete." in line:
            # Проверяем, что процесс единственный на порту
            processes = is_port_in_use(PORT)
            if len(processes) > 1:
                print(f"Обнаружено несколько процессов на порту {PORT}, перезапуск...")
                cleanup()
                return False
            return True
        if "Traceback" in line or "Errno" in line:
            cleanup()
            return False

# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    # Пробуем запустить API сервер
    max_attempts = 3
    for attempt in range(max_attempts):
        if start_api_server():
            break
        elif attempt < max_attempts - 1:
            print(f"Попытка {attempt + 1} не удалась, повторяем...")
            time.sleep(2)
        else:
            print("Не удалось запустить API сервер после всех попыток")
            sys.exit(1)

    # Запуск UI
    ui_proc = subprocess.Popen(["python", "ui/app.py"])

    # Ожидаем завершения UI процесса
    ui_proc.wait()

except KeyboardInterrupt:
    print("\nПолучен сигнал прерывания (Ctrl+C)")
except Exception as e:
    print(f"\nПроизошла ошибка: {e}")
finally:
    cleanup()
    sys.exit(0)





