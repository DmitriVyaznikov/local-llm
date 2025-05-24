import requests
import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from requests.exceptions import RequestException
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурационные константы
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
COLLECTION_NAME = "xsud-code"
EMBEDDER_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_N_RESULTS = 5
ERROR_MESSAGE = "Недостаточно информации в кодовой базе. Пожалуйста, уточни вопрос."

def initialize_clients() -> Tuple[chromadb.Collection, SentenceTransformer]:
    """
    Инициализирует клиенты ChromaDB и SentenceTransformer.

    Returns:
        Tuple[chromadb.Collection, SentenceTransformer]: Коллекция и модель эмбеддинга

    Raises:
        RuntimeError: При ошибке инициализации клиентов
    """
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(COLLECTION_NAME)
        embedder = SentenceTransformer(EMBEDDER_MODEL)
        return collection, embedder
    except Exception as e:
        raise RuntimeError(f"Ошибка инициализации клиентов: {str(e)}")

def get_relevant_documents(
        question: str,
        embedder: SentenceTransformer,
        collection: chromadb.Collection
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Получает релевантные документы из базы знаний.

    Args:
        question: Вопрос для поиска
        embedder: Модель для создания эмбеддингов
        collection: Коллекция документов

    Returns:
        Tuple[List[str], List[Dict]]: Список документов и их метаданные
    """
    query_vector = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=DEFAULT_N_RESULTS
    )
    return results["documents"][0], results["metadatas"][0]

def format_context(
        docs: List[str],
        metas: List[Dict[str, Any]],
        rich_context: bool
) -> Tuple[str, str]:
    """
    Форматирует контекст и системное сообщение.

    Args:
        docs: Список документов
        metas: Список метаданных
        rich_context: Флаг для расширенного контекста

    Returns:
        Tuple[str, str]: Отформатированный контекст и системное сообщение
    """
    if rich_context:
        context = "\n\n".join([
            f"Файл {meta.get('source', 'unknown')}:\n{doc}"
            for doc, meta in zip(docs, metas)
        ])
        system = "Ты эксперт по проекту XSUD. Ответь на вопрос, используя знания из найденных фрагментов."
    else:
        context = "\n\n".join(docs)
        system = "Ответь на вопрос на основе найденного кода."

    return context, system

def query_llm(prompt: str) -> str:
    """
    Отправляет запрос к LLM модели.

    Args:
        prompt: Подготовленный промпт

    Returns:
        str: Ответ модели или сообщение об ошибке
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt},
            timeout=30
        )
        response.raise_for_status()
        return response.text
    except RequestException as e:
        return f"Ошибка при обращении к API: {str(e)}"

def contextual_query(question: str, rich_context: bool = False) -> str:
    """
    Выполняет контекстный поиск и генерирует ответ на вопрос.

    Args:
        question: Вопрос для поиска
        rich_context: Флаг для использования расширенного контекста

    Returns:
        str: Ответ на вопрос или сообщение об ошибке
    """
    if not question.strip():
        return "Вопрос не может быть пустым"

    try:
        # Получение релевантных документов
        docs, metas = get_relevant_documents(question, embedder, collection)

        if not docs or all(not doc.strip() for doc in docs):
            return ERROR_MESSAGE

        # Логирование найденных источников
        sources = [meta.get("source", "unknown") for meta in metas]
        logger.info(f"Контекстные фрагменты: {sources}")

        # Подготовка и отправка запроса
        context, system = format_context(docs, metas, rich_context)
        prompt = f"{system}\n\nКонтекст:\n{context}\n\nВопрос: {question}\nОтвет:"

        return query_llm(prompt)

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return f"Произошла ошибка: {str(e)}"

def main():
    """Основная функция для CLI интерфейса"""
    try:
        question = sys.stdin.read().strip()
        print(contextual_query(question, rich_context=True))
    except KeyboardInterrupt:
        logger.info("Программа прервана пользователем")
        print("\nПрограмма прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        print(f"Ошибка выполнения: {str(e)}")

# Инициализация клиентов при импорте модуля
collection, embedder = initialize_clients()

if __name__ == "__main__":
    import sys
    main()
