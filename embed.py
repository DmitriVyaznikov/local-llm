import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import uuid
import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('indexing.log')
    ]
)
logger = logging.getLogger(__name__)

# Конфигурационные константы
COLLECTION_NAME = "xsud-code"
EMBEDDER_MODEL = "intfloat/multilingual-e5-base"
INPUT_FILE = "data/repomix-output.txt"
CHUNK_SEPARATOR = "⋮----"
BATCH_SIZE = 100
MAX_WORKERS = 4

@dataclass
class Chunk:
    """Класс для хранения информации о чанке кода"""
    filename: str
    code: str
    embedding: Optional[List[float]] = None
    doc_id: str = ""

    def __post_init__(self):
        self.doc_id = str(uuid.uuid4())

class CodeIndexer:
    """Класс для индексации кода"""
    def __init__(self):
        self.collection, self.embedder = self._initialize_clients()
        self.current_batch: Dict[str, List] = self._create_empty_batch()

    @staticmethod
    def _initialize_clients() -> Tuple[chromadb.Collection, SentenceTransformer]:
        """Инициализация клиентов ChromaDB и SentenceTransformer"""
        try:
            client = chromadb.Client()
            collection = client.get_or_create_collection(COLLECTION_NAME)
            embedder = SentenceTransformer(EMBEDDER_MODEL)
            return collection, embedder
        except Exception as e:
            logger.error(f"Ошибка инициализации клиентов: {str(e)}")
            raise

    @staticmethod
    def _create_empty_batch() -> Dict[str, List]:
        """Создает пустой пакет для батчинга"""
        return {
            "documents": [],
            "embeddings": [],
            "ids": [],
            "metadatas": []
        }

    def load_chunks(self, file_path: str) -> List[str]:
        """Загружает чанки из файла"""
        try:
            text = Path(file_path).read_text(encoding="utf-8")
            chunks = [chunk.strip() for chunk in text.split(CHUNK_SEPARATOR) if chunk.strip()]
            logger.info(f"Загружено {len(chunks)} чанков из файла {file_path}")
            return chunks
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {file_path}: {str(e)}")
            raise

    def parse_chunk(self, chunk: str) -> Chunk:
        """Парсит чанк в структурированный объект"""
        lines = chunk.splitlines()
        filename = lines[0].strip() if lines else f"unknown_{uuid.uuid4()}"
        code = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        return Chunk(filename=filename, code=code)

    def create_embedding(self, chunk: Chunk) -> Chunk:
        """Создает эмбеддинг для чанка"""
        try:
            chunk.embedding = self.embedder.encode(chunk.code).tolist()
            return chunk
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга для {chunk.filename}: {str(e)}")
            raise

    def add_to_batch(self, chunk: Chunk) -> None:
        """Добавляет чанк в текущий батч"""
        self.current_batch["documents"].append(chunk.code)
        self.current_batch["embeddings"].append(chunk.embedding)
        self.current_batch["ids"].append(chunk.doc_id)
        self.current_batch["metadatas"].append({"source": chunk.filename})

    def flush_batch(self) -> None:
        """Сохраняет текущий батч в базу данных"""
        if self.current_batch["documents"]:
            try:
                self.collection.add(**self.current_batch)
                self.current_batch = self._create_empty_batch()
            except Exception as e:
                logger.error(f"Ошибка сохранения батча: {str(e)}")
                raise

    def index_chunks(self, chunks: List[str]) -> int:
        """Индексирует чанки с использованием многопоточности"""
        count = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Создание эмбеддингов параллельно
            future_to_chunk = {
                executor.submit(self.create_embedding, self.parse_chunk(chunk)): chunk
                for chunk in chunks
            }

            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Индексация"):
                try:
                    chunk = future.result()
                    self.add_to_batch(chunk)
                    count += 1

                    if len(self.current_batch["documents"]) >= BATCH_SIZE:
                        self.flush_batch()
                except Exception as e:
                    logger.error(f"Ошибка обработки чанка: {str(e)}")
                    continue

            # Сохраняем оставшиеся документы
            self.flush_batch()

        return count

    def run(self) -> None:
        """Запускает процесс индексации"""
        try:
            chunks = self.load_chunks(INPUT_FILE)
            count = self.index_chunks(chunks)
            logger.info(f"Индексировано {count} фрагментов")
            print(f"Индексировано {count} фрагментов.")
        except Exception as e:
            logger.error(f"Критическая ошибка: {str(e)}")
            print(f"Произошла ошибка: {str(e)}")

def main():
    indexer = CodeIndexer()
    indexer.run()

if __name__ == "__main__":
    main()
