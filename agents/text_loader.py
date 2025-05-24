from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class PlainTextFileReader(BaseReader):
    def __init__(self, path: str = "data/repomix-output.txt"):
        self.path = path

    def load_data(self, *args, **kwargs) -> list[Document]:
        with open(self.path, encoding="utf-8") as f:
            text = f.read()
        return [Document(text=text, metadata={"filename": self.path})]
