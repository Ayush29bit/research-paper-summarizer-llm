import re
import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def clean_latex(text: str) -> str:
    """
    Remove LaTeX equations, commands, and formatting artifacts.
    """
    text = re.sub(r"\$.*?\$", " ", text)  # inline math
    text = re.sub(r"\\\[.*?\\\]", " ", text)  # block math
    text = re.sub(r"\\begin{.*?}|\\end{.*?}", " ", text)
    text = re.sub(r"\\cite\{.*?\}", " ", text)
    text = re.sub(r"\\ref\{.*?\}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1024) -> List[str]:
    """
    Break long research papers into smaller chunks for LLM input.
    """
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def sanitize_text(text: str) -> str:
    """
    Clean noise, repeated newlines, and non-UTF characters.
    """
    text = text.encode("utf-8", "ignore").decode()
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\t+", " ", text)
    return text.strip()


def log(message: str):
    """
    Simple logging wrapper.
    """
    logging.info(message)
