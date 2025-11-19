import re
from datasets import load_dataset

def clean_text(text):
    text = re.sub(r'\$.*?\$', '', text)      # Remove inline LaTeX
    text = re.sub(r'\\begin{.*?}|\\end{.*?}', '', text)  
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def chunk_text(text, chunk_size=1024):
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
def load_arxiv_dataset(split="train"):
    dataset = load_dataset("arxiv_dataset", split=split)
    return dataset