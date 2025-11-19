from datasets import Dataset
from preprocess import clean_text, chunk_text

def load_paper_dataset():
    data = [
        {"paper": open("data/samples/sample_paper.txt").read(),
         "summary": open("data/samples/sample_summary.txt").read()}
    ]

    processed = []
    for item in data:
        chunks = chunk_text(clean_text(item["paper"]))
        summary = clean_text(item["summary"])

        for c in chunks:
            processed.append({"input": c, "output": summary})

    return Dataset.from_list(processed)
