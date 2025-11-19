from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import summarize

app = FastAPI()

class Paper(BaseModel):
    text: str

@app.post("/summarize")
def get_summary(paper: Paper):
    return {"summary": summarize(paper.text)}