#  Llama Fine-Tuned Research Paper Summarizer  
A domain-adapted LLaMA model fine-tuned using LoRA for summarizing academic research papers into concise abstracts.

##  Overview  
This project fine-tunes a LLaMA 7B model on research paper → abstract pairs using LoRA and QLoRA.  
The goal: generate accurate, abstract-style summaries from long scientific papers.

##  Key Features  
- LLaMA 7B fine-tuning using LoRA  
- Training with 4-bit QLoRA for GPU efficiency  
- Research paper text preprocessing (LaTeX cleanup, section extraction, chunking)  
- FastAPI inference endpoint  
- Evaluation using ROUGE & BERTScore  
- Modular codebase  

---

##  Tech Stack
- Python  
- HuggingFace Transformers  
- PEFT (LoRA / QLoRA)  
- PyTorch  
- FastAPI  
- Uvicorn  
- Jupyter  

---

##  Installation

```bash
pip install -r requirements.txt

