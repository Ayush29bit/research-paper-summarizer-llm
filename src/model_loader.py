from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

def load_model():
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    base = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_4bit=True)
    model = PeftModel.from_pretrained(base, "outputs/llama-lora")
    return tokenizer, model