import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from dataset import load_paper_dataset

def train():
    dataset = load_paper_dataset()

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_4bit=True)

    def tokenize(batch):
        inputs = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=1024)
        labels = tokenizer(batch["output"], truncation=True, padding="max_length", max_length=256)
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = dataset.map(tokenize, batched=True)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )

    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir="outputs/",
        per_device_train_batch_size=1,
        logging_steps=10,
        num_train_epochs=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained("outputs/llama-lora")

if __name__ == "__main__":
    train()
