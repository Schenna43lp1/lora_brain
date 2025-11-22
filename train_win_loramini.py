import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen1.5-0.5B"   # <<< NUR 0.5B VERSION
DATA = r"D:\dataset\training.jsonl"
OUT = r"D:\dataset\trained_model"

print("🔤 Lade Tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("🧠 Lade Base-Modell…")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🔩 LoRA Setup…")
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)

print("📄 Lade Dataset…")
dataset = load_dataset("json", data_files=DATA, split="train")


def preprocess(x):
    text = x["input"] + "\n" + x["output"]
    return tokenizer(text, truncation=True, max_length=512)  # kleiner für 0.5B


dataset = dataset.map(preprocess)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

print("🚀 Starte Training…")
args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # 0.5B → mehr Accum für stabileres Training
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    save_steps=150,
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=dataset,
)

trainer.train()

print("🎉 Fertig!")
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
