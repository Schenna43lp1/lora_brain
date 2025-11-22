from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen1.5-1.5B-Instruct"
LORA = r"D:\dataset\trained_model"
OUT = r"D:\dataset\ollama_export"

model = AutoModelForCausalLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, LORA)

model = model.merge_and_unload()

model.save_pretrained(OUT)
print("Fertig gemerged!")
