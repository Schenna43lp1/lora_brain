from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model geladen!")
