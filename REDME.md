### 🧠 LoRA Brain – Windows Qwen1.5 Trainer

Ein ultraschneller, einfacher LoRA-Trainer für Windows 10/11.
Optimiert für NVIDIA-GPUs, Python 3.11 und lokale Modelle.

Dies ist die sauberste & stabilste Windows-LoRA-Pipeline, ideal für kleine Datensets, Chat-Finetuning oder Custom-Assistants.

### 📦 Voraussetzungen
✔ GPU

### NVIDIA GPU (4–12GB VRAM empfohlen)

Neueste NVIDIA-Treiber

✔ Software

Python 3.11 installiert

PowerShell geöffnet

Virtuelle Umgebung erstellt:
´´´
python3.11 -m venv venv
.\venv\Scripts\activate
´´´
✔ Notwendige Pakete installieren
pip install transformers datasets accelerate peft bitsandbytes sentencepiece

### 📁 Projektstruktur
lora_brain/
│
├── train_win_lora.py      # Haupt-Training-Script
├── training.jsonl         # Dein Trainings-Dataset
└── models/
      └── Qwen1.5-1.8B-Chat/   # Lokal entpacktes Modell

### 📚 Dataset-Format (training.jsonl)

Jede Zeile:
```
{"input": "Frage des Nutzers", "output": "Antwort des Modells"}
``` 

Beispiel:

{"input": "Was ist 2+2?", "output": "Die Antwort ist 4."}

### 🚀 Training starten
.\venv\Scripts\activate
python train_win_lora.py

### 🛠 Was das Script macht

Lädt dein lokales Modell

Aktiviert LoRA (q_proj, v_proj, k_proj, o_proj)

Tokenisiert dein Dataset stabil für Windows

Startet ein schnelles Training (FP16)

Speichert das fertige LoRA-Modell in:

./output_lora/
