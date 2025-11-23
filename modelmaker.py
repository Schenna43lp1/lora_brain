import os
import subprocess
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ------------------------------------------------------------
# EINSTELLUNGEN
# ------------------------------------------------------------
# WICHTIG: BASE_MODEL muss mit dem Modell übereinstimmen, mit dem der LoRA-Adapter trainiert wurde!
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Laut adapter_config.json
MODEL_DIR = Path(r"D:\dataset\trained_model")
EXPORT_DIR = Path(r"D:\dataset\final_ollama")
MODEL_NAME = "markusbrainlora"  # Name für Ollama und GGUF
# ------------------------------------------------------------

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print("🔤 Lade Tokenizer…")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
except Exception as e:
    print(f"❌ Fehler beim Laden des Tokenizers: {e}")
    sys.exit(1)

print("🧠 Lade Base-Modell…")
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=False
    )
except Exception as e:
    print(f"❌ Fehler beim Laden des Base-Modells: {e}")
    sys.exit(1)

adapter_config = MODEL_DIR / "adapter_config.json"

# ------------------------------------------------------------
# Prüfen ob LoRA oder Full Model
# ------------------------------------------------------------
if adapter_config.exists():
    print("🔗 LoRA erkannt → Merging…")
    try:
        model = PeftModel.from_pretrained(
            model,
            str(MODEL_DIR),
            local_files_only=True
        )
        model = model.merge_and_unload()
        print("✅ LoRA Adapter erfolgreich gemerged!")
    except Exception as e:
        print(f"❌ Fehler beim Mergen der LoRA-Adapter: {e}")
        sys.exit(1)
else:
    print("⚠️ Keine LoRA-Dateien gefunden.")
    print("➡️ Lade Modell als FULL HF Modell…")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            torch_dtype="auto",
            device_map="cpu",
            local_files_only=True
        )
    except Exception as e:
        print(f"❌ Fehler beim Laden des Full Models: {e}")
        sys.exit(1)

# ------------------------------------------------------------
# SAVE MERGED MODEL
# ------------------------------------------------------------
MERGED_DIR = EXPORT_DIR / "merged_model"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

print("💾 Speichere gemerged Model…")
try:
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"✅ Modell gespeichert in: {MERGED_DIR}")
except Exception as e:
    print(f"❌ Fehler beim Speichern des Modells: {e}")
    sys.exit(1)

# ------------------------------------------------------------
# EXPORT GGUF (Benötigt llama.cpp)
# ------------------------------------------------------------
GGUF_OUT = EXPORT_DIR / f"{MODEL_NAME}.gguf"

print("\n📦 Konvertiere nach GGUF…")
print("⚠️ HINWEIS: Dies benötigt llama.cpp installiert!")
print("   Installiere llama.cpp mit: pip install llama-cpp-python")
print("   Oder klone: git clone https://github.com/ggerganov/llama.cpp\n")

# Versuche mit llama.cpp convert.py
convert_script = Path("llama.cpp/convert.py")
if convert_script.exists():
    try:
        result = subprocess.run([
            sys.executable,
            str(convert_script),
            str(MERGED_DIR),
            "--outfile",
            str(GGUF_OUT),
            "--outtype",
            "f16"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ GGUF erfolgreich erstellt: {GGUF_OUT}")
        else:
            print(f"⚠️ GGUF Konvertierung fehlgeschlagen:")
            print(result.stderr)
    except Exception as e:
        print(f"⚠️ GGUF Konvertierung übersprungen: {e}")
        print("   → Du kannst das Modell manuell mit llama.cpp konvertieren")
else:
    print("⚠️ llama.cpp/convert.py nicht gefunden - überspringe GGUF Konvertierung")
    print(f"   → Merged Model verfügbar in: {MERGED_DIR}")

# ------------------------------------------------------------
# OLLAMA Modellfile (nur wenn GGUF existiert)
# ------------------------------------------------------------
if GGUF_OUT.exists():
    modelfile = f"""FROM {MODEL_NAME}.gguf
TEMPLATE \"\"\"You are {MODEL_NAME}. Respond clearly.
{{{{ .Prompt }}}}\"\"\"
PARAMETER temperature 0.4
PARAMETER top_p 0.9
"""

    modelfile_path = EXPORT_DIR / "Modelfile"
    try:
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile)
        print(f"✅ Modelfile erstellt: {modelfile_path}")
    except Exception as e:
        print(f"⚠️ Fehler beim Erstellen des Modelfiles: {e}")

    print("\n🐪 Registriere Modell in Ollama…")
    try:
        result = subprocess.run([
            "ollama",
            "create",
            MODEL_NAME,
            "-f",
            str(modelfile_path)
        ], capture_output=True, text=True, cwd=str(EXPORT_DIR))
        
        if result.returncode == 0:
            print("✅ Modell in Ollama registriert!")
            print(f"\n🎉 Fertig! Starte mit:")
            print(f"👉 ollama run {MODEL_NAME}")
        else:
            print(f"⚠️ Ollama Registrierung fehlgeschlagen:")
            print(result.stderr)
            print(f"\nManuell registrieren mit:")
            print(f"cd {EXPORT_DIR}")
            print(f"ollama create {MODEL_NAME} -f Modelfile")
    except FileNotFoundError:
        print("⚠️ Ollama nicht gefunden. Installiere Ollama von https://ollama.ai")
    except Exception as e:
        print(f"⚠️ Fehler bei Ollama-Registrierung: {e}")
else:
    print("\n⚠️ GGUF-Datei nicht gefunden - Ollama-Registrierung übersprungen")

print(f"\n✅ Prozess abgeschlossen!")
print(f"📁 Merged Model: {MERGED_DIR}")
if GGUF_OUT.exists():
    print(f"📦 GGUF: {GGUF_OUT}")