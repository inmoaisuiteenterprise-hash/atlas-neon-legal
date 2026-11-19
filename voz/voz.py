# Narración de una escena con Chatterbox Multilingual (Resemble AI, licencia MIT).
# Voces de referencia: lecturas de los corpus abiertos OpenSLR 61/71/72 (Google, CC BY-SA 4.0), en voz/refs.
# Una frase por línea, comprobación con Whisper (se repite si no se entiende) y tiempos por palabra.
# Uso: python voz.py <guion.json> <referencia.wav> <salida_dir>
import sys, os, json, subprocess, re, tempfile
G, REF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
g = json.load(open(G, encoding="utf-8"))
if g.get("ref"): REF = os.path.join(os.path.dirname(__file__), "refs", g["ref"])
import torch, torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from faster_whisper import WhisperModel
torch.set_num_threads(max(1, os.cpu_count() or 1))
model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
wm = WhisperModel("small", device="cpu", compute_type="int8")

def post(i, o):  # recorta silencios, 48 kHz, -16 LUFS
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", i, "-af",
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                    "highpass=f=70,loudnorm=I=-16:TP=-2:LRA=7", "-ar", "48000", "-ac", "2", o], check=True)
def dur(p): return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
def norm(s): return re.sub(r"[^a-záéíóúñü0-9 ]", "", s.lower())

lineas, t, palabras, lista = [], 0.35, [], []
for i, ln in enumerate(g["lineas"]):
    w = f"{OUT}/l{i:02d}.wav"; best = None
    for intento in range(3):
        wav = model.generate(ln.get("voz", ln["texto"]), language_id="es", audio_prompt_path=REF,
                             exaggeration=ln.get("exa", 0.6), cfg_weight=ln.get("cfg", 0.45))
        tmp = tempfile.mktemp(suffix=".wav"); ta.save(tmp, wav, model.sr); post(tmp, w)
        segs, _ = wm.transcribe(w, language="es", word_timestamps=True)
        ws = [x for s in segs for x in s.words]
        ref = set(norm(ln["texto"]).split()); hyp = set(norm(" ".join(x.word for x in ws)).split())
        score = len(ref & hyp) / max(1, len(ref))
        if best is None or score > best[0]: best = (score, ws); subprocess.run(["cp", w, w + ".best"])
        if score > 0.85: break
    subprocess.run(["mv", w + ".best", w])
    d = dur(w); lineas.append({"i": i, "a": round(t, 2), "b": round(t + d, 2), "acierto": round(best[0], 2), **ln})
    for x in best[1]: palabras.append({"w": x.word.strip(), "a": round(t + x.start, 2), "b": round(t + x.end, 2), "linea": i})
    lista.append(w); t += d + ln.get("pausa", 0.28)
json.dump(lineas, open(f"{OUT}/lineas.json", "w"), ensure_ascii=False, indent=1)
json.dump(palabras, open(f"{OUT}/timing.json", "w"), ensure_ascii=False)
print(json.dumps([{k: l[k] for k in ("i", "a", "b", "acierto")} for l in lineas]))
