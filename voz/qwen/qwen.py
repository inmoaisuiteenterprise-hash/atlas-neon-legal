# Narración con Qwen3-TTS (Alibaba, licencia Apache 2.0), en CPU. Sin claves ni servicios de pago.
# Modo "diseno": la voz se describe con texto (instruct). Modo "clon": se clona una voz de referencia ya diseñada.
# Uso: python qwen.py <guion.json> <salida_dir>
#   guion: {"modo": "diseno"|"clon", "instruct": "...", "ref": "ruta.wav", "ref_text": "...", "lineas": [{"texto", "voz"?, "pausa"?}]}
import sys, os, json, re, subprocess, tempfile
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
G, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
g = json.load(open(G, encoding="utf-8")); modo = g.get("modo", "diseno")
torch.set_num_threads(max(1, os.cpu_count() or 1))
nombre = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" if modo == "diseno" else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
model = Qwen3TTSModel.from_pretrained(nombre, device_map="cpu", dtype=torch.float32)
from faster_whisper import WhisperModel
wm = WhisperModel("small", device="cpu", compute_type="int8")
def norm(s): return re.sub(r"[^a-záéíóúñü0-9 ]", "", s.lower())
def post(i, o):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", i, "-af",
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                    "highpass=f=70,loudnorm=I=-16:TP=-2:LRA=9", "-ar", "48000", "-ac", "2", o], check=True)
def dur(p): return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
prompt = None
if modo == "clon":
    ref = os.path.join(os.path.dirname(os.path.abspath(G)), "..", "refs", g["ref"]) if not os.path.isabs(g["ref"]) else g["ref"]
    prompt = model.create_voice_clone_prompt(ref_audio=ref, ref_text=g["ref_text"], x_vector_only_mode=False)
lineas, palabras, t = [], [], 0.35
ENC = REFE = None
if g.get("tomas"):   # elegir, de varias tomas, la que más se parece a la voz de referencia (huella de voz con Resemblyzer, Apache 2.0)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "resemblyzer"], check=False)
    from resemblyzer import VoiceEncoder, preprocess_wav
    import numpy as np
    ENC = VoiceEncoder("cpu"); REFE = ENC.embed_utterance(preprocess_wav(ref))
def parecido(w):
    if ENC is None: return 0
    e = ENC.embed_utterance(preprocess_wav(w)); return float(np.dot(e, REFE) / (np.linalg.norm(e) * np.linalg.norm(REFE)))
for i, ln in enumerate(g["lineas"]):
    w = f"{OUT}/l{i:02d}.wav"; best = None
    for intento in range(g.get("tomas", 3)):
        torch.manual_seed(1000 + i * 17 + intento)
        txt = ln.get("voz", ln["texto"])
        if modo == "diseno": wavs, sr = model.generate_voice_design(text=txt, language="Spanish", instruct=ln.get("instruct", g["instruct"]))
        else: wavs, sr = model.generate_voice_clone(text=txt, language="Spanish", voice_clone_prompt=prompt)
        tmp = tempfile.mktemp(suffix=".wav"); sf.write(tmp, wavs[0], sr); post(tmp, w)
        segs, _ = wm.transcribe(w, language="es", word_timestamps=True); ws = [x for s in segs for x in s.words]
        ref_w = set(norm(ln["texto"]).split()); hyp = set(norm(" ".join(x.word for x in ws)).split())
        score = len(ref_w & hyp) / max(1, len(ref_w)); d = dur(w); vel = len(ln["texto"].split()) / max(d, 0.1)
        if vel < 1.6 or vel > 4.2: score -= 0.3          # atascos (muy lenta) o atropellos: se repite
        sim = parecido(w); total = score + (2.0 * sim if ENC is not None else 0); print(i, intento, round(score, 2), round(sim, 3), flush=True)
        if best is None or total > best[2]: best = (score, ws, total, sim); subprocess.run(["cp", w, w + ".best"])
        if ENC is None and score > 0.88: break
    subprocess.run(["mv", w + ".best", w]); d = dur(w)
    lineas.append({"i": i, "a": round(t, 2), "b": round(t + d, 2), "acierto": round(best[0], 2), "parecido": round(best[3], 3) if len(best) > 3 else None, **ln})
    for x in best[1]: palabras.append({"w": x.word.strip(), "a": round(t + x.start, 2), "b": round(t + x.end, 2), "linea": i})
    t += d + ln.get("pausa", 0.3); print(i, round(best[0], 2), round(d, 1), flush=True)
json.dump(lineas, open(f"{OUT}/lineas.json", "w"), ensure_ascii=False, indent=1)
json.dump(palabras, open(f"{OUT}/timing.json", "w"), ensure_ascii=False)
