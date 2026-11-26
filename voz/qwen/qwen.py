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
if g.get("tomas") and modo == "clon":   # elegir, de varias tomas, la que más se parece a la voz de referencia (huella de voz con Resemblyzer, Apache 2.0)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "resemblyzer"], check=False)
    from resemblyzer import VoiceEncoder, preprocess_wav
    import numpy as np
    ENC = VoiceEncoder("cpu"); REFE = ENC.embed_utterance(preprocess_wav(ref))
def parecido(w):
    if ENC is None: return 0
    e = ENC.embed_utterance(preprocess_wav(w)); return float(np.dot(e, REFE) / (np.linalg.norm(e) * np.linalg.norm(REFE)))
if g.get("toma_unica"):   # todo el bloque en UNA toma continua (misma voz de principio a fin), luego se corta por frases
    import difflib
    L = g["lineas"]; txt = " ".join(ln.get("voz", ln["texto"]) for ln in L); full = f"{OUT}/_toma.wav"; best = None
    for intento in range(0 if g.get("toma_wav") else g.get("tomas", 3)):
        torch.manual_seed(2000 + intento)
        if modo == "diseno": wavs, sr = model.generate_voice_design(text=txt, language="Spanish", instruct=g["instruct"])
        else: wavs, sr = model.generate_voice_clone(text=txt, language="Spanish", voice_clone_prompt=prompt)
        tmp = tempfile.mktemp(suffix=".wav"); sf.write(tmp, wavs[0], sr); post(tmp, full)
        segs, _ = wm.transcribe(full, language="es", word_timestamps=True); ws = [x for s in segs for x in s.words]
        R = [w for ln in L for w in norm(ln["texto"]).split()]; Hn = [norm(x.word).strip() for x in ws]
        sm = difflib.SequenceMatcher(None, R, Hn, autojunk=False); score = sum(b.size for b in sm.get_matching_blocks()) / max(1, len(R))
        sim = parecido(full); total = score + (2.0 * sim if ENC is not None else 0); print("toma", intento, round(score, 2), round(sim, 3), flush=True)
        if best is None or total > best[0]: best = (total, score, sim, ws, sm.get_matching_blocks(), R); subprocess.run(["cp", full, full + ".best"])
    if g.get("toma_wav"):   # reutilizar una toma ya generada (solo se vuelve a alinear y cortar)
        src = os.path.join(os.path.dirname(os.path.abspath(G)), "..", g["toma_wav"]); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "48000", "-ac", "2", full], check=True)
        segs, _ = wm.transcribe(full, language="es", word_timestamps=True); ws = [x for s in segs for x in s.words]
        R = [w for ln in L for w in norm(ln["texto"]).split()]; Hn = [norm(x.word).strip() for x in ws]
        sm = difflib.SequenceMatcher(None, R, Hn, autojunk=False); score = sum(b.size for b in sm.get_matching_blocks()) / max(1, len(R)); sim = parecido(full) if ENC is not None else 0
        bloques = sm.get_matching_blocks()
    else:
        subprocess.run(["mv", full + ".best", full]); _, score, sim, ws, bloques, R = best
    json.dump([{"w": x.word.strip(), "a": round(x.start, 2), "b": round(x.end, 2)} for x in ws], open(f"{OUT}/_toma.json", "w"), ensure_ascii=False)
    m = {}
    for b in bloques:
        for k in range(b.size): m[b.a + k] = b.b + k
    lim, c = [], 0
    for ln in L: c += len(norm(ln["texto"]).split()); lim.append(c)
    cortes, D = [0.0], dur(full)
    for k in range(len(L) - 1):
        r = lim[k]; antes = max([m[j] for j in m if j < r], default=0); desp = min([m[j] for j in m if j >= r], default=len(ws) - 1)
        cortes.append((ws[antes].end + ws[desp].start) / 2 if desp > antes else ws[antes].end + 0.05)
    cortes.append(D)
    for i, ln in enumerate(L):
        a, b = cortes[i], cortes[i + 1]; w = f"{OUT}/l{i:02d}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", full, "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-af",
                        "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,afade=t=in:d=0.015", w], check=True)
        d = dur(w); seg, _ = wm.transcribe(w, language="es", word_timestamps=True); wl = [x for s in seg for x in s.words]
        lineas.append({"i": i, "a": round(t, 2), "b": round(t + d, 2), "acierto": round(score, 2), "parecido": round(sim, 3), **ln})
        for x in wl: palabras.append({"w": x.word.strip(), "a": round(t + x.start, 2), "b": round(t + x.end, 2), "linea": i})
        t += d + ln.get("pausa", 0.3); print(i, round(d, 1), flush=True)
    json.dump(lineas, open(f"{OUT}/lineas.json", "w"), ensure_ascii=False, indent=1); json.dump(palabras, open(f"{OUT}/timing.json", "w"), ensure_ascii=False); sys.exit(0)
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
