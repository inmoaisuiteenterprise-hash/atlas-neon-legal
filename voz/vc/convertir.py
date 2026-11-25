# Unifica el timbre de toda una narración con OpenVoice V2 (MIT): cada frase se convierte al timbre de la voz de referencia.
# Uso: python convertir.py <dir_entrada> <ref.wav> <dir_salida>
import sys, os, glob, torch
sys.path.insert(0, "OpenVoice")
from openvoice.api import ToneColorConverter
ent, ref, sal = sys.argv[1:4]
cv = ToneColorConverter("ckpt/converter/config.json", device="cpu"); cv.load_ckpt("ckpt/converter/checkpoint.pth")
tgt = cv.extract_se([ref])
for w in sorted(glob.glob(f"{ent}/*/*.wav")):
    d = os.path.join(sal, os.path.basename(os.path.dirname(w))); os.makedirs(d, exist_ok=True)
    src = cv.extract_se([w])
    cv.convert(audio_src_path=w, src_se=src, tgt_se=tgt, output_path=os.path.join(d, os.path.basename(w)), tau=0.3)
    print("ok", w, flush=True)
