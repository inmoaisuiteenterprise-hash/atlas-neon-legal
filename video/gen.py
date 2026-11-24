# Clips animados a partir de una ilustración con FastWan2.2-TI2V-5B (Apache-2.0, destilado a pocos pasos), en CPU.
# Uso: python gen.py pedido.json indice_trozo n_trozos salida_dir
import sys, json, os, time, gc, torch
from PIL import Image
P = json.load(open(sys.argv[1])); k, n, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4); dt = torch.bfloat16
M = P.get("modelo", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers")
mios = [c for i, c in enumerate(P["clips"]) if i % n == k]
estilo = P.get("estilo", ""); neg = P.get("negativo", "")
# 1) textos: se codifican primero y se libera el codificador (no caben los dos en 16 GB)
from transformers import AutoTokenizer, UMT5EncoderModel
tok = AutoTokenizer.from_pretrained(M, subfolder="tokenizer")
te = UMT5EncoderModel.from_pretrained(M, subfolder="text_encoder", torch_dtype=dt)
def cod(t):
    x = tok([t], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    with torch.no_grad(): h = te(x.input_ids, x.attention_mask).last_hidden_state
    L = int(x.attention_mask.sum()); h[:, L:] = 0; return h.to(dt)
emb = {c["id"]: cod(c["prompt"] + ", " + estilo) for c in mios}; embn = cod(neg)
del te; gc.collect(); print("textos listos", flush=True)
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
vae = AutoencoderKLWan.from_pretrained(M, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(M, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=dt)
try: pipe.vae.enable_tiling()
except Exception as e: print("sin tiling", e)
for c in mios:
    t = time.time(); W, H = c.get("w", 832), c.get("h", 480)
    img = Image.open(c["imagen"]).convert("RGB").resize((W, H), Image.LANCZOS)
    fr = pipe(image=img, prompt_embeds=emb[c["id"]], negative_prompt_embeds=embn, height=H, width=W,
              num_frames=c.get("frames", 49), num_inference_steps=c.get("pasos", P.get("pasos", 4)), guidance_scale=0,
              generator=torch.Generator().manual_seed(c.get("seed", 7))).frames[0]
    export_to_video(fr, f"{out}/{c['id']}.mp4", fps=24); print(c["id"], round(time.time() - t), "s", flush=True)
