# Ilustraciones con SDXL base + SDXL-Lightning (4 pasos), en CPU. Licencias: CreativeML OpenRAIL++-M (uso comercial permitido).
# Uso: python gen.py pedido.json indice_trozo n_trozos salida_dir
import sys, json, os, time, torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
P = json.load(open(sys.argv[1])); k, n, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4); dt = torch.bfloat16
base = "stabilityai/stable-diffusion-xl-base-1.0"
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(dt)
unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors"), device="cpu"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=dt, variant="fp16")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
estilo = P.get("estilo", ""); neg = P.get("negativo", "")
for i, im in enumerate(P["imagenes"]):
    if i % n != k: continue
    t = time.time(); g = torch.Generator().manual_seed(im.get("seed", 1000 + i))
    img = pipe(prompt=im["prompt"] + ", " + estilo, negative_prompt=neg, num_inference_steps=4, guidance_scale=0, width=im.get("w", 1344), height=im.get("h", 768), generator=g).images[0]
    img.save(f"{out}/{im['id']}.jpg", quality=92); print(im["id"], round(time.time() - t, 1), "s", flush=True)
    if im.get("recorte"):   # personaje recortado (fondo transparente) con rembg (MIT)
        from rembg import remove, new_session
        global SES
        try: SES
        except NameError: SES = new_session("isnet-general-use")
        remove(img, session=SES).save(f"{out}/{im['id']}.png")
    if im.get("profundidad"):   # mapa de profundidad para el paralaje (Depth Anything V2 Small, Apache 2.0)
        from transformers import pipeline as tp
        global DEP
        try: DEP
        except NameError: DEP = tp("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu")
        DEP(img)["depth"].save(f"{out}/{im['id']}_prof.png")
