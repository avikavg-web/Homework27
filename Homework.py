import os, io, time, requests, mimetypes
from datetime import datetime
from PIL import Image
from config import HF_API_KEY

MODEL = "runwayml/stable-diffusion-inpainting"
API = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
MAX_MB = 8

def ask_image(prompt_text="Image path: "):
    print(f"\nPick an image (JPG/ PNG/ WebP/ BMP/ TIFF ≤ 8MB) from this folder.")
    while True:
        p = input(prompt_text).strip().strip('"').strip("'")
        if not p or not os.path.isfile(p): print("Not found."); continue
        if os.path.splitext(p)[1].lower() not in ALLOWED: print("Unsupported type"); continue
        if os.path.getsize(p)/(1024*1024) > MAX_MB: print("Too big (>8MB)"); continue
        try: Image.open(p).verify()
        except: print("Corrupted image"); continue
        return p

def infer(img_bytes, mask_bytes, prompt, tries=8):
    files = {
        "image": ("image.png", img_bytes, "image/png"),
        "mask_image": ("mask.png", mask_bytes, "image/png")
    }
    data = {"parameters": {"prompt": prompt}}
    for _ in range(tries):
        r = requests.post(API, headers={"Authorization": f"Bearer {HF_API_KEY}"}, files=files, data={"data": str(data)}, timeout=60)
        if r.status_code == 200:
            return r.content
        if r.status_code == 503:
            time.sleep(2)
            continue
        raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Model warm-up timeout")

def main():
    path = ask_image("Original image path: ")
    mask_path = ask_image("Mask image path: ")
    prompt = input("Prompt: ").strip()

    with open(path, "rb") as fh: img_bytes = fh.read()
    with open(mask_path, "rb") as fh: mask_bytes = fh.read()

    try:
        res_bytes = infer(img_bytes, mask_bytes, prompt)
    except Exception as e:
        return print("❌", e)

    orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize(orig_img.size)
    restored_img = Image.open(io.BytesIO(res_bytes)).convert("RGB").resize(orig_img.size)

    final_img = Image.composite(restored_img, orig_img, mask_img)

    out = f"restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    final_img.save(out)
    print(f"✅ Saved: {out}")

if __name__ == "__main__":
    main()