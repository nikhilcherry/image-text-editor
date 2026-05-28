# Windows Install Guide

Step-by-step setup for **Windows 10/11**. The `.sh` scripts in this repo are
Linux-only — on Windows you run the Python tools directly.

There are three pieces:
1. **The app + CLI tools** (this repo's Python code)
2. **IOPaint** — the inpainting engine (separate, pulls in PyTorch)
3. **Tesseract** — OCR for the `replace_word.py` "find & replace a word" tool

---

## 0. Prerequisites

- **Python 3.10–3.12** — https://www.python.org/downloads/
  - ✅ On the first installer screen, tick **“Add python.exe to PATH”.**
- **Git** (optional, to clone) — https://git-scm.com/download/win

Check in a new PowerShell window:
```powershell
python --version
```

---

## 1. Get the code

```powershell
git clone https://github.com/nikhilcherry/image-text-editor.git
cd image-text-editor
```
(or download the ZIP from GitHub and extract it, then `cd` into the folder.)

---

## 2. Create a virtual environment + install app deps

```powershell
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
You should now see `(venv)` at the start of your prompt.

---

## 3. Install IOPaint (the inpainting engine)

IOPaint needs **PyTorch**. Pick CPU or GPU.

### Option A — NVIDIA GPU (fast, recommended if you have one)

Install the CUDA build of PyTorch **first**, then IOPaint:

```powershell
# CUDA 12.1 build (works on most recent NVIDIA GPUs):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install iopaint
```
> RTX 50-series (Blackwell) need a newer CUDA build — use `cu124` (or the
> nightly) instead of `cu121`. Check your driver with `nvidia-smi`.

Verify the GPU is seen:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
It should print `CUDA: True`.

### Option B — CPU only (no NVIDIA GPU)

```powershell
pip install iopaint
```
Works everywhere, just slower (inpainting takes longer per image).

### Start IOPaint

Leave this running in **its own terminal**:
```powershell
# GPU:
iopaint start --model=mat --port=8080 --device=cuda
# CPU:
iopaint start --model=mat --port=8080 --device=cpu
```
First run downloads the MAT model (~700 MB). Keep this window open while you use
the tools.

---

## 4. Install Tesseract (for `replace_word.py`)

1. Download the installer from **https://github.com/UB-Mannheim/tesseract/wiki**
2. Install it (default path is `C:\Program Files\Tesseract-OCR`).
3. **Add it to PATH:** Start → “Edit the system environment variables” →
   *Environment Variables* → under *User variables* select **Path** → *Edit* →
   *New* → paste `C:\Program Files\Tesseract-OCR` → OK.
4. Open a **new** terminal and check:
   ```powershell
   tesseract --version
   ```

(If you skip Tesseract, `replace_word.py` still works on stylised text via the
Gemini fallback — see next step.)

---

## 5. (Optional) Gemini API key — fallback OCR for stylised/artistic text

```powershell
copy .env.example .env
notepad .env      # set GEMINI_API_KEY=your_key
```
Free key: https://aistudio.google.com/apikey

---

## 6. Run it

Make sure **IOPaint is running** (step 3) and your venv is active
(`venv\Scripts\activate`).

**Find & replace one word (auto-detected):**
```powershell
python replace_word.py C:\Users\you\Pictures\poster.png -f cat -r dog
```

**Batch a whole folder:**
```powershell
python advanced_replace.py "NEW TEXT" --folder C:\Users\you\Pictures --dry-run
python advanced_replace.py "NEW TEXT" --folder C:\Users\you\Pictures
```

**Web UI:**
```powershell
python app\app.py
```
then open http://localhost:5000 in a browser.

Pick the inpaint model with `--model`:
- `mat` (default) — documents / clean erase
- `sd-v1-5-inpainting.ckpt` — painted/artistic backgrounds (needs the SD setup)

---

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| `python` not found | Reinstall Python with **“Add to PATH”** ticked, open a new terminal |
| `'iopaint' is not recognized` | Activate the venv first: `venv\Scripts\activate` |
| `IOPaint server is not running` | Start it (step 3) and leave that terminal open |
| `tesseract is not recognized` | Add `C:\Program Files\Tesseract-OCR` to PATH, open a **new** terminal |
| `CUDA: False` but you have an NVIDIA GPU | Update GPU driver; reinstall torch with the right `cuXXX` index URL |
| Out of GPU memory | Use `--device=cpu`, or the lighter `--model=lama` |
| PowerShell blocks `activate` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| `replace_word` command not found | That launcher is Linux-only; on Windows run `python replace_word.py ...` |
