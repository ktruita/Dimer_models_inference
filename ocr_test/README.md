# OCR / document model inference test

`ocr_inference.ipynb` runs four document models on the 10 images in `../sample_docs/` (5 invoices, 5 receipts),
scores them against the datasets' answer key and saves every output.

| Model | Source | Runs on CPU |
|---|---|---|
| Tesseract 5.5 | system program (`conda install tesseract`) + `pytesseract` | yes, ~1–3 s per page |
| LayoutLM doc-QA | `impira/layoutlm-document-qa` (uses Tesseract for words and boxes) | yes, ~1 s per question |
| SmolDocling | `docling-project/SmolDocling-256M-preview` | yes, ~30–70 s per page |
| GOT-OCR 2.0 | `stepfun-ai/GOT-OCR-2.0-hf` | yes in bfloat16, ~40–60 s per page (float32 needs ~4 GB+ free RAM) |

No GPU is required. If one is available it is used automatically.

## Environment (already created on this machine as `ocr`)

```bash
conda create -y -n ocr python=3.12
conda activate ocr
conda install -y tesseract                     # from the default channel (conda-forge is blocked here by an SSL error)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU build
pip install transformers accelerate pillow pytesseract docling-core matplotlib pandas jinja2 ipykernel
python -m ipykernel install --user --name ocr --display-name "Python (ocr)"
```

On a GPU machine, install the normal CUDA `torch` instead. On Ubuntu without conda, use `sudo apt install tesseract-ocr`.

## Run

```bash
cd ocr_test
jupyter lab ocr_inference.ipynb        # pick the "Python (ocr)" kernel, Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 ocr_inference.ipynb
```

The full run takes about 20 minutes on CPU, mostly GOT-OCR and SmolDocling. Close other memory-heavy programs first:
GOT-OCR peaks at about 3.7 GB of RAM.

## Outputs

- `outputs/tesseract/*.txt`, `outputs/got_ocr/*.txt` - plain text per document (+ `invoice_1_formatted.md`)
- `outputs/smoldocling/*.doctags` and `*.md` - raw DocTags and the converted Markdown
- `outputs/layoutlm_answers.csv` - every question, answer, score, expected value
- `outputs/field_check.csv` - per answer-key field: did each model get it exactly
- `outputs/summary.csv` - % of answer-key values found and seconds per page, per model

## Using your own documents

Put images in a folder, point `DOCS` and `NAMES` in the Settings cell at them, and remove or replace the answer-key cell
(`KEY`). Without a key, you can still read every output; only the scoring needs it. For LayoutLM, edit `QUESTIONS`.
