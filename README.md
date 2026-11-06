# Model tests (CPU inference notebooks)

Each folder holds one Jupyter notebook that runs a model (or a group of models) on bundled sample data.
Every notebook is saved with its outputs, so you can read the results without running anything.
Each folder also has its own `README.md` (what the model does, results, CPU notes) and a `requirements.txt`.

| Folder | Models | Task | Env / kernel |
|---|---|---|---|
| `object_detection_test/` | YOLOX-S/X, RT-DETR, OWLv2, Grounding DINO | object detection (fixed classes + text prompts) | `detect` |
| `forecast_test/` | Toto 2, TiRex-2, Chronos-2 | time-series forecasting | `forecast` |
| `ocr_test/` | Tesseract, LayoutLM doc-QA, SmolDocling, GOT-OCR 2.0 | OCR / document understanding | `ocr` |
| `tabular_test/` | TabICLv2, TabDPT v1.2, Mitra (+ gradient boosting baseline) | tabular classification / regression | `tabular` |
| `asr_test/` | Whisper large-v3-turbo | speech recognition (English + Tagalog, mic or files) | `asr` |
| `zeroshot_test/` | BART-large-MNLI | zero-shot text classification | `ocr` |
| `gliner_test/` | GLiNER multi-v2.1 | zero-shot named-entity recognition | `gliner` |
| `rag_test/` | Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | retrieve-then-rerank search | `ocr` |
| `flood_segmentation_test/` | Prithvi-EO-2.0-300M-TL Sen1Floods11 | satellite flood segmentation | `prithvi` |

`sample_docs/` holds the invoice and receipt images used by `ocr_test` and `gliner_test`.
`zeroshot_test` also reads `ocr_test/outputs/`, so keep the folders side by side.

## Setup

One environment per row in the table (`zeroshot_test` and `rag_test` reuse the `ocr` env). Example:

```bash
conda create -y -n asr python=3.12
conda activate asr
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build first (see the top of requirements.txt)
pip install -r asr_test/requirements.txt
python -m ipykernel install --user --name asr --display-name "Python (asr)"
```

The top of each `requirements.txt` lists the Python version, any system tools (Tesseract, ffmpeg) and special install
steps. Versions are pinned to what the notebooks were run with.

Then open the notebook, pick the matching kernel, and Run All.

## Notes

- Tested on a CPU-only laptop (WSL2, 8 cores, 7.8 GB RAM). On a GPU machine, install the normal CUDA `torch`.
  The notebooks use the GPU automatically.
- Model weights are not in the repo. They download on the first run, into `~/.cache/huggingface`, or into
  `object_detection_test/weights/` and `flood_segmentation_test/hf_model/` (both git-ignored).
- The saved outputs in `ocr_test/ocr_inference.ipynb` are partial: the last full run ran out of memory on the laptop.
