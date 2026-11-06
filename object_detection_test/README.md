# Object detection inference test

`detection_inference.ipynb` runs five detectors on the 10 images in `images/` and saves every output:

| Model | Source | Runs on CPU |
|---|---|---|
| YOLOX-S | official ONNX export (Megvii release 0.1.1rc0) | yes |
| YOLOX-X | official ONNX export (Megvii release 0.1.1rc0) | yes (slower) |
| RT-DETR R50-VD | `PekingU/rtdetr_r50vd` | yes |
| OWLv2 base/16 ensemble | `google/owlv2-base-patch16-ensemble` | yes (slowest) |
| Grounding DINO tiny | `IDEA-Research/grounding-dino-tiny` | yes |

No GPU is required. If one is available it is used automatically.

## Environment (already created on this machine as `detect`)

```bash
conda create -y -n detect python=3.12
conda activate detect
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install transformers onnxruntime pillow matplotlib pandas opencv-python-headless jupyter ipykernel
python -m ipykernel install --user --name detect --display-name "Python (detect)"
```

For a GPU machine, install the normal CUDA `torch`/`torchvision` wheels and `onnxruntime-gpu` instead.

## Run

```bash
cd object_detection_test
jupyter lab detection_inference.ipynb          # pick the "Python (detect)" kernel, Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace detection_inference.ipynb
```

Weights download on first run: YOLOX ONNX files into `weights/` (~430 MB total), the
Hugging Face models into `~/.cache/huggingface` (~1.3 GB total).

## Outputs

- `outputs/<model>/<image>.jpg` - annotated images
- `outputs/detections.csv` - every box: model, image, label, score, x1, y1, x2, y2
- `outputs/summary.csv` - load time, average seconds per image, detection count per model

## Using your own images

Drop `.jpg` files into `images/`. Edit `PHOTO_PROMPTS` / `DOC_PROMPTS` (open-vocabulary models only)
and `DOC_IMAGES` (which images get the document prompts) in the Settings cell.

## Image sources

- `01`-`08`: COCO val2017 (Flickr images under various Creative Commons licenses)
- `09_invoice`: `katanaml-org/invoices-donut-data-v1` (synthetic invoice)
- `10_receipt`: `naver-clova-ix/cord-v2` (receipt photo)
