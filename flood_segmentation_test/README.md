# Prithvi-EO-2.0 flood segmentation test

`prithvi_sen1floods11_flood_segmentation.ipynb` runs `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`
(IBM/NASA Prithvi ViT-L encoder + UperNet decoder, 319M parameters) on three Sentinel-2 example scenes and maps
each pixel to **water / no water**. It scores the result against the Sen1Floods11 hand labels and saves GeoTIFFs.

- Input: 6 Sentinel-2 bands (B2, B3, B4, B8A, B11, B12), reflectance = DN / 10000, any image size (tiled at 512×512)
- Output: class map + water probability per pixel (`outputs/pred_*.tif`, `outputs/prob_*.tif`)
- CPU: ~10 s per 512×512 tile; GPU: well under 1 s

## Environment (created on this machine as `prithvi`)

```bash
conda create -y -n prithvi python=3.11
conda activate prithvi
pip install -r requirements.txt
python -m ipykernel install --user --name prithvi --display-name "Python (prithvi)"
```

## Data

The model checkpoint, config and example scenes (~1.3 GB) download into `hf_model/` on the first run.
The hand labels come from the public Sen1Floods11 bucket and are only used for scoring.
