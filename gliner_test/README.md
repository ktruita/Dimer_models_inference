# GLiNER multi-v2.1 zero-shot NER test

`gliner_ner.ipynb` uses `urchade/gliner_multi-v2.1` to extract text spans for entity types chosen at run time:

1. Custom examples (payments, Tagalog, medical, products) with coloured highlighting
2. Threshold effect
3. English news with gold labels (CoNLL-2003, 60 sentences): precision / recall / F1 + threshold sweep
4. Tagalog Wikipedia with gold labels (WikiANN-tl, 60 sentences)
5. Invoice/receipt field extraction from the OCR notebook's SmolDocling output, checked against the answer key
   (run `../ocr_test/ocr_inference.ipynb` first)

Runs on CPU: ~0.15–0.2 s per sentence, <1 s per invoice page. Multilingual.

## Environment

Existing `gliner` env (kernel **Python (gliner)**): `gliner`, torch (CPU), transformers, sentencepiece, protobuf, pandas, matplotlib, jinja2.

## Outputs

`outputs/invoice_field_check.csv`, `outputs/summary.csv`

## Data

- `data/conll2003_en_60.jsonl`: 60 sentences from the CoNLL-2003 test set (`tner/conll2003`)
- `data/wikiann_tl_60.jsonl`: 60 sentences from WikiANN Tagalog test set (`unimelb-nlp/wikiann`)

Each line: `{"tokens": [...], "entities": [[start_token, end_token_exclusive, "PER|ORG|LOC|MISC"], ...]}`
