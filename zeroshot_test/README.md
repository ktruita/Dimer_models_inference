# BART-large-MNLI zero-shot classification test

`zeroshot_classification.ipynb` uses `facebook/bart-large-mnli` to classify text into labels chosen at run time, with no training:

1. Raw NLI (premise/hypothesis → entailment / neutral / contradiction)
2. AG News topics (40 articles): plain vs descriptive labels vs a task template
3. IMDB sentiment (20 reviews)
4. Multi-label message routing (custom examples)
5. Document-type classification of SmolDocling's output from `../ocr_test` (run that first); fields can then be extracted with GLiNER (`../gliner_test`)

Runs on CPU (~1.6 GB RAM, ~0.3–0.6 s per text-label pair). English only.

## Environment

Uses the existing `ocr` env (kernel **Python (ocr)**): transformers, torch (CPU), pandas, scikit-learn, matplotlib, jinja2.

## Outputs

`outputs/ag_news_predictions.csv`, `imdb_predictions.csv`, `document_type_predictions.csv`, `summary.csv`

## Data

- `data/ag_news_40.csv`: 10 per class from the AG News test set (`fancyzhx/ag_news`)
- `data/imdb_20.csv`: 10 positive + 10 negative short reviews from the IMDB test set (`stanfordnlp/imdb`)
