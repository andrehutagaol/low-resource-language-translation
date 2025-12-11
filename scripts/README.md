# Pivot-Based Translation Without Fine-Tuning

This notebook implements pivot-based machine translation between Nepali and Sinhala using English as an intermediate language, without any model fine-tuning.

## Overview

Pivot-based translation is a technique for translating between low-resource language pairs by using a high-resource language (English) as an intermediary. Instead of translating directly from Nepali to Sinhala, the approach:

1. Translates Nepali → English
2. Translates English → Sinhala

## Requirements

```bash
pip install transformers torch pandas evaluate sacrebleu
pip install git+https://github.com/google-research/bleurt.git
```

## Data

The notebook uses the **FLORES-200** dataset, which provides parallel sentences across 200 languages:
- `npi_Deva.dev` - Nepali (Devanagari script)
- `eng_Latn.dev` - English (Latin script)
- `sin_Sinh.dev` - Sinhala (Sinhala script)

## Model

Uses **NLLB-200-distilled-600M** (`facebook/nllb-200-distilled-600M`), a multilingual translation model supporting 200 languages. The model is used as-is without any fine-tuning.

## Pipeline

### Translation Functions

The notebook defines four translation functions using forced BOS tokens to control the target language:

| Function | Direction |
|----------|-----------|
| `nep_to_eng()` | Nepali → English |
| `eng_to_sin()` | English → Sinhala |
| `sin_to_eng()` | Sinhala → English |
| `eng_to_nep()` | English → Nepali |

### Translation Paths

Two pivot translation paths are evaluated:

1. **Nepali → Sinhala**: Nepali → English → Sinhala
2. **Sinhala → Nepali**: Sinhala → English → Nepali

## Evaluation Metrics

Three metrics are used to evaluate translation quality:

- **SacreBLEU**: Measures n-gram overlap between hypothesis and reference
- **ChrF**: Character-level F-score, useful for morphologically rich languages
- **BLEURT**: Learned metric that captures semantic similarity

## Results

### Nepali → Sinhala (via English pivot)

| Dataset | SacreBLEU | ChrF | BLEURT |
|---------|-----------|------|--------|
| dev | 8.62 | 39.56 | 0.3894 |
| devtest | 8.87 | 40.03 | 0.4092 |

### Sinhala → Nepali (via English pivot)

| Dataset | SacreBLEU | ChrF | BLEURT |
|---------|-----------|------|--------|
| dev | 9.01 | 43.94 | 0.0877 |
| devtest | 9.70 | 43.20 | 0.0860 |

## Usage

1. Update the file paths in the `files` dictionary to point to your FLORES-200 data
2. Run all cells sequentially
3. Results will be displayed showing translation quality metrics

## Limitations

- **Error propagation**: Errors in the first translation step (e.g., Nepali → English) propagate to the final output
- **Increased latency**: Requires two model inference passes instead of one
- **Information loss**: Nuances may be lost when translating through an intermediate language

## References

- [NLLB-200 Model](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [FLORES-200 Dataset](https://github.com/facebookresearch/flores)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [BLEURT](https://github.com/google-research/bleurt)
