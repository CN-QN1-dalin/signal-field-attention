# Soma: A Signal Field Native Architecture for Efficient Long-Sequence Processing

## arXiv Submission Package

**Author**: Dalin Jia (Independent Researcher)  
**Contact**: 362118251@qq.com  
**Date**: June 2026  
**Category**: cs.LG (Machine Learning) or cs.CL (Computation and Language)

---

## Files

| File | Description |
|------|-------------|
| `main.tex` | Main paper (LaTeX source) |
| `references.bib` | Bibliography entries |
| `README.md` | This file |
| `soma_engine.py` | SFA engine implementation |
| `lingya_source.py` | LingYa PEFT implementation |
| `heritage_source.py` | Heritage distillation implementation |
| `native_source.py` | Soma Native architecture |
| `benchmark_suite.py` | Benchmark verification script |
| `扩展实验.py` | Extended experiments (simulated) |

---

## arXiv Submission Checklist

### Required Fields
- [x] Title: "Soma: A Signal Field Native Architecture for Efficient Long-Sequence Processing"
- [x] Abstract: Complete, mentions both verified and simulated results
- [x] Author: Dalin Jia, Independent Researcher
- [x] Categories: cs.LG, cs.CL
- [x] License: MIT (code), CC-BY-4.0 (paper)

### Data Transparency
- [x] All simulated data clearly labeled with "Simulated data" tag
- [x] All theoretical calculations labeled as such
- [x] MLX prototype limitations explicitly stated
- [x] C++/Metal deployment targets separated from verified results
- [x] Disclaimer about preliminary nature included
- [x] Correction of 248x → 284x/567x compression ratio

### Known Limitations (in paper)
- [x] Prototype only (no C++/Metal benchmark)
- [x] Simulated distillation PPL data
- [x] Unvalidated Homeostasis/GrowthTemporal
- [x] Single model size (0.5B primary)
- [x] Missing SOTA comparisons (QLoRA, DoRA, AdaLoRA, Mamba-2)
- [x] Single seed (seed=42)
- [x] Theoretical proof issues noted

---

## Compilation

Requires standard LaTeX distribution (TeX Live, MiKTeX, or MacTeX):

```bash
pdflatex main.tex
bibtex main.aux  # if using BibTeX
pdflatex main.tex
pdflatex main.tex
```

Or use overleaf.com for online compilation.

---

## Disclaimer

This work is preliminary research by an independent researcher. Some results are simulated or theoretical. Claims should be interpreted as research hypotheses pending further validation. The code repository is available at: https://github.com/CN-QN1-dalin/dalin-soma-
