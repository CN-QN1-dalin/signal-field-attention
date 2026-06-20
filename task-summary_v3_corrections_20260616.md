# Task Summary: v3 Strict Review + Paper Corrections

## Objective
Execute strict peer review from Area Chair perspective and apply all corrections to the three main papers (Engine, Heritage, Native, LingYa).

## Key Reasoning & Findings

### Memory Calculation Correction
- **462KB for 7B is CORRECT**: Ring KV Buffer (448KB) + Field State (14KB)
- **248x compression is WRONG**: This was from 0.5B model at 4096 sequence, mislabeled as 7B/64K
- **Correct 7B compression ratios** (GQA kv_heads=4):
  - float16: 284x
  - float32: 567x

### Critical Issues Fixed
1. **Engine v3**: Corrected 248x → 284x/567x, added "Independent Researcher" affiliation, marked 4.16x as deployment target
2. **Heritage v3**: Marked ALL PPL/ablation/downstream data as simulated, corrected Lemma 1 error, removed false institution name
3. **Native v3**: Marked delay estimates as theoretical, Homeostasis/GrowthTemporal as unvalidated, 7B PPL as TBD
4. **LingYa v3**: Marked channel ablation/delay/PPL as simulated, added QLoRA/DoRA/AdaLoRA references

### Data Classification System
All papers now classify data into three categories:
- ✅ True experiments (MLX prototype)
- 🔬 Theoretical calculations (verifiable formulas)
- ⚠️ Simulated data (formula-generated, not real training)

## Files Created/Modified

### New Papers (v3.0)
- `01_soma_engine/学术论文_v3.md` - Engine paper with corrected memory/compression
- `05_soma_heritage/学术论文_v3.md` - Heritage paper with simulated data labeled
- `03_soma_native/学术论文_v3.md` - Native paper with honest limitations
- `02_soma_lingya/学术论文_v3.md` - LingYa paper with simulated data labeled

### Supporting Files
- `测试数据汇总_v3.md` - Data classification guide
- `顶会审查报告_v3_strict.md` - Already existed, not modified (too long to edit)

### Key Statistics
| Paper | Lines | Size | Status |
|-------|-------|------|--------|
| Engine v3 | ~280 | ~8.7KB | ✅ Ready for arXiv |
| Heritage v3 | ~240 | ~6.2KB | ⚠️ Needs real PPL data |
| Native v3 | ~230 | ~5.8KB | ⚠️ Needs Homeostasis/Native validation |
| LingYa v3 | ~220 | ~5.4KB | ⚠️ Needs real channel ablation |

## Next Steps
1. Run real distillation training for Heritage PPL data
2. Train Soma Native architecture and validate Homeostasis/GrowthTemporal
3. Run real channel ablation for LingYa
4. Add QLoRA/DoRA/AdaLoRA comparisons
5. Target arXiv submission with clear "preliminary results" disclaimer
