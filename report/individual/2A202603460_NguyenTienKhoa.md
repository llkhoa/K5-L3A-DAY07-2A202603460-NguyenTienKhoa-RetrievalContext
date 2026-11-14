# Day 7 Retrieval & Context Engineering Report – 2A202603460 Khoa

## Overview
This report summarizes the results of the three retrieval strategies required by **GUIDE.md**:

| Strategy | Split | # Queries | Hit@1 | Hit@3 | Context‑Hit | Avg Context Chars | Median Runtime (ms) |
|----------|-------|----------|------|------|------------|------------------|---------------------|
| TF‑IDF   | dev   | 15 | 10 | 13 | 2 | 215.6 | 0.28 |
| TF‑IDF   | test  | 5  | 2  | 5  | 0 | 130.4 | 0.31 |
| BM25     | dev   | 15 | 12 | 13 | 2 | 217.4 | 0.28 |
| BM25     | test  | 5  | 3  | 5  | 0 | 126.8 | 0.29 |
| Dense (Gemini) | dev | – | – | – | – | – | – |
| Dense (Gemini) | test| – | – | – | – | – | – |

*The dense‑retrieval columns are empty because the Gemini embedding calls could not be executed in the sandbox (network access is blocked).*  

## Files Produced
- **Chunks (Day 7 schema)**: `starter/data/converted/chunks.json`
- **Answerable questions**: `starter/data/converted/questions_answerable.json`
- **TF‑IDF results**: `starter/results/dev-tfidf/summary.json`, `starter/results/test-tfidf/summary.json`
- **BM25 results**: `starter/results/dev-bm25/summary.json`, `starter/results/test-bm25/summary.json`
- **Dense retrieval placeholder** (script `starter/dense.py` is present but not executed).

## Next Steps / Recommendations
1. **Run dense retrieval locally**
   - Ensure a valid Gemini API key is exported as `GEMINI_API_KEY`.
   - Run the script with network access enabled (e.g., outside the sandbox) using:
   ```
   python starter\dense.py \
       --chunks starter\data\converted\chunks.json \
       --questions starter\data\converted\questions_answerable.json \
       --split dev \
       --output starter\results\dev-dense
   ```
   - Repeat for the `test` split.
2. **If network is unavailable**, consider using an offline embedding model (e.g., `sentence‑transformers/all‑MPNet‑Base‑V2`) that can be installed from a local wheel or pre‑built artifact.
3. **Aggregate final metrics**
   - Once dense results are available, run `starter/agent_metrics.py` (or a custom aggregator) to produce a combined summary table.
4. **Submit**
   - Verify that the `report/individual/2A202603460_NguyenTienKhoa.md` file follows the template from `REPORT_TEMPLATE.md` and that all required JSON files are committed.

---
*All paths are relative to the repository root.  No secret keys are stored in the repository.*
