# Methodology

## Overview

This document details the complete methodology for benchmarking LLaMA 3's robustness in financial applications via prompt perturbations. The approach quantifies how consistently the model responds to semantically identical questions phrased differently.

## Theoretical Foundation

### Semantic Entropy

**Definition**: Semantic entropy measures the diversity of meanings in a set of LLM outputs. It's based on Shannon entropy applied to semantic clusters rather than raw text.

**Formula**:
```
H = -Σ p_i log₂(p_i)
```

Where:
- H = semantic entropy
- p_i = proportion of responses in cluster i
- Summation over all unique semantic clusters

**Interpretation**:
- **H ≈ 0**: All responses convey the same meaning (high consistency)
- **H > 2**: Many distinct meanings (model confusion/hallucination)
- **Intermediate H**: Moderate response diversity

### Robustness Metric

**Definition**: Robustness (R) is the inverse of semantic entropy, normalized to [0, 1].

**Formula**:
```
R = 1 / (1 + H)
```

**Properties**:
- R → 1 as H → 0 (perfect robustness)
- R → 0 as H → ∞ (no robustness)
- R = 0.5 when H = 1 (moderate)

**Stability Score**:
```
S = R_mean × (1 - min(σ_R, 1))
```

Where σ_R is the standard deviation of robustness across prompt variants.

## Pipeline Architecture

```
┌──────────────────┐
│ Raw Datasets     │
│ (FinQA, Alpaca)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Preprocessing    │
│ Extract Seeds    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Paraphrasing     │
│ (Back-trans/T5)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLaMA 3 Sampling │
│ (M=20 per prompt)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Embedding        │
│ (SBERT)          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Clustering       │
│ (HDBSCAN)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Entropy & R      │
│ Calculation      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Visualization    │
└──────────────────┘
```

## Detailed Methods

### Phase 1: Dataset Acquisition

**Datasets Used**:

1. **FinQA** (Chen et al., 2021)
   - 8,281 Q&A pairs from earnings reports
   - License: CC-BY 4.0
   - Source: HuggingFace `ibm/finqa`

2. **Alpaca-Finance**
   - 70k finance instruction-output pairs
   - License: MIT
   - Source: HuggingFace `gbharti/finance-alpaca`

3. **BillSum** (Kornilova & Eidelman, 2019)
   - 23k Congressional bill summaries
   - License: CC0 1.0
   - Source: HuggingFace `billsum`

**Selection Criteria**:
- Domain relevance (finance/legal)
- Open licensing
- High quality (human-verified)
- Diversity of query types

### Phase 2: Prompt Generation

**Seed Extraction**:
- Stratified sampling across topics
- N = 50 seed prompts
- Coverage: Q&A, summarization, analysis, advisory

**Paraphrasing Methods**:

1. **Back-Translation**
   - Model: MarianMT (Helsinki-NLP/opus-mt-en-fr)
   - Process: English → French → English
   - Preserves meaning, varies syntax

2. **T5 Paraphrasing**
   - Model: T5-base
   - Task prefix: "paraphrase: {text}"
   - Temperature: 1.2, Top-p: 0.95

**Quality Control**:
- Semantic similarity threshold: cosine > 0.85
- Embedding model: all-MiniLM-L6-v2
- Manual spot-checking (10% sample)

**Variant Count**: 10 per seed (plus original)

### Phase 3: LLM Sampling

**Model Configuration**:
- **Model**: LLaMA 3 8B Instruct
- **Quantization**: 4-bit (BitsAndBytes)
- **Framework**: HuggingFace Transformers / Ollama

**Sampling Parameters**:
- **Samples per prompt**: M = 20
- **Temperature**: 0.7
- **Top-p**: 0.9
- **Top-k**: 50
- **Max tokens**: 512

**Rationale for M=20**:
- Balances statistical reliability with compute cost
- Literature precedent (Farquhar et al., 2024)
- Sufficient for entropy estimation

**Prompt Format** (LLaMA 3 Instruct):
```xml
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a financial AI assistant...
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Phase 4: Semantic Entropy Measurement

**Embedding**:
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimension**: 384
- **Normalization**: L2 normalized

**Clustering**:
- **Algorithm**: HDBSCAN (Hierarchical Density-Based)
- **Min cluster size**: 2
- **Metric**: Euclidean distance
- **Selection method**: Excess of Mass (EOM)

**Why HDBSCAN?**:
- No need to predefine cluster count
- Robust to noise
- Finds natural groupings
- Handles varying densities

**Alternative**: KMeans with auto-k (elbow/silhouette)

**Entropy Computation**:
1. Cluster M outputs into k groups
2. Compute proportions: p_i = count_i / M
3. Calculate: H = -Σ p_i log₂(p_i)
4. Aggregate across variants: H̄ = mean(H_variant)

### Phase 5: Robustness Metric

**Formula Application**:
```python
R = 1 / (1 + H̄)
```

**Categorization**:
| Robustness | Category |
|------------|----------|
| R ≥ 0.8    | Very Robust |
| 0.6 ≤ R < 0.8 | Robust |
| 0.4 ≤ R < 0.6 | Moderately Robust |
| 0.2 ≤ R < 0.4 | Weak |
| R < 0.2    | Very Weak |

**Stability Analysis**:
- Variance across variants
- Consistency checks
- Outlier detection

### Phase 6: Visualization

**Generated Plots**:

1. **Entropy Heatmap**
   - Rows: Prompt families
   - Columns: Variants
   - Color: Entropy (red = high)

2. **Robustness Distribution**
   - Histogram + KDE
   - Box plots by category

3. **Entropy vs Robustness Scatter**
   - Points: Prompt families
   - Color: Stability score
   - Overlay: Theoretical curve

4. **Model Comparison** (if applicable)
   - Bar charts: Mean R and H
   - Error bars: Standard deviations

5. **Top/Bottom Performers**
   - Most robust prompts
   - Least robust prompts

6. **Interactive Dashboard** (Plotly)
   - Hover details
   - Zoom/pan
   - Export capabilities

## Statistical Considerations

### Sample Size

**For M=20 samples**:
- Margin of error in entropy: ±0.2 (95% CI)
- Acceptable for comparative analysis
- Increase to M=50 for publication

### Significance Testing

**Comparing models**:
- Paired t-test on entropy values
- Effect size: Cohen's d
- Bonferroni correction for multiple comparisons

**Correlation analysis**:
- Pearson correlation: entropy vs prompt length
- Spearman rank: robustness vs topic complexity

### Reproducibility

**Fixed seeds**:
- Random seed: 42
- PyTorch seed: 42
- NumPy seed: 42

**Version control**:
- Model checkpoint specified
- Library versions locked (requirements.txt)
- Dataset versions documented

## Limitations

1. **Single model size**: 8B only (not 70B)
2. **Language**: English only
3. **Modality**: Text only (no charts/tables)
4. **Evaluation**: Automated (no human judgment)
5. **Coverage**: Limited topic sampling

## Extensions

**Potential enhancements**:
- Multi-model comparison (GPT-4, Claude)
- Fine-tuning experiments
- Human evaluation of high-entropy outputs
- Real-time robustness API
- Cross-lingual analysis

## References

1. **Farquhar et al. (2024)**: "Detecting hallucinations in large language models using semantic entropy." *Nature*.
2. **Chen et al. (2021)**: "FinQA: A Dataset of Numerical Reasoning over Financial Data." *EMNLP*.
3. **Kornilova & Eidelman (2019)**: "BillSum: A Corpus for Automatic Summarization of US Legislation." *NLP4IF*.
4. **Sentence-BERT** (Reimers & Gurevych, 2019): "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP*.
5. **HDBSCAN** (McInnes et al., 2017): "hdbscan: Hierarchical density based clustering." *JOSS*.

---

*For implementation details, see source code in `/src/` directory.*
