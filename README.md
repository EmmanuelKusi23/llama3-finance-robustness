# Benchmarking LLaMA 3 Robustness in Finance via Prompt Perturbations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This project rigorously evaluates **LLaMA 3's robustness** when handling financial queries under lexical variations. By generating paraphrased prompts and measuring semantic consistency through entropy-based metrics, we quantify how stable the model's responses are to prompt perturbations—a critical factor for deploying LLMs in high-stakes financial applications.

**Key Question**: *Does LLaMA 3 provide consistent answers to semantically identical financial questions phrased differently?*

### Why This Matters for Finance

- **Risk Management**: Inconsistent AI responses can lead to contradictory financial advice
- **Regulatory Compliance**: Financial institutions need predictable, auditable AI behavior
- **Trust & Reliability**: Clients require stable recommendations regardless of query phrasing
- **Production Readiness**: Measures real-world robustness beyond standard benchmarks

## Project Highlights

- **Novel Metric**: Semantic entropy-based robustness score for financial LLM evaluation
- **Domain-Specific**: Uses real financial datasets (FinQA, Alpaca-Finance, BillSum)
- **Reproducible**: Complete pipeline from data acquisition to visualization
- **Scalable**: Framework extensible to other LLMs (GPT-4, Claude, Mistral)
- **Production-Oriented**: Identifies failure modes for financial AI deployment

## Methodology

### 1. Dataset Acquisition
- **FinQA**: 8k+ Q&A pairs from earnings reports and financial documents
- **Alpaca-Finance**: 70k finance-specific instruction-output pairs
- **BillSum**: 19k legislative/contract summaries for regulatory text

### 2. Prompt Generation
- Extract 20-50 seed prompts covering key finance topics
- Generate 5-10 paraphrased variants per seed using:
  - Back-translation (MarianMT, M2M100)
  - T5-based paraphrase models
  - Semantic similarity validation (cosine > 0.9)

### 3. LLM Sampling
- Run LLaMA 3 (8B/70B) locally via:
  - Ollama (easy setup)
  - llama.cpp (optimized inference)
  - Hugging Face Transformers
- Generate M=20-50 completions per prompt variant (temperature=0.7)

### 4. Semantic Entropy Measurement
- Embed outputs using Sentence-BERT (all-MiniLM-L6-v2)
- Cluster responses via HDBSCAN/KMeans
- Compute Shannon entropy: H = -Σ p_i log(p_i)
  - Low entropy (H≈0): Consistent responses (one cluster)
  - High entropy: Diverse responses (model confusion)

### 5. Robustness Metric
**R = 1 / (1 + H̄)**

Where H̄ is average entropy across prompt variants. Higher R indicates greater robustness.

### 6. Analysis & Visualization
- Heatmaps: Entropy across prompt families and variants
- Comparative charts: LLaMA 3 vs other LLMs
- Failure mode identification: High-entropy prompt patterns

## Repository Structure

```
llama3-finance-robustness/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned data
│   └── prompts/          # Generated prompt variants
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_prompt_generation.ipynb
│   ├── 03_llm_sampling.ipynb
│   ├── 04_entropy_analysis.ipynb
│   └── 05_visualization.ipynb
├── src/
│   ├── data/
│   │   ├── download_datasets.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── prompt_generator.py
│   │   └── llm_runner.py
│   ├── evaluation/
│   │   ├── semantic_clustering.py
│   │   ├── entropy_calculator.py
│   │   └── robustness_metric.py
│   └── visualization/
│       └── plot_results.py
├── results/
│   ├── figures/          # Generated plots
│   ├── metrics/          # Computed scores
│   └── reports/          # Analysis summaries
├── config/
│   └── config.yaml       # Experiment configurations
├── docs/
│   ├── METHODOLOGY.md
│   └── RESULTS.md
├── requirements.txt
├── environment.yml
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (16GB+ recommended for LLaMA 3 8B)
- 50GB+ disk space

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/llama3-finance-robustness.git
cd llama3-finance-robustness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install LLaMA 3 runner (choose one)
# Option 1: Ollama (easiest)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3

# Option 2: llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Download datasets
python src/data/download_datasets.py
```

## Quick Start

### 1. Download and Prepare Data
```bash
python src/data/download_datasets.py --datasets finqa alpaca-finance
python src/data/preprocess.py --output data/processed/
```

### 2. Generate Prompt Variants
```bash
python src/models/prompt_generator.py \
  --input data/processed/seed_prompts.csv \
  --method backtranslation \
  --variants 10 \
  --output data/prompts/
```

### 3. Run LLaMA 3 Sampling
```bash
python src/models/llm_runner.py \
  --model llama3-8b \
  --prompts data/prompts/variants.json \
  --samples 20 \
  --temperature 0.7 \
  --output results/raw_outputs/
```

### 4. Compute Semantic Entropy
```bash
python src/evaluation/entropy_calculator.py \
  --inputs results/raw_outputs/ \
  --embedder sentence-transformers/all-MiniLM-L6-v2 \
  --clustering hdbscan \
  --output results/metrics/
```

### 5. Generate Visualizations
```bash
python src/visualization/plot_results.py \
  --metrics results/metrics/entropy_scores.csv \
  --output results/figures/
```

## Usage Examples

### Interactive Notebook Workflow
```python
# In notebooks/03_llm_sampling.ipynb
from src.models.llm_runner import LLaMARunner

runner = LLaMARunner(model="llama3-8b", device="cuda")
prompts = load_prompt_variants("data/prompts/variants.json")

results = runner.sample_batch(
    prompts=prompts,
    num_samples=20,
    temperature=0.7
)
```

### Entropy Calculation
```python
from src.evaluation.entropy_calculator import SemanticEntropy

calculator = SemanticEntropy(
    embedder="all-MiniLM-L6-v2",
    clustering_method="hdbscan"
)

entropy_scores = calculator.compute(
    outputs=model_outputs,
    group_by="prompt_family"
)
```

## Key Results

*[To be updated after running experiments]*

### Preliminary Findings:
- **Overall Robustness**: R = 0.XX (avg across 50 prompt families)
- **Most Stable Topics**: [e.g., "Risk metrics calculation", "Portfolio allocation"]
- **Least Stable Topics**: [e.g., "Market sentiment analysis", "Investment advice"]
- **Failure Modes**: High entropy observed in open-ended advisory questions vs. factual queries

### Comparative Benchmarks:
| Model | Avg Robustness (R) | Avg Entropy (H) | Std Dev |
|-------|-------------------|-----------------|---------|
| LLaMA 3 8B | TBD | TBD | TBD |
| GPT-4 | TBD | TBD | TBD |
| Claude 3 | TBD | TBD | TBD |

## Visualizations

### Entropy Heatmap
![Entropy Heatmap](results/figures/entropy_heatmap.png)
*Shows semantic entropy across prompt families and variants*

### Robustness Distribution
![Robustness Distribution](results/figures/robustness_distribution.png)
*Distribution of robustness scores across financial query types*

### Comparative Analysis
![Model Comparison](results/figures/model_comparison.png)
*LLaMA 3 vs other LLMs on identical prompt sets*

## Reproducibility

All experiments are fully reproducible:

1. **Random Seeds**: Fixed seeds for paraphrasing and sampling
2. **Versioned Data**: Dataset versions documented in `data/README.md`
3. **Model Checkpoints**: Exact model versions specified
4. **Environment**: `environment.yml` locks all dependencies

To reproduce results:
```bash
# Set seeds in config/config.yaml
# Run full pipeline
bash scripts/run_full_pipeline.sh
```

## Citations & References

### Datasets
- **FinQA**: Chen et al. (2021) - [HuggingFace](https://huggingface.co/datasets/ibm-research/finqa)
- **Alpaca-Finance**: [HuggingFace](https://huggingface.co/datasets/gbharti/finance-alpaca)
- **BillSum**: Kornilova & Eidelman (2019) - [Papers With Code](https://paperswithcode.com/dataset/billsum)

### Methodology
- **Semantic Entropy**: Farquhar et al. (2024) - [Nature](https://www.nature.com/articles/s41586-024-07421-0)
- **Prompt Perturbations**: Learn Prompting - [Guide](https://learnprompting.org)
- **LLaMA 3**: Meta AI - [Model Card](https://ai.meta.com/llama/)

### Tools
- **Ollama**: [GitHub](https://github.com/ollama/ollama)
- **llama.cpp**: [GitHub](https://github.com/ggerganov/llama.cpp)
- **Sentence-Transformers**: [HuggingFace](https://huggingface.co/sentence-transformers)

## Project Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Dataset Acquisition | 1-2 weeks | ⏳ In Progress |
| Prompt Generation | 1-2 weeks | 📋 Planned |
| LLM Sampling | 2-3 weeks | 📋 Planned |
| Entropy Measurement | 2 weeks | 📋 Planned |
| Metric Development | 0.5-1 week | 📋 Planned |
| Analysis & Viz | 1-2 weeks | 📋 Planned |
| Documentation | 2-3 weeks | 📋 Planned |
| **Total** | **~2-3 months** | |

## Future Extensions

- [ ] Fine-tune LLaMA 3 on finance corpus and re-evaluate
- [ ] Expand to multimodal prompts (charts, tables)
- [ ] Human evaluation of high-entropy responses
- [ ] Real-time API for prompt robustness testing
- [ ] Deployment guide for financial institutions

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Author

**Emmanuel Kwadwo Kusi**
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## Acknowledgments

- Meta AI for LLaMA 3
- Hugging Face for dataset and model hosting
- Research community for semantic entropy methodology

---

**⭐ If this project helps your research or work, please star the repository!**

*Built to demonstrate practical LLM robustness evaluation for financial AI systems*
