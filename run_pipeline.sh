#!/bin/bash
#
# Full Pipeline Execution Script
# Runs complete LLaMA 3 robustness benchmarking pipeline
#
# Author: Emmanuel Kwadwo Kusi

set -e  # Exit on error

echo "========================================"
echo "LLaMA 3 Finance Robustness Benchmarking"
echo "========================================"
echo ""

# Configuration
DATA_DIR="data"
RESULTS_DIR="results"
CONFIG_FILE="config/config.yaml"

# Phase 1: Data Acquisition
echo "[Phase 1/6] Downloading datasets..."
python src/data/download_datasets.py \
    --datasets finqa alpaca-finance billsum \
    --output-dir ${DATA_DIR}/raw

echo ""
echo "[Phase 1/6] Preprocessing datasets..."
python src/data/preprocess.py \
    --input-dir ${DATA_DIR}/raw \
    --output-dir ${DATA_DIR}/processed \
    --max-samples 5000

# Phase 2: Prompt Generation
echo ""
echo "[Phase 2/6] Generating prompt variants..."
python src/models/prompt_generator.py \
    --input ${DATA_DIR}/processed/seed_prompts.csv \
    --output ${DATA_DIR}/prompts \
    --variants 10 \
    --method both \
    --min-similarity 0.85

# Phase 3: LLM Sampling
echo ""
echo "[Phase 3/6] Running LLaMA 3 sampling..."
echo "This may take several hours..."

# Check if Ollama is available
if command -v ollama &> /dev/null; then
    echo "Using Ollama for inference"
    python src/models/llm_runner.py \
        --prompts ${DATA_DIR}/prompts/prompt_variants.json \
        --model llama3 \
        --use-ollama \
        --samples 20 \
        --temperature 0.7 \
        --output ${RESULTS_DIR}/raw_outputs
else
    echo "Using HuggingFace Transformers (4-bit)"
    python src/models/llm_runner.py \
        --prompts ${DATA_DIR}/prompts/prompt_variants.json \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --samples 20 \
        --temperature 0.7 \
        --4bit \
        --output ${RESULTS_DIR}/raw_outputs
fi

# Phase 4: Semantic Entropy
echo ""
echo "[Phase 4/6] Computing semantic entropy..."
python src/evaluation/entropy_calculator.py \
    --inputs ${RESULTS_DIR}/raw_outputs/llama3_outputs.csv \
    --output ${RESULTS_DIR}/metrics \
    --embedder sentence-transformers/all-MiniLM-L6-v2 \
    --clustering hdbscan

# Phase 5: Robustness Metrics
echo ""
echo "[Phase 5/6] Computing robustness metrics..."
python src/evaluation/robustness_metric.py \
    --entropy-file ${RESULTS_DIR}/metrics/entropy_detailed.csv \
    --output ${RESULTS_DIR}/metrics

# Phase 6: Visualization
echo ""
echo "[Phase 6/6] Generating visualizations..."
python src/visualization/plot_results.py \
    --entropy ${RESULTS_DIR}/metrics/entropy_detailed.csv \
    --robustness ${RESULTS_DIR}/metrics/robustness_summary.csv \
    --output ${RESULTS_DIR}/figures

# Summary
echo ""
echo "========================================"
echo "✓ Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Metrics:        ${RESULTS_DIR}/metrics/"
echo "  - Visualizations: ${RESULTS_DIR}/figures/"
echo ""
echo "View interactive dashboard:"
echo "  ${RESULTS_DIR}/figures/interactive_dashboard.html"
echo ""
