"""
Data Preprocessing Script
Cleans and prepares financial datasets for prompt generation.

Phase 1: Dataset Acquisition
Author: Emmanuel Kwadwo Kusi
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinanceDataPreprocessor:
    """Preprocesses financial datasets for prompt generation."""

    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        """
        Initialize the preprocessor.

        Args:
            input_dir: Directory containing raw datasets
            output_dir: Directory to save processed datasets
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text string

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to string
        text = str(text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s$%.,!?-]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def process_finqa(self, max_samples: int = None) -> pd.DataFrame:
        """
        Process FinQA dataset and extract seed prompts.

        Args:
            max_samples: Maximum number of samples to process (None for all)

        Returns:
            Processed DataFrame
        """
        logger.info("Processing FinQA dataset...")

        input_path = self.input_dir / "finqa_raw.csv"
        if not input_path.exists():
            logger.error(f"FinQA file not found: {input_path}")
            return pd.DataFrame()

        df = pd.read_csv(input_path)

        if max_samples:
            df = df.head(max_samples)

        # Extract relevant columns
        processed_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing FinQA"):
            # Clean question and answer
            question = self.clean_text(row.get('question', ''))
            answer = self.clean_text(str(row.get('answer', '')))

            if len(question) > 10 and len(answer) > 0:
                processed_data.append({
                    'dataset': 'finqa',
                    'prompt_type': 'qa',
                    'question': question,
                    'answer': answer,
                    'context': self.clean_text(str(row.get('pre_text', '')))[:500],
                    'topic': 'financial_qa',
                    'complexity': 'medium'
                })

        processed_df = pd.DataFrame(processed_data)

        # Save processed data
        output_path = self.output_dir / "finqa_processed.csv"
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed FinQA saved to {output_path} ({len(processed_df)} samples)")

        return processed_df

    def process_alpaca_finance(self, max_samples: int = None) -> pd.DataFrame:
        """
        Process Alpaca-Finance dataset.

        Args:
            max_samples: Maximum number of samples to process

        Returns:
            Processed DataFrame
        """
        logger.info("Processing Alpaca-Finance dataset...")

        input_path = self.input_dir / "alpaca_finance_raw.csv"
        if not input_path.exists():
            logger.error(f"Alpaca-Finance file not found: {input_path}")
            return pd.DataFrame()

        df = pd.read_csv(input_path)

        if max_samples:
            df = df.head(max_samples)

        processed_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Alpaca-Finance"):
            instruction = self.clean_text(row.get('instruction', ''))
            input_text = self.clean_text(row.get('input', ''))
            output_text = self.clean_text(row.get('output', ''))

            # Combine instruction and input as question
            if input_text:
                question = f"{instruction} {input_text}"
            else:
                question = instruction

            if len(question) > 10 and len(output_text) > 0:
                processed_data.append({
                    'dataset': 'alpaca_finance',
                    'prompt_type': 'instruction',
                    'question': question,
                    'answer': output_text,
                    'context': '',
                    'topic': 'financial_instruction',
                    'complexity': 'varied'
                })

        processed_df = pd.DataFrame(processed_data)

        # Save processed data
        output_path = self.output_dir / "alpaca_finance_processed.csv"
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed Alpaca-Finance saved to {output_path} ({len(processed_df)} samples)")

        return processed_df

    def process_billsum(self, max_samples: int = None) -> pd.DataFrame:
        """
        Process BillSum dataset.

        Args:
            max_samples: Maximum number of samples to process

        Returns:
            Processed DataFrame
        """
        logger.info("Processing BillSum dataset...")

        input_path = self.input_dir / "billsum_raw.csv"
        if not input_path.exists():
            logger.error(f"BillSum file not found: {input_path}")
            return pd.DataFrame()

        df = pd.read_csv(input_path)

        if max_samples:
            df = df.head(max_samples)

        processed_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing BillSum"):
            text = self.clean_text(row.get('text', ''))
            summary = self.clean_text(row.get('summary', ''))

            # Create summarization prompts
            question = f"Summarize the following financial/legal document: {text[:500]}"

            if len(question) > 20 and len(summary) > 0:
                processed_data.append({
                    'dataset': 'billsum',
                    'prompt_type': 'summarization',
                    'question': question,
                    'answer': summary,
                    'context': text[:1000],
                    'topic': 'legal_financial',
                    'complexity': 'high'
                })

        processed_df = pd.DataFrame(processed_data)

        # Save processed data
        output_path = self.output_dir / "billsum_processed.csv"
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed BillSum saved to {output_path} ({len(processed_df)} samples)")

        return processed_df

    def extract_seed_prompts(self, n_prompts: int = 50) -> pd.DataFrame:
        """
        Extract diverse seed prompts from processed datasets.

        Args:
            n_prompts: Number of seed prompts to extract

        Returns:
            DataFrame with seed prompts
        """
        logger.info(f"Extracting {n_prompts} seed prompts...")

        # Load all processed datasets
        all_data = []
        for filepath in self.output_dir.glob("*_processed.csv"):
            df = pd.read_csv(filepath)
            all_data.append(df)

        if not all_data:
            logger.error("No processed datasets found")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)

        # Sample diverse prompts
        # Strategy: stratified sampling by topic and complexity
        seed_prompts = []

        # Group by topic
        for topic, group in combined_df.groupby('topic'):
            # Sample proportionally
            n_samples = max(1, int(len(group) / len(combined_df) * n_prompts))
            samples = group.sample(n=min(n_samples, len(group)), random_state=42)
            seed_prompts.append(samples)

        seed_df = pd.concat(seed_prompts, ignore_index=True)

        # Ensure we have exactly n_prompts
        if len(seed_df) > n_prompts:
            seed_df = seed_df.sample(n=n_prompts, random_state=42)
        elif len(seed_df) < n_prompts:
            # Add more random samples
            remaining = n_prompts - len(seed_df)
            additional = combined_df.sample(n=remaining, random_state=42)
            seed_df = pd.concat([seed_df, additional], ignore_index=True)

        # Add family IDs
        seed_df['family_id'] = [f"family_{i:03d}" for i in range(len(seed_df))]

        # Save seed prompts
        output_path = self.output_dir / "seed_prompts.csv"
        seed_df.to_csv(output_path, index=False)
        logger.info(f"Seed prompts saved to {output_path}")

        return seed_df

    def process_all(self, max_samples_per_dataset: int = 5000) -> Dict[str, pd.DataFrame]:
        """
        Process all datasets.

        Args:
            max_samples_per_dataset: Maximum samples per dataset

        Returns:
            Dictionary of processed DataFrames
        """
        results = {}

        # Process each dataset
        results['finqa'] = self.process_finqa(max_samples=max_samples_per_dataset)
        results['alpaca_finance'] = self.process_alpaca_finance(max_samples=max_samples_per_dataset)
        results['billsum'] = self.process_billsum(max_samples=max_samples_per_dataset)

        # Extract seed prompts
        results['seed_prompts'] = self.extract_seed_prompts(n_prompts=50)

        # Generate summary
        self.generate_summary()

        return results

    def generate_summary(self):
        """Generate preprocessing summary."""
        summary_data = []

        for filepath in self.output_dir.glob("*_processed.csv"):
            df = pd.read_csv(filepath)
            summary_data.append({
                'dataset': filepath.stem,
                'samples': len(df),
                'avg_question_length': df['question'].str.len().mean(),
                'avg_answer_length': df['answer'].str.len().mean(),
                'topics': df['topic'].nunique() if 'topic' in df.columns else 0
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "preprocessing_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*60)
        print(summary_df.to_string(index=False))


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Preprocess financial datasets"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Input directory with raw datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed datasets'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000,
        help='Maximum samples per dataset'
    )

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = FinanceDataPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

    # Process all datasets
    logger.info("Starting data preprocessing...")
    results = preprocessor.process_all(max_samples_per_dataset=args.max_samples)

    logger.info("\n✓ All datasets processed successfully!")
    logger.info(f"Processed data saved to: {preprocessor.output_dir}")


if __name__ == "__main__":
    main()
