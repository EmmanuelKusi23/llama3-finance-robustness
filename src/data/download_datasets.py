"""
Dataset Download Script
Downloads financial datasets from HuggingFace and other sources.

Phase 1: Dataset Acquisition
Author: Emmanuel Kwadwo Kusi
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Handles downloading and initial processing of financial datasets."""

    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize the dataset downloader.

        Args:
            output_dir: Directory to save downloaded datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_finqa(self) -> pd.DataFrame:
        """
        Download FinQA dataset from HuggingFace.

        FinQA: A large-scale financial question answering dataset
        Contains 8,281 Q&A pairs from 2,871 earnings reports
        License: CC-BY 4.0

        Returns:
            DataFrame with FinQA data
        """
        logger.info("Downloading FinQA dataset...")
        try:
            dataset = load_dataset("ibm/finqa", split="train")

            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)

            # Save to disk
            output_path = self.output_dir / "finqa_raw.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"FinQA saved to {output_path}")
            logger.info(f"Total samples: {len(df)}")

            return df

        except Exception as e:
            logger.error(f"Error downloading FinQA: {e}")
            raise

    def download_alpaca_finance(self) -> pd.DataFrame:
        """
        Download Alpaca-Finance dataset from HuggingFace.

        Alpaca-Finance: Financial instruction-following dataset
        Contains ~70k finance-related instruction-output pairs
        License: MIT

        Returns:
            DataFrame with Alpaca-Finance data
        """
        logger.info("Downloading Alpaca-Finance dataset...")
        try:
            dataset = load_dataset("gbharti/finance-alpaca", split="train")

            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)

            # Save to disk
            output_path = self.output_dir / "alpaca_finance_raw.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Alpaca-Finance saved to {output_path}")
            logger.info(f"Total samples: {len(df)}")

            return df

        except Exception as e:
            logger.error(f"Error downloading Alpaca-Finance: {e}")
            raise

    def download_billsum(self) -> pd.DataFrame:
        """
        Download BillSum dataset from HuggingFace.

        BillSum: Congressional bill summarization dataset
        Contains ~23k US Congressional bills with summaries
        License: CC0 1.0 Universal

        Returns:
            DataFrame with BillSum data
        """
        logger.info("Downloading BillSum dataset...")
        try:
            dataset = load_dataset("billsum", split="train")

            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)

            # Save to disk
            output_path = self.output_dir / "billsum_raw.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"BillSum saved to {output_path}")
            logger.info(f"Total samples: {len(df)}")

            return df

        except Exception as e:
            logger.error(f"Error downloading BillSum: {e}")
            raise

    def download_financial_phrasebank(self) -> pd.DataFrame:
        """
        Download Financial PhraseBank dataset.

        Financial PhraseBank: Sentiment analysis on financial news
        Contains 4,840 sentences from financial news
        License: CC-BY-NC-SA 3.0

        Returns:
            DataFrame with Financial PhraseBank data
        """
        logger.info("Downloading Financial PhraseBank dataset...")
        try:
            dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train")

            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)

            # Save to disk
            output_path = self.output_dir / "financial_phrasebank_raw.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Financial PhraseBank saved to {output_path}")
            logger.info(f"Total samples: {len(df)}")

            return df

        except Exception as e:
            logger.error(f"Error downloading Financial PhraseBank: {e}")
            raise

    def download_all(self, datasets: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Download all specified datasets.

        Args:
            datasets: List of dataset names to download.
                     Options: 'finqa', 'alpaca-finance', 'billsum', 'phrasebank'
                     If None, downloads all datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        if datasets is None:
            datasets = ['finqa', 'alpaca-finance', 'billsum', 'phrasebank']

        dataset_map = {
            'finqa': self.download_finqa,
            'alpaca-finance': self.download_alpaca_finance,
            'billsum': self.download_billsum,
            'phrasebank': self.download_financial_phrasebank
        }

        results = {}
        for dataset_name in datasets:
            if dataset_name in dataset_map:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing: {dataset_name}")
                logger.info(f"{'='*60}")
                try:
                    results[dataset_name] = dataset_map[dataset_name]()
                except Exception as e:
                    logger.error(f"Failed to download {dataset_name}: {e}")
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")

        return results

    def generate_summary(self) -> pd.DataFrame:
        """
        Generate a summary of downloaded datasets.

        Returns:
            DataFrame with dataset statistics
        """
        summary_data = []

        for filepath in self.output_dir.glob("*.csv"):
            try:
                df = pd.read_csv(filepath, nrows=1000)  # Sample for efficiency
                summary_data.append({
                    'dataset': filepath.stem,
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                    'estimated_rows': len(df) * (filepath.stat().st_size / df.memory_usage(deep=True).sum()),
                    'columns': len(df.columns),
                    'column_names': ', '.join(df.columns.tolist()[:5])
                })
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "dataset_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nDataset summary saved to {summary_path}")

        return summary_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download financial datasets for LLaMA 3 robustness evaluation"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        choices=['finqa', 'alpaca-finance', 'billsum', 'phrasebank'],
        help='Datasets to download (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for datasets'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = DatasetDownloader(output_dir=args.output_dir)

    # Download datasets
    logger.info("Starting dataset download...")
    results = downloader.download_all(datasets=args.datasets)

    # Generate summary
    logger.info("\nGenerating dataset summary...")
    summary = downloader.generate_summary()
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))

    logger.info("\n✓ All datasets downloaded successfully!")
    logger.info(f"Data saved to: {downloader.output_dir}")


if __name__ == "__main__":
    main()
