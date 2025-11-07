"""
Robustness Metric Calculator
Computes robustness scores based on semantic entropy.

Phase 5: Robustness Metric
Author: Emmanuel Kwadwo Kusi
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobustnessCalculator:
    """Calculates robustness metrics from semantic entropy."""

    def __init__(self, entropy_weight: float = 1.0):
        """
        Initialize robustness calculator.

        Args:
            entropy_weight: Weight for entropy in robustness formula
        """
        self.entropy_weight = entropy_weight

    def compute_robustness(self, entropy: float) -> float:
        """
        Compute robustness score from entropy.

        R = 1 / (1 + H)

        Where H is semantic entropy. Lower entropy (more consistent) = higher robustness.

        Args:
            entropy: Semantic entropy value

        Returns:
            Robustness score in [0, 1]
        """
        if pd.isna(entropy) or entropy < 0:
            return 0.0

        robustness = 1.0 / (1.0 + self.entropy_weight * entropy)
        return robustness

    def compute_stability_score(
        self,
        entropies: List[float]
    ) -> Dict[str, float]:
        """
        Compute stability metrics from multiple entropy values.

        Args:
            entropies: List of entropy values across variants

        Returns:
            Dictionary with stability metrics
        """
        entropies = [e for e in entropies if not pd.isna(e)]

        if not entropies:
            return {
                'mean_robustness': 0.0,
                'std_robustness': 0.0,
                'min_robustness': 0.0,
                'max_robustness': 0.0,
                'stability_score': 0.0
            }

        # Convert entropies to robustness scores
        robustness_scores = [self.compute_robustness(e) for e in entropies]

        # Compute statistics
        mean_R = np.mean(robustness_scores)
        std_R = np.std(robustness_scores)

        # Stability score: penalize high variance
        # Higher stability when variance is low
        stability = mean_R * (1.0 - min(std_R, 1.0))

        return {
            'mean_robustness': mean_R,
            'std_robustness': std_R,
            'min_robustness': np.min(robustness_scores),
            'max_robustness': np.max(robustness_scores),
            'stability_score': stability,
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies)
        }

    def categorize_robustness(self, robustness: float) -> str:
        """
        Categorize robustness score.

        Args:
            robustness: Robustness score

        Returns:
            Category label
        """
        if robustness >= 0.8:
            return "Very Robust"
        elif robustness >= 0.6:
            return "Robust"
        elif robustness >= 0.4:
            return "Moderately Robust"
        elif robustness >= 0.2:
            return "Weak"
        else:
            return "Very Weak"

    def compute_from_entropy_file(
        self,
        entropy_file: str,
        output_dir: str = "results/metrics"
    ) -> pd.DataFrame:
        """
        Compute robustness metrics from entropy CSV.

        Args:
            entropy_file: Path to entropy results CSV
            output_dir: Output directory for metrics

        Returns:
            DataFrame with robustness metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load entropy data
        logger.info(f"Loading entropy data from {entropy_file}")
        entropy_df = pd.read_csv(entropy_file)

        # Compute robustness for each row
        entropy_df['robustness'] = entropy_df['entropy'].apply(self.compute_robustness)
        entropy_df['robustness_category'] = entropy_df['robustness'].apply(
            self.categorize_robustness
        )

        # Compute family-level statistics
        family_stats = []

        for family_id in entropy_df['family_id'].unique():
            family_data = entropy_df[entropy_df['family_id'] == family_id]

            # Get entropies for this family
            entropies = family_data['entropy'].tolist()

            # Compute stability metrics
            stability = self.compute_stability_score(entropies)

            family_stats.append({
                'family_id': family_id,
                **stability,
                'n_variants': len(family_data),
                'category': self.categorize_robustness(stability['mean_robustness'])
            })

        family_df = pd.DataFrame(family_stats)

        # Save detailed results
        detailed_path = output_dir / "robustness_detailed.csv"
        entropy_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed robustness saved to {detailed_path}")

        # Save family summary
        summary_path = output_dir / "robustness_summary.csv"
        family_df.to_csv(summary_path, index=False)
        logger.info(f"Family summary saved to {summary_path}")

        # Generate overall statistics
        self._print_statistics(family_df)

        return family_df

    def _print_statistics(self, family_df: pd.DataFrame):
        """Print robustness statistics."""
        logger.info("\n" + "="*60)
        logger.info("ROBUSTNESS STATISTICS")
        logger.info("="*60)

        # Overall metrics
        logger.info(f"Total prompt families: {len(family_df)}")
        logger.info(f"Mean robustness: {family_df['mean_robustness'].mean():.3f}")
        logger.info(f"Std robustness: {family_df['mean_robustness'].std():.3f}")
        logger.info(f"Mean stability: {family_df['stability_score'].mean():.3f}")

        # Category distribution
        logger.info("\nRobustness Categories:")
        category_counts = family_df['category'].value_counts()
        for category, count in category_counts.items():
            pct = 100 * count / len(family_df)
            logger.info(f"  {category}: {count} ({pct:.1f}%)")

        # Top and bottom performers
        logger.info("\nTop 5 Most Robust Prompts:")
        top5 = family_df.nlargest(5, 'mean_robustness')[['family_id', 'mean_robustness', 'mean_entropy']]
        for _, row in top5.iterrows():
            logger.info(f"  {row['family_id']}: R={row['mean_robustness']:.3f}, H={row['mean_entropy']:.3f}")

        logger.info("\nTop 5 Least Robust Prompts:")
        bottom5 = family_df.nsmallest(5, 'mean_robustness')[['family_id', 'mean_robustness', 'mean_entropy']]
        for _, row in bottom5.iterrows():
            logger.info(f"  {row['family_id']}: R={row['mean_robustness']:.3f}, H={row['mean_entropy']:.3f}")

    def compare_models(
        self,
        entropy_files: Dict[str, str],
        output_dir: str = "results/metrics"
    ) -> pd.DataFrame:
        """
        Compare robustness across multiple models.

        Args:
            entropy_files: Dictionary mapping model names to entropy CSV files
            output_dir: Output directory

        Returns:
            DataFrame with comparative metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_data = []

        for model_name, entropy_file in entropy_files.items():
            logger.info(f"\nProcessing {model_name}...")

            # Load and compute robustness
            entropy_df = pd.read_csv(entropy_file)
            entropy_df['robustness'] = entropy_df['entropy'].apply(self.compute_robustness)

            # Compute model-level statistics
            model_stats = {
                'model': model_name,
                'mean_robustness': entropy_df['robustness'].mean(),
                'std_robustness': entropy_df['robustness'].std(),
                'mean_entropy': entropy_df['entropy'].mean(),
                'std_entropy': entropy_df['entropy'].std(),
                'n_families': entropy_df['family_id'].nunique(),
                'n_total_outputs': len(entropy_df)
            }

            comparison_data.append(model_stats)

        comparison_df = pd.DataFrame(comparison_data)

        # Save comparison
        comparison_path = output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nModel comparison saved to {comparison_path}")

        # Print comparison
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        print(comparison_df.to_string(index=False))

        return comparison_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute robustness metrics from semantic entropy"
    )
    parser.add_argument(
        '--entropy-file',
        type=str,
        default='results/metrics/entropy_detailed.csv',
        help='Path to entropy results CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/metrics',
        help='Output directory for robustness metrics'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Additional entropy files for model comparison (format: model_name:file_path)'
    )

    args = parser.parse_args()

    # Initialize calculator
    calculator = RobustnessCalculator()

    # Compute robustness
    logger.info("Computing robustness metrics...")
    robustness_df = calculator.compute_from_entropy_file(
        entropy_file=args.entropy_file,
        output_dir=args.output
    )

    # If comparison files provided
    if args.compare:
        logger.info("\nComparing across models...")

        entropy_files = {}
        for item in args.compare:
            if ':' in item:
                model_name, file_path = item.split(':', 1)
                entropy_files[model_name] = file_path

        if entropy_files:
            comparison_df = calculator.compare_models(
                entropy_files=entropy_files,
                output_dir=args.output
            )

    logger.info("\n✓ Robustness computation completed!")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
