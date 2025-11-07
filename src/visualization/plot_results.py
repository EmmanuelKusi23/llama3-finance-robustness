"""
Results Visualization Script
Creates publication-quality visualizations for robustness analysis.

Phase 6: Analysis and Visualization
Author: Emmanuel Kwadwo Kusi
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class RobustnessVisualizer:
    """Creates visualizations for robustness analysis."""

    def __init__(self, output_dir: str = "results/figures"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_entropy_heatmap(
        self,
        entropy_df: pd.DataFrame,
        title: str = "Semantic Entropy Across Prompt Variants"
    ):
        """
        Create heatmap of entropy across prompt families and variants.

        Args:
            entropy_df: DataFrame with entropy values
            title: Plot title
        """
        logger.info("Creating entropy heatmap...")

        # Pivot data for heatmap
        pivot_data = entropy_df.pivot_table(
            values='entropy',
            index='family_id',
            columns='variant_id',
            aggfunc='mean'
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot heatmap
        sns.heatmap(
            pivot_data,
            cmap='RdYlGn_r',  # Red = high entropy, Green = low entropy
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'Semantic Entropy'},
            ax=ax
        )

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Variant ID', fontsize=12)
        ax.set_ylabel('Prompt Family ID', fontsize=12)

        plt.tight_layout()
        save_path = self.output_dir / "entropy_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {save_path}")
        plt.close()

    def plot_robustness_distribution(
        self,
        robustness_df: pd.DataFrame,
        title: str = "Distribution of Robustness Scores"
    ):
        """
        Create distribution plot of robustness scores.

        Args:
            robustness_df: DataFrame with robustness metrics
            title: Plot title
        """
        logger.info("Creating robustness distribution plot...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram with KDE
        axes[0].hist(
            robustness_df['mean_robustness'],
            bins=30,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )
        axes[0].axvline(
            robustness_df['mean_robustness'].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Mean: {robustness_df['mean_robustness'].mean():.3f}"
        )
        axes[0].set_xlabel('Robustness Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Robustness Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Box plot by category
        if 'category' in robustness_df.columns:
            category_order = ['Very Robust', 'Robust', 'Moderately Robust', 'Weak', 'Very Weak']
            categories_present = [c for c in category_order if c in robustness_df['category'].unique()]

            sns.boxplot(
                data=robustness_df,
                y='category',
                x='mean_robustness',
                order=categories_present,
                palette='viridis',
                ax=axes[1]
            )
            axes[1].set_xlabel('Robustness Score', fontsize=12)
            axes[1].set_ylabel('Category', fontsize=12)
            axes[1].set_title('Robustness by Category', fontsize=14, fontweight='bold')
            axes[1].grid(alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.output_dir / "robustness_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")
        plt.close()

    def plot_entropy_vs_robustness(
        self,
        data_df: pd.DataFrame,
        title: str = "Entropy vs Robustness Relationship"
    ):
        """
        Plot relationship between entropy and robustness.

        Args:
            data_df: DataFrame with both metrics
            title: Plot title
        """
        logger.info("Creating entropy vs robustness scatter plot...")

        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            data_df['mean_entropy'],
            data_df['mean_robustness'],
            c=data_df['stability_score'],
            cmap='viridis',
            s=100,
            alpha=0.6,
            edgecolor='black'
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Stability Score', fontsize=12)

        # Add theoretical curve
        x_theory = np.linspace(0, data_df['mean_entropy'].max(), 100)
        y_theory = 1 / (1 + x_theory)
        ax.plot(x_theory, y_theory, 'r--', linewidth=2, label='Theoretical: R = 1/(1+H)')

        ax.set_xlabel('Mean Entropy (H)', fontsize=12)
        ax.set_ylabel('Mean Robustness (R)', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "entropy_vs_robustness.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {save_path}")
        plt.close()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        title: str = "Model Robustness Comparison"
    ):
        """
        Create comparative visualization across models.

        Args:
            comparison_df: DataFrame with model comparison metrics
            title: Plot title
        """
        logger.info("Creating model comparison plot...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mean Robustness
        axes[0, 0].barh(
            comparison_df['model'],
            comparison_df['mean_robustness'],
            color='steelblue',
            edgecolor='black'
        )
        axes[0, 0].set_xlabel('Mean Robustness', fontsize=12)
        axes[0, 0].set_title('Mean Robustness by Model', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. Mean Entropy
        axes[0, 1].barh(
            comparison_df['model'],
            comparison_df['mean_entropy'],
            color='coral',
            edgecolor='black'
        )
        axes[0, 1].set_xlabel('Mean Entropy', fontsize=12)
        axes[0, 1].set_title('Mean Entropy by Model', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Robustness with error bars
        axes[1, 0].barh(
            comparison_df['model'],
            comparison_df['mean_robustness'],
            xerr=comparison_df['std_robustness'],
            color='mediumseagreen',
            edgecolor='black',
            capsize=5
        )
        axes[1, 0].set_xlabel('Robustness (with std)', fontsize=12)
        axes[1, 0].set_title('Robustness Variability', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)

        # 4. Entropy with error bars
        axes[1, 1].barh(
            comparison_df['model'],
            comparison_df['mean_entropy'],
            xerr=comparison_df['std_entropy'],
            color='mediumpurple',
            edgecolor='black',
            capsize=5
        )
        axes[1, 1].set_xlabel('Entropy (with std)', fontsize=12)
        axes[1, 1].set_title('Entropy Variability', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
        plt.close()

    def plot_top_bottom_prompts(
        self,
        robustness_df: pd.DataFrame,
        n: int = 10,
        title: str = "Most and Least Robust Prompts"
    ):
        """
        Visualize top and bottom performing prompts.

        Args:
            robustness_df: DataFrame with robustness scores
            n: Number of top/bottom prompts to show
            title: Plot title
        """
        logger.info("Creating top/bottom prompts visualization...")

        # Get top and bottom
        top_n = robustness_df.nlargest(n, 'mean_robustness')
        bottom_n = robustness_df.nsmallest(n, 'mean_robustness')

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Top performers
        axes[0].barh(
            range(len(top_n)),
            top_n['mean_robustness'],
            color='forestgreen',
            edgecolor='black'
        )
        axes[0].set_yticks(range(len(top_n)))
        axes[0].set_yticklabels(top_n['family_id'], fontsize=9)
        axes[0].set_xlabel('Robustness Score', fontsize=12)
        axes[0].set_title(f'Top {n} Most Robust Prompts', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)

        # Bottom performers
        axes[1].barh(
            range(len(bottom_n)),
            bottom_n['mean_robustness'],
            color='crimson',
            edgecolor='black'
        )
        axes[1].set_yticks(range(len(bottom_n)))
        axes[1].set_yticklabels(bottom_n['family_id'], fontsize=9)
        axes[1].set_xlabel('Robustness Score', fontsize=12)
        axes[1].set_title(f'Top {n} Least Robust Prompts', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / "top_bottom_prompts.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved top/bottom plot to {save_path}")
        plt.close()

    def create_interactive_dashboard(
        self,
        robustness_df: pd.DataFrame,
        entropy_df: pd.DataFrame
    ):
        """
        Create interactive Plotly dashboard.

        Args:
            robustness_df: DataFrame with robustness metrics
            entropy_df: DataFrame with detailed entropy
        """
        logger.info("Creating interactive dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Robustness Distribution',
                'Entropy vs Robustness',
                'Top 10 Prompts',
                'Category Breakdown'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'pie'}]
            ]
        )

        # 1. Robustness histogram
        fig.add_trace(
            go.Histogram(
                x=robustness_df['mean_robustness'],
                name='Robustness',
                marker_color='steelblue'
            ),
            row=1, col=1
        )

        # 2. Entropy vs Robustness scatter
        fig.add_trace(
            go.Scatter(
                x=robustness_df['mean_entropy'],
                y=robustness_df['mean_robustness'],
                mode='markers',
                name='Prompts',
                marker=dict(
                    size=8,
                    color=robustness_df['stability_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Stability")
                ),
                text=robustness_df['family_id'],
                hovertemplate='<b>%{text}</b><br>Entropy: %{x:.3f}<br>Robustness: %{y:.3f}'
            ),
            row=1, col=2
        )

        # 3. Top 10 prompts
        top10 = robustness_df.nlargest(10, 'mean_robustness')
        fig.add_trace(
            go.Bar(
                y=top10['family_id'],
                x=top10['mean_robustness'],
                orientation='h',
                name='Top 10',
                marker_color='forestgreen'
            ),
            row=2, col=1
        )

        # 4. Category pie chart
        if 'category' in robustness_df.columns:
            category_counts = robustness_df['category'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    name='Categories'
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="LLaMA 3 Robustness Analysis Dashboard",
            title_font_size=20,
            showlegend=False,
            height=900
        )

        # Save interactive HTML
        html_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(html_path))
        logger.info(f"Saved interactive dashboard to {html_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for robustness analysis"
    )
    parser.add_argument(
        '--entropy',
        type=str,
        default='results/metrics/entropy_detailed.csv',
        help='Path to entropy CSV'
    )
    parser.add_argument(
        '--robustness',
        type=str,
        default='results/metrics/robustness_summary.csv',
        help='Path to robustness CSV'
    )
    parser.add_argument(
        '--comparison',
        type=str,
        help='Path to model comparison CSV (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/figures',
        help='Output directory for figures'
    )

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = RobustnessVisualizer(output_dir=args.output)

    # Load data
    logger.info("Loading data...")
    entropy_df = pd.read_csv(args.entropy)
    robustness_df = pd.read_csv(args.robustness)

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    visualizer.plot_entropy_heatmap(entropy_df)
    visualizer.plot_robustness_distribution(robustness_df)
    visualizer.plot_entropy_vs_robustness(robustness_df)
    visualizer.plot_top_bottom_prompts(robustness_df, n=10)
    visualizer.create_interactive_dashboard(robustness_df, entropy_df)

    # Model comparison if provided
    if args.comparison and Path(args.comparison).exists():
        comparison_df = pd.read_csv(args.comparison)
        visualizer.plot_model_comparison(comparison_df)

    logger.info("\n✓ All visualizations generated successfully!")
    logger.info(f"Figures saved to: {args.output}")


if __name__ == "__main__":
    main()
