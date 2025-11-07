"""
Semantic Entropy Calculator
Clusters LLM outputs and computes semantic entropy as a measure of response consistency.

Phase 4: Semantic Entropy Measurement
Author: Emmanuel Kwadwo Kusi
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan
from scipy.stats import entropy as scipy_entropy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticEntropy:
    """Computes semantic entropy from LLM outputs."""

    def __init__(
        self,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        clustering_method: str = "hdbscan",
        device: str = None
    ):
        """
        Initialize semantic entropy calculator.

        Args:
            embedder_model: Sentence transformer model for embeddings
            clustering_method: Clustering algorithm ('hdbscan', 'kmeans', 'dbscan')
            device: Device for embedder (cuda/cpu)
        """
        self.clustering_method = clustering_method

        # Load embedding model
        logger.info(f"Loading embedding model: {embedder_model}")
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embedder_model, device=self.device)
        logger.info(f"Embedder loaded on {self.device}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        # Filter empty texts
        texts = [t if t else " " for t in texts]

        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 2
    ) -> np.ndarray:
        """
        Cluster embeddings using specified method.

        Args:
            embeddings: Embedding vectors
            min_cluster_size: Minimum cluster size for HDBSCAN

        Returns:
            Cluster labels array
        """
        n_samples = len(embeddings)

        if n_samples < 2:
            return np.array([0])

        if self.clustering_method == "hdbscan":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min_cluster_size),
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(embeddings)

        elif self.clustering_method == "kmeans":
            # Auto-determine k using elbow method
            max_k = min(10, n_samples)
            best_k = self._find_optimal_k(embeddings, max_k)

            clusterer = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)

        elif self.clustering_method == "dbscan":
            # Use DBSCAN with auto epsilon
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min(5, n_samples))
            neighbors_fit = neighbors.fit(embeddings)
            distances, _ = neighbors_fit.kneighbors(embeddings)
            eps = np.median(distances[:, -1])

            clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size)
            labels = clusterer.fit_predict(embeddings)

        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        return labels

    def _find_optimal_k(self, embeddings: np.ndarray, max_k: int) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            embeddings: Embedding vectors
            max_k: Maximum number of clusters to try

        Returns:
            Optimal k
        """
        if len(embeddings) < 4:
            return min(2, len(embeddings))

        best_k = 2
        best_score = -1

        for k in range(2, min(max_k + 1, len(embeddings))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue

        return best_k

    def compute_entropy(self, labels: np.ndarray) -> float:
        """
        Compute Shannon entropy from cluster labels.

        Args:
            labels: Cluster assignment labels

        Returns:
            Shannon entropy value
        """
        # Get cluster counts (ignore noise label -1 for HDBSCAN)
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Filter out noise points (label -1)
        if -1 in unique_labels:
            mask = unique_labels != -1
            counts = counts[mask]

        if len(counts) == 0:
            return 0.0

        # Compute probabilities
        probabilities = counts / counts.sum()

        # Compute Shannon entropy
        H = scipy_entropy(probabilities, base=2)

        return H

    def analyze_prompt_family(
        self,
        outputs_df: pd.DataFrame,
        family_id: str
    ) -> Dict:
        """
        Analyze outputs for a single prompt family.

        Args:
            outputs_df: DataFrame with LLM outputs
            family_id: Prompt family identifier

        Returns:
            Dictionary with entropy metrics
        """
        family_data = outputs_df[outputs_df['family_id'] == family_id]

        if len(family_data) == 0:
            return {}

        results = {
            'family_id': family_id,
            'variant_entropies': [],
            'variant_cluster_counts': [],
            'total_responses': len(family_data)
        }

        # Analyze each variant
        for variant_id in family_data['variant_id'].unique():
            variant_data = family_data[family_data['variant_id'] == variant_id]
            responses = variant_data['response'].tolist()

            # Skip if too few responses
            if len(responses) < 2:
                continue

            # Embed responses
            embeddings = self.embed_texts(responses)

            # Cluster
            labels = self.cluster_embeddings(embeddings)

            # Compute entropy
            H = self.compute_entropy(labels)

            # Count unique clusters
            n_clusters = len(np.unique(labels[labels >= 0]))

            results['variant_entropies'].append({
                'variant_id': variant_id,
                'entropy': H,
                'n_clusters': n_clusters,
                'n_responses': len(responses)
            })

        # Compute average entropy for family
        if results['variant_entropies']:
            entropies = [v['entropy'] for v in results['variant_entropies']]
            results['mean_entropy'] = np.mean(entropies)
            results['std_entropy'] = np.std(entropies)
            results['max_entropy'] = np.max(entropies)
            results['min_entropy'] = np.min(entropies)
        else:
            results['mean_entropy'] = 0.0
            results['std_entropy'] = 0.0
            results['max_entropy'] = 0.0
            results['min_entropy'] = 0.0

        return results

    def compute_all(
        self,
        outputs_df: pd.DataFrame,
        output_dir: str = "results/metrics"
    ) -> pd.DataFrame:
        """
        Compute semantic entropy for all prompt families.

        Args:
            outputs_df: DataFrame with all LLM outputs
            output_dir: Directory to save results

        Returns:
            DataFrame with entropy metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all unique families
        families = outputs_df['family_id'].unique()
        logger.info(f"Computing entropy for {len(families)} prompt families")

        all_results = []

        for family_id in tqdm(families, desc="Computing entropy"):
            result = self.analyze_prompt_family(outputs_df, family_id)
            if result:
                all_results.append(result)

        # Convert to DataFrame
        # Flatten variant entropies
        rows = []
        for result in all_results:
            base_data = {
                'family_id': result['family_id'],
                'mean_entropy': result['mean_entropy'],
                'std_entropy': result['std_entropy'],
                'max_entropy': result['max_entropy'],
                'min_entropy': result['min_entropy'],
                'total_responses': result['total_responses']
            }

            for variant in result['variant_entropies']:
                row = base_data.copy()
                row.update(variant)
                rows.append(row)

        entropy_df = pd.DataFrame(rows)

        # Save detailed results
        detailed_path = output_dir / "entropy_detailed.csv"
        entropy_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed entropy saved to {detailed_path}")

        # Save summary by family
        summary_df = entropy_df.groupby('family_id').agg({
            'mean_entropy': 'first',
            'std_entropy': 'first',
            'max_entropy': 'first',
            'min_entropy': 'first',
            'total_responses': 'first'
        }).reset_index()

        summary_path = output_dir / "entropy_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary entropy saved to {summary_path}")

        # Save as JSON
        json_path = output_dir / "entropy_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Generate statistics
        logger.info("\n" + "="*60)
        logger.info("ENTROPY STATISTICS")
        logger.info("="*60)
        logger.info(f"Mean entropy across families: {summary_df['mean_entropy'].mean():.3f}")
        logger.info(f"Std entropy: {summary_df['mean_entropy'].std():.3f}")
        logger.info(f"Max entropy: {summary_df['max_entropy'].max():.3f}")
        logger.info(f"Min entropy: {summary_df['min_entropy'].min():.3f}")

        return entropy_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute semantic entropy from LLM outputs"
    )
    parser.add_argument(
        '--inputs',
        type=str,
        default='results/raw_outputs/llama3_outputs.csv',
        help='Path to LLM outputs CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/metrics',
        help='Output directory for metrics'
    )
    parser.add_argument(
        '--embedder',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model'
    )
    parser.add_argument(
        '--clustering',
        type=str,
        default='hdbscan',
        choices=['hdbscan', 'kmeans', 'dbscan'],
        help='Clustering method'
    )

    args = parser.parse_args()

    # Load outputs
    logger.info(f"Loading outputs from {args.inputs}")
    outputs_df = pd.read_csv(args.inputs)
    logger.info(f"Loaded {len(outputs_df)} outputs")

    # Initialize calculator
    calculator = SemanticEntropy(
        embedder_model=args.embedder,
        clustering_method=args.clustering
    )

    # Compute entropy
    logger.info("Computing semantic entropy...")
    entropy_df = calculator.compute_all(
        outputs_df=outputs_df,
        output_dir=args.output
    )

    logger.info("\n✓ Entropy computation completed!")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
