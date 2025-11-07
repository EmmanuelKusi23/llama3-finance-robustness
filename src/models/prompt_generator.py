"""
Prompt Generation Script
Generates paraphrased prompt variants using back-translation and T5 models.

Phase 2: Prompt Generation
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
import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptParaphraser:
    """Generates paraphrased prompt variants."""

    def __init__(
        self,
        paraphrase_method: str = "backtranslation",
        device: str = None
    ):
        """
        Initialize the paraphraser.

        Args:
            paraphrase_method: Method to use ('backtranslation', 't5', 'both')
            device: Device to run models on (cuda/cpu)
        """
        self.method = paraphrase_method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize models
        self.models = {}
        self._initialize_models()

        # Semantic similarity model for validation
        logger.info("Loading semantic similarity model...")
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_model.to(self.device)

    def _initialize_models(self):
        """Initialize paraphrasing models."""
        if self.method in ["backtranslation", "both"]:
            self._load_translation_models()

        if self.method in ["t5", "both"]:
            self._load_t5_model()

    def _load_translation_models(self):
        """Load MarianMT translation models for back-translation."""
        logger.info("Loading translation models for back-translation...")

        # English to French
        self.models['en_to_fr'] = {
            'model': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr').to(self.device),
            'tokenizer': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        }

        # French to English
        self.models['fr_to_en'] = {
            'model': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en').to(self.device),
            'tokenizer': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        }

        logger.info("Translation models loaded")

    def _load_t5_model(self):
        """Load T5 model for paraphrasing."""
        logger.info("Loading T5 paraphrase model...")

        # Using T5-base for paraphrasing
        model_name = "t5-base"
        self.models['t5'] = {
            'model': T5ForConditionalGeneration.from_pretrained(model_name).to(self.device),
            'tokenizer': T5Tokenizer.from_pretrained(model_name)
        }

        logger.info("T5 model loaded")

    def backtranslate(self, text: str, intermediate_lang: str = "fr") -> str:
        """
        Paraphrase text via back-translation.

        Args:
            text: Original text
            intermediate_lang: Intermediate language for back-translation

        Returns:
            Back-translated text
        """
        try:
            # Translate to intermediate language (French)
            en_to_fr = self.models['en_to_fr']
            inputs = en_to_fr['tokenizer'](text, return_tensors="pt", padding=True).to(self.device)
            translated = en_to_fr['model'].generate(**inputs, max_length=512)
            fr_text = en_to_fr['tokenizer'].decode(translated[0], skip_special_tokens=True)

            # Translate back to English
            fr_to_en = self.models['fr_to_en']
            inputs = fr_to_en['tokenizer'](fr_text, return_tensors="pt", padding=True).to(self.device)
            back_translated = fr_to_en['model'].generate(**inputs, max_length=512)
            en_text = fr_to_en['tokenizer'].decode(back_translated[0], skip_special_tokens=True)

            return en_text

        except Exception as e:
            logger.error(f"Back-translation error: {e}")
            return text

    def paraphrase_with_t5(self, text: str, num_variants: int = 1) -> List[str]:
        """
        Paraphrase text using T5 model.

        Args:
            text: Original text
            num_variants: Number of paraphrases to generate

        Returns:
            List of paraphrased texts
        """
        try:
            t5 = self.models['t5']

            # T5 expects task prefix
            input_text = f"paraphrase: {text}"
            inputs = t5['tokenizer'](
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            # Generate multiple paraphrases with different sampling
            outputs = t5['model'].generate(
                **inputs,
                max_length=512,
                num_return_sequences=num_variants,
                num_beams=num_variants * 2,
                temperature=1.2,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

            paraphrases = [
                t5['tokenizer'].decode(output, skip_special_tokens=True)
                for output in outputs
            ]

            return paraphrases

        except Exception as e:
            logger.error(f"T5 paraphrase error: {e}")
            return [text]

    def check_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Check semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        embeddings = self.similarity_model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return similarity

    def generate_variants(
        self,
        prompt: str,
        num_variants: int = 10,
        min_similarity: float = 0.85
    ) -> List[Dict[str, any]]:
        """
        Generate paraphrased variants of a prompt.

        Args:
            prompt: Original prompt
            num_variants: Number of variants to generate
            min_similarity: Minimum semantic similarity threshold

        Returns:
            List of variant dictionaries with text and similarity scores
        """
        variants = []

        # Always include the original
        variants.append({
            'variant_id': 0,
            'text': prompt,
            'similarity': 1.0,
            'method': 'original'
        })

        variant_id = 1

        # Back-translation variants
        if self.method in ["backtranslation", "both"]:
            for _ in range(num_variants // 2):
                paraphrase = self.backtranslate(prompt)
                similarity = self.check_semantic_similarity(prompt, paraphrase)

                if similarity >= min_similarity and paraphrase not in [v['text'] for v in variants]:
                    variants.append({
                        'variant_id': variant_id,
                        'text': paraphrase,
                        'similarity': similarity,
                        'method': 'backtranslation'
                    })
                    variant_id += 1

        # T5 paraphrasing variants
        if self.method in ["t5", "both"]:
            t5_paraphrases = self.paraphrase_with_t5(prompt, num_variants=num_variants // 2)

            for paraphrase in t5_paraphrases:
                similarity = self.check_semantic_similarity(prompt, paraphrase)

                if similarity >= min_similarity and paraphrase not in [v['text'] for v in variants]:
                    variants.append({
                        'variant_id': variant_id,
                        'text': paraphrase,
                        'similarity': similarity,
                        'method': 't5'
                    })
                    variant_id += 1

        # Ensure we have enough variants
        while len(variants) < num_variants + 1:  # +1 for original
            # Try more back-translations with different seeds
            paraphrase = self.backtranslate(prompt)
            similarity = self.check_semantic_similarity(prompt, paraphrase)

            if paraphrase not in [v['text'] for v in variants]:
                variants.append({
                    'variant_id': variant_id,
                    'text': paraphrase,
                    'similarity': similarity,
                    'method': 'backtranslation_extra'
                })
                variant_id += 1

        return variants[:num_variants + 1]  # +1 to include original


def process_seed_prompts(
    seed_file: str,
    output_dir: str,
    num_variants: int = 10,
    paraphrase_method: str = "backtranslation",
    min_similarity: float = 0.85
):
    """
    Process seed prompts and generate variants.

    Args:
        seed_file: Path to seed prompts CSV
        output_dir: Output directory for variants
        num_variants: Number of variants per seed
        paraphrase_method: Paraphrasing method
        min_similarity: Minimum semantic similarity
    """
    # Load seed prompts
    logger.info(f"Loading seed prompts from {seed_file}")
    seeds_df = pd.read_csv(seed_file)

    # Initialize paraphraser
    paraphraser = PromptParaphraser(paraphrase_method=paraphrase_method)

    # Generate variants for each seed
    all_variants = []

    for idx, row in tqdm(seeds_df.iterrows(), total=len(seeds_df), desc="Generating variants"):
        family_id = row.get('family_id', f'family_{idx:03d}')
        original_prompt = row['question']

        logger.info(f"\nProcessing {family_id}: {original_prompt[:50]}...")

        # Generate variants
        variants = paraphraser.generate_variants(
            prompt=original_prompt,
            num_variants=num_variants,
            min_similarity=min_similarity
        )

        # Add metadata
        for variant in variants:
            all_variants.append({
                'family_id': family_id,
                'variant_id': variant['variant_id'],
                'prompt_text': variant['text'],
                'similarity_to_original': variant['similarity'],
                'generation_method': variant['method'],
                'original_prompt': original_prompt,
                'dataset': row.get('dataset', 'unknown'),
                'topic': row.get('topic', 'unknown')
            })

    # Save variants
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    variants_df = pd.DataFrame(all_variants)
    csv_path = output_dir / "prompt_variants.csv"
    variants_df.to_csv(csv_path, index=False)
    logger.info(f"\nVariants saved to {csv_path}")

    # Save as JSON for easy loading
    json_path = output_dir / "prompt_variants.json"
    with open(json_path, 'w') as f:
        json.dump(all_variants, f, indent=2)
    logger.info(f"Variants saved to {json_path}")

    # Generate summary
    logger.info("\n" + "="*60)
    logger.info("VARIANT GENERATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total seed prompts: {len(seeds_df)}")
    logger.info(f"Variants per seed: {num_variants}")
    logger.info(f"Total variants: {len(variants_df)}")
    logger.info(f"Average similarity: {variants_df['similarity_to_original'].mean():.3f}")
    logger.info(f"Methods used: {variants_df['generation_method'].unique()}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate paraphrased prompt variants"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/seed_prompts.csv',
        help='Path to seed prompts CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/prompts',
        help='Output directory for variants'
    )
    parser.add_argument(
        '--variants',
        type=int,
        default=10,
        help='Number of variants per seed'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='backtranslation',
        choices=['backtranslation', 't5', 'both'],
        help='Paraphrasing method'
    )
    parser.add_argument(
        '--min-similarity',
        type=float,
        default=0.85,
        help='Minimum semantic similarity threshold'
    )

    args = parser.parse_args()

    # Process seed prompts
    logger.info("Starting prompt variant generation...")
    process_seed_prompts(
        seed_file=args.input,
        output_dir=args.output,
        num_variants=args.variants,
        paraphrase_method=args.method,
        min_similarity=args.min_similarity
    )

    logger.info("\n✓ Prompt variants generated successfully!")


if __name__ == "__main__":
    main()
