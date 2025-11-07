"""
LLM Sampling Script
Runs LLaMA 3 inference on prompt variants and collects multiple outputs.

Phase 3: LLM Sampling
Author: Emmanuel Kwadwo Kusi
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLaMARunner:
    """Handles LLaMA 3 inference for robustness evaluation."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ):
        """
        Initialize LLaMA runner.

        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to run on (cuda/cpu)
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self._load_model(load_in_4bit, load_in_8bit)

    def _load_model(self, load_in_4bit: bool, load_in_8bit: bool):
        """Load LLaMA model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        kwargs = {}
        if load_in_4bit:
            logger.info("Using 4-bit quantization")
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif load_in_8bit:
            logger.info("Using 8-bit quantization")
            kwargs['load_in_8bit'] = True

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **kwargs
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully")

    def format_prompt(self, question: str, system_prompt: str = None) -> str:
        """
        Format prompt for LLaMA 3 Instruct.

        Args:
            question: User question/prompt
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = (
                "You are a financial AI assistant. Provide accurate, "
                "clear, and helpful responses to financial questions."
            )

        # LLaMA 3 Instruct format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def generate_single(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate a single response.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Generated text
        """
        formatted_prompt = self.format_prompt(prompt)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            response = response.replace("<|eot_id|>", "").strip()
        else:
            response = generated_text

        return response

    def sample_multiple(
        self,
        prompt: str,
        num_samples: int = 20,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate multiple samples for a single prompt.

        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            **generate_kwargs: Additional generation parameters

        Returns:
            List of generated responses
        """
        responses = []

        for i in range(num_samples):
            try:
                response = self.generate_single(prompt, **generate_kwargs)
                responses.append(response)

                if (i + 1) % 5 == 0:
                    logger.info(f"  Generated {i + 1}/{num_samples} samples")

            except Exception as e:
                logger.error(f"Error generating sample {i}: {e}")
                responses.append("")

        return responses

    def sample_batch(
        self,
        prompts: List[Dict],
        num_samples: int = 20,
        output_dir: str = "results/raw_outputs",
        **generate_kwargs
    ) -> pd.DataFrame:
        """
        Sample multiple outputs for a batch of prompts.

        Args:
            prompts: List of prompt dictionaries
            num_samples: Number of samples per prompt
            output_dir: Directory to save outputs
            **generate_kwargs: Generation parameters

        Returns:
            DataFrame with all outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_outputs = []

        for prompt_data in tqdm(prompts, desc="Processing prompts"):
            family_id = prompt_data.get('family_id', 'unknown')
            variant_id = prompt_data.get('variant_id', 0)
            prompt_text = prompt_data.get('prompt_text', prompt_data.get('question', ''))

            logger.info(f"\nSampling: {family_id} - Variant {variant_id}")
            logger.info(f"Prompt: {prompt_text[:80]}...")

            # Generate multiple samples
            start_time = time.time()
            responses = self.sample_multiple(
                prompt=prompt_text,
                num_samples=num_samples,
                **generate_kwargs
            )
            elapsed = time.time() - start_time

            logger.info(f"  Generated {len(responses)} samples in {elapsed:.2f}s")

            # Store results
            for sample_id, response in enumerate(responses):
                all_outputs.append({
                    'family_id': family_id,
                    'variant_id': variant_id,
                    'sample_id': sample_id,
                    'prompt_text': prompt_text,
                    'response': response,
                    'response_length': len(response),
                    'generation_time': elapsed / num_samples,
                    'model': self.model_name
                })

            # Save intermediate results
            if len(all_outputs) % 100 == 0:
                temp_df = pd.DataFrame(all_outputs)
                temp_path = output_dir / f"outputs_checkpoint_{len(all_outputs)}.csv"
                temp_df.to_csv(temp_path, index=False)
                logger.info(f"Checkpoint saved: {temp_path}")

        # Save final results
        outputs_df = pd.DataFrame(all_outputs)
        final_path = output_dir / "llama3_outputs.csv"
        outputs_df.to_csv(final_path, index=False)
        logger.info(f"\nAll outputs saved to {final_path}")

        # Save as JSON
        json_path = output_dir / "llama3_outputs.json"
        outputs_df.to_json(json_path, orient='records', indent=2)

        return outputs_df


class OllamaRunner:
    """Alternative: Run LLaMA 3 via Ollama CLI."""

    def __init__(self, model_name: str = "llama3"):
        """
        Initialize Ollama runner.

        Args:
            model_name: Ollama model name
        """
        self.model_name = model_name
        logger.info(f"Using Ollama model: {model_name}")

    def generate_single(
        self,
        prompt: str,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response via Ollama CLI.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        import subprocess

        cmd = [
            "ollama",
            "run",
            self.model_name,
            "--temperature", str(temperature),
            prompt
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout.strip()

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""

    def sample_multiple(
        self,
        prompt: str,
        num_samples: int = 20,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate multiple samples."""
        responses = []

        for i in range(num_samples):
            response = self.generate_single(prompt, temperature=temperature)
            responses.append(response)

            if (i + 1) % 5 == 0:
                logger.info(f"  Generated {i + 1}/{num_samples} samples")

        return responses


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run LLaMA 3 sampling on prompt variants"
    )
    parser.add_argument(
        '--prompts',
        type=str,
        default='data/prompts/prompt_variants.json',
        help='Path to prompt variants JSON'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='Model name or path'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=20,
        help='Number of samples per prompt'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/raw_outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--use-ollama',
        action='store_true',
        help='Use Ollama instead of HuggingFace'
    )
    parser.add_argument(
        '--4bit',
        action='store_true',
        help='Use 4-bit quantization'
    )
    parser.add_argument(
        '--8bit',
        action='store_true',
        help='Use 8-bit quantization'
    )

    args = parser.parse_args()

    # Load prompts
    logger.info(f"Loading prompts from {args.prompts}")
    with open(args.prompts, 'r') as f:
        prompts = json.load(f)

    logger.info(f"Loaded {len(prompts)} prompts")

    # Initialize runner
    if args.use_ollama:
        runner = OllamaRunner(model_name=args.model)
    else:
        runner = LLaMARunner(
            model_name=args.model,
            load_in_4bit=args.__dict__.get('4bit', False),
            load_in_8bit=args.__dict__.get('8bit', False)
        )

    # Run sampling
    logger.info("Starting LLaMA 3 sampling...")
    outputs = runner.sample_batch(
        prompts=prompts,
        num_samples=args.samples,
        output_dir=args.output,
        temperature=args.temperature,
        max_length=args.max_length
    )

    logger.info("\n✓ Sampling completed successfully!")
    logger.info(f"Total outputs: {len(outputs)}")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
