

# -*- coding: utf-8 -*-
"""
Causal Reasoning Evaluation Pipeline for Language Models.

This script evaluates a language model's ability to generate step-by-step reasoning
for Natural Language Inference (NLI) tasks. It loads a model, processes a benchmark
dataset (SNLI), generates explanations, and calculates performance metrics.

---
Prerequisites:
You must install the required libraries before running this script. It is recommended
to use a virtual environment.

pip install -q "transformers==4.40.2"
pip install -q "accelerate==0.29.3"
pip install -q "peft==0.10.0"
pip install -q "torch==2.3.0"
pip install -q "datasets==2.19.0" "evaluate==0.4.2" "pandas"
pip install -q "bert_score" "sentencepiece" "rouge_score"
pip install -q "optimum==1.19.1" "auto-gptq==0.7.1" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/

---
Security Note:
This script requires a Hugging Face authentication token. Set it as an
environment variable named 'HF_TOKEN' rather than hardcoding it.
Example (Linux/macOS):
    export HF_TOKEN='your_token_here'
Example (Windows PowerShell):
    $env:HF_TOKEN='your_token_here'

---
Usage:
python pipeline.py
"""

import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

# --- Configuration ---
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
BENCHMARK_NAME = "snli"
EVAL_SAMPLES = 100  # Number of samples to use for evaluation


def setup_and_load_assets(model_name: str, benchmark_name: str) -> tuple:
    """
    Handles authentication, and loads the model, tokenizer, and dataset.
    
    Args:
        model_name (str): The name of the model on Hugging Face Hub.
        benchmark_name (str): The name of the dataset on Hugging Face Hub.

    Returns:
        tuple: A tuple containing the model, tokenizer, and evaluation subset.
    """
    # --- Authentication ---
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Please set the HF_TOKEN environment variable."
        )
    login(token=hf_token)
    print("Login to Hugging Face successful.")

    # --- Load Model and Tokenizer ---
    print(f"Loading baseline model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Baseline model and tokenizer loaded.")

    # --- Prepare Benchmark Dataset ---
    print(f"Loading benchmark dataset: {benchmark_name}...")
    dataset = load_dataset(benchmark_name, split="test")
    dataset_filtered = dataset.filter(lambda example: example['label'] != -1)
    
    print(f"Original size of SNLI test set: {len(dataset)}")
    print(f"Size after filtering ambiguous examples: {len(dataset_filtered)}")

    eval_subset = dataset_filtered.shuffle(seed=42).select(range(EVAL_SAMPLES))
    print(f"Using a subset of {len(eval_subset)} examples for this run.")
    print("Dataset prepared.")
    
    return model, tokenizer, eval_subset


def run_evaluation(model, tokenizer, eval_subset, device: str) -> pd.DataFrame:
    """
    Runs the evaluation loop, generating explanations for each example.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        eval_subset: The dataset subset to evaluate on.
        device (str): The device to run inference on ('cuda' or 'cpu').

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """
    prompt_template = """### INSTRUCTION:
Explain the step-by-step reasoning that connects the following premise to the hypothesis.

### PREMISE:
{premise}

### CONCLUSION:
{hypothesis}

### RESPONSE:
"""
    print(f"\nRunning evaluation loop on {len(eval_subset)} examples...")
    results_list = []
    model.eval()
    
    with torch.no_grad():
        for example in tqdm(eval_subset, desc="Generating Explanations"):
            prompt = prompt_template.format(premise=example['premise'], hypothesis=example['hypothesis'])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=128, do_sample=False)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                explanation = generated_text.split("### RESPONSE:")[1].strip()
            except IndexError:
                explanation = "Model did not generate a valid response."

            results_list.append({
                "premise": example['premise'],
                "hypothesis": example['hypothesis'],
                "true_label_id": example['label'],
                "generated_explanation": explanation
            })

    print("Evaluation complete. Results collected.")
    return pd.DataFrame(results_list)


def analyze_results(results_df: pd.DataFrame, device: str):
    """
    Performs qualitative and quantitative analysis of the results.

    Args:
        results_df (pd.DataFrame): The DataFrame with evaluation results.
        device (str): The device for metric calculation ('cuda' or 'cpu').
    """
    print("\n--- Analyzing Results ---")
    
    # --- Qualitative Review ---
    print("\nQualitative Review of Baseline Model's Reasoning (Top 10 Results):")
    labels_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    results_df['true_label'] = results_df['true_label_id'].map(labels_map)
    pd.set_option('display.max_colwidth', 400)
    print(results_df.head(10)[['premise', 'hypothesis', 'true_label', 'generated_explanation']].to_string())

    # --- Automated Metrics ---
    print("\n\n--- Automated Metrics on 'Entailment' Examples ---")
    entailment_df = results_df[results_df['true_label'] == 'Entailment']

    if not entailment_df.empty:
        predictions = entailment_df['generated_explanation'].tolist()
        references = entailment_df['hypothesis'].tolist()

        # 1. ROUGE-2 Calculation
        print("Calculating ROUGE scores...")
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        rouge2_scores = [scorer.score(ref, pred)['rouge2'].fmeasure for ref, pred in zip(references, predictions)]
        avg_rouge2 = np.mean(rouge2_scores)
        
        # 2. BERTScore Calculation
        print("Calculating BERTScore... (This may take a moment)")
        P, R, F1 = bert_score_calc(predictions, references, lang="en", device=device, verbose=False)
        avg_bert_f1 = F1.mean().item()

        print(f"\nBERTScore F1 (Semantic Similarity): {avg_bert_f1:.4f}")
        print(f"ROUGE-2 Score (Lexical Overlap):   {avg_rouge2:.4f}")
    else:
        print("No 'Entailment' examples were present in the random subset to calculate scores.")


def main():
    """Main function to orchestrate the evaluation pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Cell 1 & 2: Setup and loading
        model, tokenizer, eval_subset = setup_and_load_assets(MODEL_NAME, BENCHMARK_NAME)
        
        # Cell 3: Run evaluation
        results_df = run_evaluation(model, tokenizer, eval_subset, device)
        
        # Cell 4: Analyze results
        analyze_results(results_df, device)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    finally:
        print("\nPipeline execution finished.")


if __name__ == "__main__":
    main()
