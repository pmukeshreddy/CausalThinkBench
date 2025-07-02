import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import os
from trl import SFTConfig
from transformers import DataCollatorForLanguageModeling

from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get("HUGGINGFACE_TOKEN")
login(hf_token)

# Load and split dataset
dataset_path = "/content/causal_think_bench_dataset_corrected.jsonl"
full_dataset = load_dataset("json", data_files=dataset_path, split="train")

# Split dataset: 80% train, 20% test
train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)

def format_prompt(example):
  prompt = f"""### INSTRUCTION:
Explain the causal link between the premise and the conclusion with a clear, step-by-step chain of reasoning.

### PREMISE:
{example['premise']}

### CONCLUSION:
{example['conclusion']}

### RESPONSE:
{' -> '.join(example['human_like_reasoning_chain'])}"""
  return {"text": prompt}

# Apply formatting to both training and validation data
train_dataset = train_test_split['train'].map(format_prompt)
eval_dataset = train_test_split['test'].map(format_prompt)  # Format eval data too
test_dataset = train_test_split['test']  # Keep original format for final testing

base_model = "mistralai/Mistral-7B-Instruct-v0.2"

# Configure quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto", attn_implementation="eager")

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

new_model_name = "mistral-7b-causal-think-bench"

sft_config = SFTConfig(
    output_dir=new_model_name,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=4,
    logging_steps=10,
    save_steps=50,
    eval_steps=10,  # Evaluate every 10 steps
    evaluation_strategy="steps",  # Evaluate during training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    bf16=True,
    max_seq_length=1024,
    packing=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add validation dataset
    peft_config=lora_config,
    args=sft_config,
    data_collator=data_collator,
    formatting_func=lambda x: x["text"],
)

# Train the model
trainer.train()

# Test the model
def evaluate_model(model, tokenizer, test_examples, num_samples=3):
    model.eval()
    results = []
    
    for i, example in enumerate(test_examples.select(range(num_samples))):
        test_prompt = f"""### INSTRUCTION:
Explain the causal link between the premise and the conclusion with a clear, step-by-step chain of reasoning.

### PREMISE:
{example['premise']}

### CONCLUSION:
{example['conclusion']}

### RESPONSE:
"""
        
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # Add autocast
                outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated.split("### RESPONSE:")[-1].strip()
        
        print(f"\n--- Test {i+1} ---")
        print(f"Premise: {example['premise']}")
        print(f"Conclusion: {example['conclusion']}")
        print(f"Expected: {' -> '.join(example['human_like_reasoning_chain'])}")
        print(f"Generated: {response}")
        print("-" * 50)
    
    return results

# Run evaluation
print("Testing the fine-tuned model:")
test_results = evaluate_model(model, tokenizer, test_dataset)
