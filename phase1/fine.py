import torch
from datasets import load_dataset , concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import os
from trl import SFTConfig
from transformers import DataCollatorForLanguageModeling

from huggingface_hub import login
from google.colab import userdata

from transformers import EarlyStoppingCallback


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




combined_dataset = train_test_split['train'].map(
    create_all_examples,
    batched=True,
    remove_columns=['category', 'premise', 'conclusion', 'human_like_reasoning_chain']
)

base_model = "mistralai/Mistral-7B-Instruct-v0.2"

# Configure quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto", attn_implementation="eager")

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.3, bias="none", task_type="CAUSAL_LM")

new_model_name = "mistral-7b-causal-think-bench"

sft_config = SFTConfig(
    output_dir=new_model_name,
    per_device_train_batch_size=4,        # Back to 4 (was working)
    gradient_accumulation_steps=4,        # Back to 4
    learning_rate=1e-4,                   # Middle ground: 1e-4 (was 2e-4 originally)
    num_train_epochs=2,                   # Keep reduced epochs
    logging_steps=10,
    save_steps=50,
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
    train_dataset=combined_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=sft_config,
    data_collator=data_collator,
    formatting_func=lambda x: x["text"],
)

#trainer.add_callback(EarlyStoppingCallback(
   # early_stopping_patience=3,
    #early_stopping_threshold=0.01
#))

# Simple validation callback
from transformers import TrainerCallback

class ValidationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

    def on_log(self, args, state, control, model, **kwargs):
        if state.global_step % 10 == 0 and state.global_step > 0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for i in range(0, min(16, len(self.eval_dataset)), 4):
                    batch = self.eval_dataset.select(range(i, min(i+4, len(self.eval_dataset))))
                    texts = [ex["text"] for ex in batch]
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                          truncation=True, max_length=1024)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    outputs = model(**inputs, labels=inputs["input_ids"])
                    val_losses.append(outputs.loss.item())

            val_loss = sum(val_losses) / len(val_losses)
            print(f"Step {state.global_step}: Validation Loss = {val_loss:.6f}")
            model.train()

# Add callback to trainer
trainer.add_callback(ValidationCallback(eval_dataset, tokenizer))

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
