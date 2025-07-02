import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import os


from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get("HUGGINGFACE_TOKEN")
login(hf_token)

dataset_path = ""
dataset = load_dataset("json",data_files=dataset_path,split="train")

def format_prompt(example):
  prompt = f"""### INSTRUCTION:
Explain the causal link between the premise and the conclusion with a clear, step-by-step chain of reasoning.

### PREMISE:
{example['premise']}

### CONCLUSION:
{example['conclusion']}

### RESPONSE:
{' -> '.join(example['human_like_reasoning_chain'])}"""
  return {"text":prompt}

dataset = dataset.map(format_prompt)


base_model = "mistralai/Mistral-7B-Instruct-v0.2"

#configure quantization 
bnb_congig = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(base_model,quantization_config=bnb_congig,device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(base_model)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(r=16,lora_alpha=32,target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],lora_dropout=0.05,bias="none",task_type="CAUSAL_LM")


new_model_name = "mistral-7b-causal-think-bench"

training_args = TrainingArguments(output_dir=new_model_name,per_device_train_batch_size=4,gradient_accumulation_steps=4,
                                  learning_rate=2e-4,num_train_epochs=3, 
    logging_steps=10, 
    save_steps=50, 
    fp16=True, 
     )

trainer = SFTTrainer(model=model,train_dataset=dataset,peft_config=lora_config,dataset_text_field="text",max_seq_length=1024,tokenizer=tokenizer,args=training_args)

trainer.train()
