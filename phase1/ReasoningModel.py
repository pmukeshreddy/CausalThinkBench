import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Any
import numpy as np



class ReasoningModel(nn.Module):
    def __init__(self, model_name: str, hidden_size: int,):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, num_classes)

        if self.tokenizer is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = self.config.hidden_size

		#specilased human like reasoining or not
        self.confidence_head = nn.Sequential(
        									nn.Linear(self.hidden_size, 512),
        									nn.ReLU(),
        									nn.Dropout(0.1),
											nn.Linear(512, 1)
        									)
        # emotional state predication
       	self.emotion_head = nn.Sequential(
       						nn.Linear(self.hidden_size, 512)
       						, nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 6), nn.Softmax(dim=1))
        # curious, cautious, confident, creative, analytical, empathetic
        # reasoining mode selector
        self.mode_selector = nn.Sequential(
        						nn.Linear(self.hidden_size, 256),
        						nn.ReLU(), nn.Linear(256, 4), nn.Softmax(dim=1))
        # analytical, creative, social, intuitive
        self.memory_relavance = nn.Sequential(
        							nn.Linear(self.hidden_size * 2, 256), nn.ReLU(),
        							 nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, input_ids, attention_mask=None, memory_context=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)

        if attention_mask is not None:
            masked_hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
			pooled_output = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

		if memory_context is not None:
            combined = torch.cat((pooled_output, memory_context), dim=-1)
            memory_score = self.memory_relavance(combined)
        return {
                    'base_outputs': outputs,
                    'pooled_representation': pooled_output,
                    'confidence': confidence,
                    'emotions': emotions,
                    'reasoning_mode': reasoning_mode,
                    'memory_relevance': memory_scores
                }
    def generate_response(self, prompt, max_length, temperature,do_sample):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            reasoining_outputs = self.forward(inputs.input_ids, inputs.attention_mask)

            generated_response = self.model.generate(inputs.input_ids,
            										attention_mask=inputs.attention_mask, max_length=max_length, temperature=temperature, do_sample=do_sample,pad_token_id=self.tokenizer.pad_token_id,


                                                                                                                                                                              eos_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(generated_response[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

                        return {
                            'response': response,
                            'confidence': reasoning_outputs['confidence'].item(),
                            'emotions': reasoning_outputs['emotions'].squeeze().tolist(),
                            'reasoning_mode': reasoning_outputs['reasoning_mode'].squeeze().tolist()
                        }

	def get_emotion_labels(self):
            """Return emotion labels for interpretation"""
            return ['curious', 'cautious', 'confident', 'creative', 'analytical', 'empathetic']
        
    def get_mode_labels(self):
            """Return reasoning mode labels"""
            return ['analytical', 'creative', 'social', 'intuitive'] 
