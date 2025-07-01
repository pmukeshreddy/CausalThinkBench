import google.generativeai as genai
from google.colab import userdata
import os
import json
import random
import time


api_key = "***************"
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro',generation_config=GenerationConfig(response_mime_type="application/json"))
categories = [
    "Ecological/Biological",
    "Economic",
    "Sociological/Historical",
    "Physical/Mechanical"
]
prompt = """
You are an expert curriculum designer and a master of causal reasoning. Your task is to generate a high-quality training example for a smaller language model. The example must be a causal puzzle within a specific category.

The output must be a single, clean JSON object.

The JSON object must have the following structure:
{{
  "category": "The category of the puzzle.",
  "premise": "A starting event or condition.",
  "conclusion": "A distant, non-obvious outcome caused by the premise.",
  "human_like_reasoning_chain": [
    "A first logical step.",
    "A second logical step that follows from the first.",
    "A third logical step...",
    "The final step that connects to the conclusion."
  ]
}}

Please adhere to these rules for the content:
1.  **Causal Chain Quality:** The `human_like_reasoning_chain` is the most important part. It must be a plausible, step-by-step logical path from the premise to the conclusion. Avoid giant logical leaps. Each step should naturally cause the next.
2.  **Human-Like Tone:** The reasoning should be clear and easy to understand, as if a knowledgeable and patient person were explaining it. Avoid overly technical jargon or robotic language.
3.  **Originality:** The puzzle should be creative and non-trivial.

Now, generate one such JSON object for the following category: {category}
"""


def genrate_and_save_data(num_example,output_file,prompt_template):
  with open(output_file,"a",encoding="utf-8") as f:
    for i in range(num_example):
      try:
        selected_category = random.choice(categories)
        print("a")
        formatted_prompt = prompt_template.format(category=selected_category)
        print("b")
        response = model.generate_content(formatted_prompt)
        print("c")
        json_string = response.text
        json.loads(json_string)
        f.write(json_string="\n")
        print(f"Successfully generated and saved example {i + 1}/{num_example}")
      except Exception as e:
        print(f"Error generating example {i + 1}: {e}")
        print(response)
        print("Skipping this example.")
      time.sleep(2) 
  print(f"\nData generation complete. Data saved to '{output_file}'.")

