import google.generativeai as genai
from google.colab import userdata
import os
import json
import random
import time


api_key = "****************"
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


import os
import json
import random
import time
from tqdm import tqdm
import concurrent.futures # Use the 'futures' library for threading

# The function is no longer async.
def generate_and_save_data(num_examples, output_filename, prompt):

  # This simple function makes one synchronous API call.
  # It will be executed in a separate thread for each request.
  def generate_one(p):
      try:
          response = model.generate_content(p)
          return response.text
      except Exception as e:
          return e

  # Create all the prompts we need ahead of time
  prompts = []
  for _ in range(num_examples):
        selected_category = random.choice(categories)
        formatted_prompt = prompt.format(category=selected_category)
        prompts.append(formatted_prompt)

  successful_generations = 0

  # Use ThreadPoolExecutor to run many synchronous calls at the same time.
  # max_workers determines how many run concurrently. 50 is a good start.
  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
      # Submit all tasks to the executor
      future_to_prompt = {executor.submit(generate_one, p): p for p in prompts}

      with open(output_filename, "a", encoding="utf-8") as f:
          # Use tqdm to create a progress bar as tasks complete
          for future in tqdm(concurrent.futures.as_completed(future_to_prompt), total=num_examples, desc="Generating Examples"):
              result = future.result()

              if isinstance(result, Exception):
                  continue

              try:
                  json_string = result
                  if "```json" in json_string:
                      json_string = json_string.split("```json\\n")[1].split("\\n```")[0]
                  json.loads(json_string)
                  f.write(json_string + "\\n")
                  successful_generations += 1
              except Exception as e:
                  pass # Skip malformed results

  print(f"\\nData generation complete. {successful_generations}/{num_examples} examples saved to '{output_filename}'.")

