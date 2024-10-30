import json
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# Initialize translation models and tokenizers
src_lang = "en"
tgt_lang = "fr"
model_name_en_to_fr = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
model_name_fr_to_en = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"

tokenizer_en_to_fr = MarianTokenizer.from_pretrained(model_name_en_to_fr)
model_en_to_fr = MarianMTModel.from_pretrained(model_name_en_to_fr)

tokenizer_fr_to_en = MarianTokenizer.from_pretrained(model_name_fr_to_en)
model_fr_to_en = MarianMTModel.from_pretrained(model_name_fr_to_en)

# Function to translate text
def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the input JSONL file
input_file = "demo_data.jsonl"
output_file = "augmented_data.jsonl"

# Read JSONL into a DataFrame
data = pd.read_json(input_file, lines=True)

# Perform back-translation for each user message in the conversations
for i, conversation in tqdm(data.iterrows(), total=data.shape[0]):
    for message in conversation['conversations']:
        if message['role'] == 'user':
            original_text = message['content']
            # Perform English -> French -> English back-translation
            fr_text = translate(original_text, tokenizer_en_to_fr, model_en_to_fr)
            back_translated_text = translate(fr_text, tokenizer_fr_to_en, model_fr_to_en)
            message['content'] = back_translated_text

# Save the augmented data back to a JSONL file
with open(output_file, 'w') as f:
    for record in data.to_dict(orient='records'):
        f.write(json.dumps(record) + '\n')

print(f"Augmented data saved to {output_file}")
