import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the pre-trained paraphrasing model and tokenizer
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to paraphrase text using T5
def paraphrase(text, num_return_sequences=1, max_length=512):
    input_text = f"paraphrase: {text} </s>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, padding=True, truncation=True)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=10,
        temperature=2.0,  # Makes the output more diverse
        top_k=120,
        top_p=0.95,
        do_sample=True
    )
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Load the input JSONL file
input_file = "demo_data.jsonl"
output_file = "paraphrased.jsonl"

# Helper function to filter out identical paraphrases
def get_best_paraphrase(original_text, paraphrases):
    for para in paraphrases:
        if para.strip().lower() != original_text.strip().lower():
            return para
    return original_text


# Read JSONL into a DataFrame
data = pd.read_json(input_file, lines=True)

# Apply paraphrasing to user messages
for i, conversation in tqdm(data.iterrows(), total=data.shape[0]):
    for message in conversation['conversations']:
        original_text = message['content']
        paraphrases = paraphrase(original_text, num_return_sequences=3)
        best_paraphrase = get_best_paraphrase(original_text, paraphrases)
        message['content'] = best_paraphrase

# Save the augmented data back to a JSONL file
with open(output_file, 'w') as f:
    for record in data.to_dict(orient='records'):
        f.write(json.dumps(record) + '\n')

print(f"Paraphrased data saved to {output_file}")
