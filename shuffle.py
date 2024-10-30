import pandas as pd
import json

# Load the original and paraphrased datasets
original_file = "demo_data.jsonl"
paraphrased_file = "paraphrased.jsonl"

# Read both JSONL files into DataFrames
original_data = pd.read_json(original_file, lines=True)
paraphrased_data = pd.read_json(paraphrased_file, lines=True)

# Combine the datasets (duplicates allowed)
combined_data = pd.concat([original_data, paraphrased_data], ignore_index=True)

# Shuffle the combined dataset
shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled dataset back to a JSONL file
output_file = "shuffled.jsonl"

with open(output_file, 'w') as f:
    for record in shuffled_data.to_dict(orient='records'):
        f.write(json.dumps(record) + '\n')

print(f"Shuffled combined dataset saved to {output_file}")
