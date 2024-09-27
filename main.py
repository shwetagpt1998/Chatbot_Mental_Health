import pandas as pd
import json

# Load the CSV file
csv_file = 'Merged_Conversation.csv'
df_csv = pd.read_csv(csv_file)

# Load the JSON file with specified encoding
json_file = 'intents.json'
with open(json_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract questions and answers from the JSON
new_data = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        new_data.append({'Questions': pattern, 'Answers': intent['responses'][0]})

# Create a DataFrame from the new data
df_json = pd.DataFrame(new_data)

# Merge the DataFrames
df_merged = pd.concat([df_csv, df_json], ignore_index=True)

# Save the merged DataFrame back to CSV
df_merged.to_csv(csv_file, index=False)
