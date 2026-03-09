import pandas as pd
import json

# Load the JSON data
with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Convert JSON data to a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('arxiv_metadata.csv', index=False)
