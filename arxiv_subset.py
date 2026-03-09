import pandas as pd

df = pd.read_csv("arxiv_metadata.csv")

subset = df.sample(n=50000, random_state=42)

subset.to_csv("arxiv_subset.csv", index=False)
