"""
Generate Evaluation Queries from ArXiv Subset Dataset

This script automatically generates evaluation queries by:
1. Loading arxiv_subset.csv
2. Defining query topics with associated keywords
3. Finding papers that match those keywords in title/abstract
4. Saving queries with relevant paper indices to evaluation_queries.json

Usage:
    python evaluation/generate_queries.py
"""

import json
import pandas as pd
import re
from typing import List, Dict, Set


# Query topics with associated keywords for matching
QUERY_TOPICS = [
    {
        "query": "transformer attention mechanism neural networks",
        "keywords": ["transformer", "attention", "self-attention", "multi-head"]
    },
    {
        "query": "BERT pre-trained language models",
        "keywords": ["BERT", "pre-trained", "bidirectional", "language model"]
    },
    {
        "query": "graph neural networks GNN message passing",
        "keywords": ["graph neural", "GNN", "message passing", "graph convolution"]
    },
    {
        "query": "knowledge graph embedding representation learning",
        "keywords": ["knowledge graph", "embedding", "representation learning", "link prediction"]
    },
    {
        "query": "reinforcement learning deep Q-learning policy gradient",
        "keywords": ["reinforcement learning", "Q-learning", "policy gradient", "reward"]
    },
    {
        "query": "collaborative filtering recommendation algorithms",
        "keywords": ["collaborative filtering", "recommendation", "user-item", "matrix factorization"]
    },
    {
        "query": "large language models GPT generative pre-training",
        "keywords": ["large language model", "GPT", "generative pre-training", "LLM"]
    },
    {
        "query": "semantic search sentence embeddings similarity",
        "keywords": ["semantic search", "sentence embedding", "similarity", "semantic similarity"]
    },
    {
        "query": "computer vision convolutional neural networks CNN",
        "keywords": ["computer vision", "CNN", "convolutional neural", "image classification"]
    }
]


def load_dataset(data_path: str = "arxiv_subset.csv") -> pd.DataFrame:
    """Load the arXiv subset dataset."""
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df)} papers")
    return df


def find_relevant_papers(df: pd.DataFrame, keywords: List[str], max_papers: int = 50) -> List[int]:
    """
    Find papers relevant to given keywords.

    Searches for keywords in title and abstract fields.

    Args:
        df: DataFrame with paper metadata.
        keywords: List of keywords to search for.
        max_papers: Maximum number of relevant papers to return.

    Returns:
        List of row indices for relevant papers.
    """
    # Normalize keywords to lowercase
    keywords_lower = [kw.lower() for kw in keywords]

    relevant_indices = []

    for idx, row in df.iterrows():
        # Combine title and abstract for searching
        title = str(row.get("title", "")).lower() if pd.notna(row.get("title")) else ""
        abstract = str(row.get("abstract", "")).lower() if pd.notna(row.get("abstract")) else ""
        text = f"{title} {abstract}"

        # Check if any keyword appears in the text
        match_found = False
        for keyword in keywords_lower:
            if keyword in text:
                match_found = True
                break

        if match_found:
            relevant_indices.append(int(idx))

        # Stop if we have enough papers
        if len(relevant_indices) >= max_papers:
            break

    return relevant_indices


def generate_evaluation_queries(
    df: pd.DataFrame,
    max_papers_per_query: int = 50
) -> List[Dict]:
    """
    Generate evaluation queries with relevant papers.

    Args:
        df: DataFrame with paper metadata.
        max_papers_per_query: Maximum relevant papers per query.

    Returns:
        List of query dictionaries with relevant_papers indices.
    """
    queries = []

    for i, topic in enumerate(QUERY_TOPICS, 1):
        query_text = topic["query"]
        keywords = topic["keywords"]

        print(f"\n[{i}/{len(QUERY_TOPICS)}] Processing: {query_text}")
        print(f"  Keywords: {keywords}")

        # Find relevant papers
        relevant_indices = find_relevant_papers(df, keywords, max_papers_per_query)

        print(f"  Found {len(relevant_indices)} relevant papers")

        # Create query entry
        query_entry = {
            "query": query_text,
            "keywords": keywords,
            "relevant_papers": relevant_indices,
            "num_relevant": len(relevant_indices)
        }

        queries.append(query_entry)

    return queries


def save_queries(queries: List[Dict], output_path: str = "evaluation/evaluation_queries.json") -> None:
    """Save generated queries to JSON file."""
    with open(output_path, "w") as f:
        json.dump(queries, f, indent=4)
    print(f"\nSaved {len(queries)} queries to: {output_path}")


def print_summary(queries: List[Dict]) -> None:
    """Print summary statistics about generated queries."""
    print("\n" + "=" * 60)
    print("QUERY GENERATION SUMMARY")
    print("=" * 60)

    total_relevant = sum(q["num_relevant"] for q in queries)
    avg_relevant = total_relevant / len(queries) if queries else 0

    min_relevant = min(q["num_relevant"] for q in queries) if queries else 0
    max_relevant = max(q["num_relevant"] for q in queries) if queries else 0

    print(f"Total queries: {len(queries)}")
    print(f"Total relevant papers: {total_relevant}")
    print(f"Average relevant papers per query: {avg_relevant:.1f}")
    print(f"Min relevant papers: {min_relevant}")
    print(f"Max relevant papers: {max_relevant}")
    print("=" * 60)


def main():
    """Main function to generate evaluation queries."""
    print("=" * 60)
    print("ArXiv Evaluation Query Generator")
    print("=" * 60)

    # Load dataset
    df = load_dataset()

    # Generate queries
    queries = generate_evaluation_queries(df, max_papers_per_query=50)

    # Print summary
    print_summary(queries)

    # Save queries
    save_queries(queries)

    print("\nQuery generation complete!")
    print("Run evaluation with: python evaluation/run_experiments.py")


if __name__ == "__main__":
    main()
