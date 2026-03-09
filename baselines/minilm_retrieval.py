"""
MiniLM-L3 Baseline Retrieval Model for ArXiv Papers

This module implements a lightweight Sentence-BERT baseline model using
paraphrase-MiniLM-L3-v2 for retrieving relevant academic papers from ArXiv.

Pipeline:
    query -> MiniLM-L3 embedding -> cosine similarity -> top-k papers

Model:
    paraphrase-MiniLM-L3-v2
    - 384-dimensional embeddings
    - ~60M parameters
    - Fast inference, suitable for resource-constrained environments

Usage:
    # One-off search
    from minilm_retrieval import search_minilm
    results = search_minilm("deep learning", k=5)

    # Repeated searches (more efficient)
    from minilm_retrieval import MinilmRetriever
    retriever = MinilmRetriever()
    retriever.load_data()
    retriever.build_index()
    results = retriever.search_with_details("your query", k=5)

    # Command line
    python baselines/minilm_retrieval.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class MinilmRetriever:
    """
    MiniLM-L3 based paper retriever for ArXiv dataset.

    Attributes:
        data (pd.DataFrame): Loaded ArXiv metadata.
        model: SentenceTransformer MiniLM-L3-v2 model.
        embeddings (np.ndarray): MiniLM embeddings for all papers.
        corpus (list): Combined text (title + abstract) for each paper.
    """

    def __init__(self, data_path: str = "arxiv_subset.csv",
                 model_name: str = "paraphrase-MiniLM-L3-v2"):
        """
        Initialize the MiniLM retriever.

        Args:
            data_path: Path to the ArXiv metadata CSV file.
            model_name: Name of the SentenceTransformer model.
                       Default is 'paraphrase-MiniLM-L3-v2'.

        Alternative models:
            - paraphrase-MiniLM-L6-v2: Larger model, better accuracy
            - all-MiniLM-L12-v2: Even larger model
            - paraphrase-multilingual-MiniLM-L12-v2: Multilingual support
        """
        self.data_path = data_path
        self.model_name = model_name
        self.data = None
        self.model = None
        self.embeddings = None
        self.corpus = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the ArXiv metadata from CSV.

        Returns:
            pd.DataFrame: Loaded dataset with paper metadata.
        """
        self.data = pd.read_csv(self.data_path, low_memory=False)
        return self.data

    def preprocess(self) -> list:
        """
        Combine title and abstract fields into a single text column.

        Returns:
            list: List of combined text strings for each paper.
        """
        # Fill NaN values with empty strings
        titles = self.data["title"].fillna("")
        abstracts = self.data["abstract"].fillna("")

        # Combine title and abstract
        self.corpus = (titles + " " + abstracts).tolist()
        return self.corpus

    def load_model(self):
        """
        Load the MiniLM-L3 model from SentenceTransformers.

        The model is loaded once and cached for reuse.
        """
        if self.model is None:
            print(f"Loading MiniLM-L3 model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("MiniLM-L3 model loaded successfully.")

    def build_index(self) -> None:
        """
        Generate MiniLM-L3 embeddings for all papers in the dataset.

        Computes sentence embeddings for the entire corpus and stores
        them in memory for fast similarity search.
        """
        if self.corpus is None:
            self.preprocess()

        # Load model if not already loaded
        self.load_model()

        # Generate embeddings for all papers
        # Process in batches to avoid memory issues with large datasets
        print(f"Generating embeddings for {len(self.corpus)} papers...")

        batch_size = 1024
        all_embeddings = []

        for i in range(0, len(self.corpus), batch_size):
            batch = self.corpus[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)

            # Progress update
            processed = min(i + batch_size, len(self.corpus))
            print(f"  Processed {processed}/{len(self.corpus)} papers")

        # Concatenate all batch embeddings
        self.embeddings = np.vstack(all_embeddings).astype("float32")
        print(f"Embeddings shape: {self.embeddings.shape}")

    def search(self, query: str, k: int = 5) -> list:
        """
        Search for top-k most similar papers given a query.

        Args:
            query: The search query string.
            k: Number of top results to return (default: 5).

        Returns:
            list: List of tuples containing (paper_index, similarity_score).
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Generate embedding for query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        # Compute cosine similarity between query and all papers
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()

        # Get indices of top-k most similar papers
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Return list of (index, score) tuples
        results = [(idx, similarities[idx]) for idx in top_k_indices]
        return results

    def search_with_details(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Search for top-k papers and return detailed information.

        Args:
            query: The search query string.
            k: Number of top results to return (default: 5).

        Returns:
            pd.DataFrame: DataFrame with paper details and similarity scores.
        """
        results = self.search(query, k)

        # Create result DataFrame
        result_data = []
        for idx, score in results:
            paper = self.data.iloc[idx].copy()
            paper["similarity_score"] = score
            result_data.append(paper)

        return pd.DataFrame(result_data)

    def save_embeddings(self, output_path: str = "minilm_embeddings.npy") -> None:
        """
        Save computed embeddings to disk for later reuse.

        Args:
            output_path: Path to save the embeddings file.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Call build_index() first.")

        np.save(output_path, self.embeddings)
        print(f"Embeddings saved to: {output_path}")

    def load_embeddings(self, input_path: str = "minilm_embeddings.npy") -> None:
        """
        Load pre-computed embeddings from disk.

        Args:
            input_path: Path to the embeddings file.
        """
        self.embeddings = np.load(input_path)
        print(f"Embeddings loaded from: {input_path}")
        print(f"Embeddings shape: {self.embeddings.shape}")


def search_minilm(query: str, k: int = 5, data_path: str = "arxiv_subset.csv") -> pd.DataFrame:
    """
    Convenience function to search for top-k papers using MiniLM-L3.

    This function loads the data, builds the index, and performs the search
    in a single call. For repeated searches, use the MinilmRetriever class
    directly to avoid rebuilding the index.

    Args:
        query: The search query string.
        k: Number of top results to return (default: 5).
        data_path: Path to the ArXiv metadata CSV file.

    Returns:
        pd.DataFrame: DataFrame with top-k paper recommendations.
    """
    retriever = MinilmRetriever(data_path)
    retriever.load_data()
    retriever.build_index()
    return retriever.search_with_details(query, k)


def print_results(results: pd.DataFrame) -> None:
    """
    Print paper titles and similarity scores in a formatted manner.

    Args:
        results: DataFrame containing search results with similarity scores.
    """
    print("\n" + "=" * 80)
    print(f"Top {len(results)} Papers")
    print("=" * 80)

    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"\n{i}. {row['title']}")
        print(f"   Similarity Score: {row['similarity_score']:.4f}")
        print(f"   Categories: {row.get('categories', 'N/A')}")
        print(f"   Year: {row.get('year', 'N/A')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage demonstrating the MiniLM-L3 retrieval pipeline
    print("MiniLM-L3 Baseline Retrieval for ArXiv Papers")
    print("=" * 60)

    # Initialize retriever
    retriever = MinilmRetriever()

    # Load data
    print("\n[1/3] Loading dataset...")
    retriever.load_data()
    print(f"      Loaded {len(retriever.data)} papers")

    # Build index
    print("\n[2/3] Building MiniLM-L3 index (this may take a few minutes)...")
    retriever.build_index()

    # Example queries
    test_queries = [
        "deep learning neural networks",
        "natural language processing transformers",
        "computer vision image classification"
    ]

    print("\n[3/3] Running example queries...")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retriever.search_with_details(query, k=3)
        print_results(results)
