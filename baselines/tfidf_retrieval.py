"""
TF-IDF Baseline Retrieval Model for ArXiv Paper Recommendation

This module implements a TF-IDF (Term Frequency-Inverse Document Frequency)
baseline model for retrieving relevant academic papers from ArXiv.

Pipeline:
    query -> TF-IDF vector -> cosine similarity -> top-k papers
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TfidfRetriever:
    """
    TF-IDF based paper retriever for ArXiv dataset.

    Attributes:
        data (pd.DataFrame): Loaded ArXiv metadata.
        vectorizer (TfidfVectorizer): Scikit-learn TF-IDF vectorizer.
        tfidf_matrix (csr_matrix): TF-IDF vectors for all papers.
        corpus (list): Combined text (title + abstract) for each paper.
    """

    def __init__(self, data_path: str = "arxiv_subset.csv"):
        """
        Initialize the TF-IDF retriever.

        Args:
            data_path: Path to the ArXiv metadata CSV file.
        """
        self.data_path = data_path
        self.data = None
        self.vectorizer = None
        self.tfidf_matrix = None
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

    def build_index(self) -> None:
        """
        Build TF-IDF vectors for all papers in the dataset.

        Creates a TF-IDF vectorizer and transforms the corpus into
        a sparse TF-IDF matrix.
        """
        if self.corpus is None:
            self.preprocess()

        # Initialize TF-IDF vectorizer
        # - min_df=2: ignore terms that appear in fewer than 2 documents
        # - max_df=0.95: ignore terms that appear in more than 95% of documents
        # - stop_words='english': remove common English stop words
        # - ngram_range=(1, 2): include unigrams and bigrams
        self.vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.95,
            stop_words="english",
            ngram_range=(1, 2)
        )

        # Compute TF-IDF vectors for all papers
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, k: int = 5) -> list:
        """
        Search for top-k most similar papers given a query.

        Args:
            query: The search query string.
            k: Number of top results to return (default: 5).

        Returns:
            list: List of tuples containing (paper_index, similarity_score).
        """
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between query and all papers
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

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


def search_tfidf(query: str, k: int = 5, data_path: str = "arxiv_subset.csv") -> pd.DataFrame:
    """
    Convenience function to search for top-k papers using TF-IDF.

    This function loads the data, builds the index, and performs the search
    in a single call. For repeated searches, use the TfidfRetriever class
    directly to avoid rebuilding the index.

    Args:
        query: The search query string.
        k: Number of top results to return (default: 5).
        data_path: Path to the ArXiv metadata CSV file.

    Returns:
        pd.DataFrame: DataFrame with top-k paper recommendations.
    """
    retriever = TfidfRetriever(data_path)
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
    # Example usage demonstrating the TF-IDF retrieval pipeline
    print("TF-IDF Baseline Retrieval for ArXiv Papers")
    print("=" * 50)

    # Initialize retriever
    retriever = TfidfRetriever()

    # Load data
    print("\n[1/3] Loading dataset...")
    retriever.load_data()
    print(f"      Loaded {len(retriever.data)} papers")

    # Build index
    print("\n[2/3] Building TF-IDF index...")
    retriever.build_index()
    print(f"      Index shape: {retriever.tfidf_matrix.shape}")

    # Example queries
    test_queries = [
        "deep learning neural networks",
        "natural language processing transformers",
        "computer vision image classification"
    ]

    print("\n[3/3] Running example queries...")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retriever.search_with_details(query, k=3)
        print_results(results)
