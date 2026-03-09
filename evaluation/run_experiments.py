"""
Experiment Runner for ArXiv Paper Recommendation Models

This script evaluates three retrieval models using pseudo relevance labeling:
1. TF-IDF (Term Frequency-Inverse Document Frequency)
2. MiniLM-L3 (paraphrase-MiniLM-L3-v2) - Lightweight semantic baseline
3. SBERT + FAISS (all-MiniLM-L6-v2 with FAISS index)

Pseudo Relevance Labeling Method:
    Since manual annotations are unavailable, we use SBERT to generate
    pseudo ground truth labels:
    1. Generate SBERT embedding for each query
    2. Compute cosine similarity with all paper embeddings
    3. Select top-20 most similar papers as "relevant"
    4. Evaluate all models against this ground truth

    This approach follows standard IR evaluation practice (Thakur et al., 2021).

Results are saved to evaluation/results.csv and printed to console.

Usage:
    python evaluation/run_experiments.py

    # With custom parameters
    python evaluation/run_experiments.py --k 10 --output evaluation/my_results.csv

    # Run specific models
    python evaluation/run_experiments.py --models tfidf minilm_l3
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Evaluation Metrics
# =============================================================================

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Compute Precision@K metric.

    Args:
        retrieved: List of retrieved paper indices (ordered by relevance).
        relevant: Set of relevant paper indices (ground truth).
        k: Number of top results to consider.

    Returns:
        Precision@K score (0.0 to 1.0).
    """
    if not retrieved or k == 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item in relevant)
    return relevant_count / k


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Compute Average Precision (AP) for a single query.

    Args:
        retrieved: List of retrieved paper indices (ordered by relevance).
        relevant: Set of relevant paper indices (ground truth).

    Returns:
        Average Precision score (0.0 to 1.0).
    """
    if not retrieved or not relevant:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, idx in enumerate(retrieved):
        if idx in relevant:
            relevant_count += 1
            precision_at_position = relevant_count / (i + 1)
            precision_sum += precision_at_position

    return precision_sum / len(relevant)


def mean_average_precision(
    queries_results: List[List[int]],
    queries_relevant: List[Set[int]]
) -> float:
    """
    Compute Mean Average Precision (MAP) across all queries.

    Args:
        queries_results: List of retrieved paper indices for each query.
        queries_relevant: List of relevant paper indices for each query.

    Returns:
        MAP score (0.0 to 1.0).
    """
    if not queries_results:
        return 0.0

    ap_scores = []
    for retrieved, relevant in zip(queries_results, queries_relevant):
        ap = average_precision(retrieved, relevant)
        ap_scores.append(ap)

    return np.mean(ap_scores)


# =============================================================================
# Pseudo Relevance Labeling
# =============================================================================

class PseudoRelevanceLabeler:
    """
    Generate pseudo ground truth relevance labels using SBERT.

    This is used when manual annotations are unavailable. The method
    uses a strong retrieval model (SBERT) to identify relevant papers
    for each query, which then serve as ground truth for evaluation.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k_relevant: int = 20
    ):
        """
        Initialize the labeler.

        Args:
            model_name: SBERT model for generating embeddings.
            top_k_relevant: Number of papers to mark as relevant per query.
        """
        self.model_name = model_name
        self.top_k_relevant = top_k_relevant
        self.model = None
        self.paper_embeddings = None

    def load_model(self):
        """Load SBERT model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading SBERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def compute_paper_embeddings(
        self,
        corpus: List[str],
        batch_size: int = 512
    ) -> np.ndarray:
        """
        Compute embeddings for all papers.

        Args:
            corpus: List of paper texts (title + abstract).
            batch_size: Batch size for embedding computation.

        Returns:
            Numpy array of shape (num_papers, embedding_dim).
        """
        self.load_model()

        print(f"Computing embeddings for {len(corpus)} papers...")
        self.paper_embeddings = self.model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )

        print(f"Embeddings shape: {self.paper_embeddings.shape}")
        return self.paper_embeddings

    def save_embeddings(self, path: str) -> None:
        """Save embeddings to disk."""
        if self.paper_embeddings is None:
            raise ValueError("No embeddings to save")
        np.save(path, self.paper_embeddings)
        print(f"Embeddings saved to: {path}")

    def load_embeddings(self, path: str) -> np.ndarray:
        """Load pre-computed embeddings from disk."""
        self.paper_embeddings = np.load(path)
        print(f"Embeddings loaded from: {path}")
        return self.paper_embeddings

    def generate_labels(
        self,
        queries: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Set[int]]]:
        """
        Generate pseudo relevance labels for all queries.

        For each query:
        1. Generate query embedding
        2. Compute cosine similarity with all papers
        3. Select top-k most similar as relevant (pseudo ground truth)

        Args:
            queries: List of query dictionaries with 'query' text.

        Returns:
            Tuple of (labeled_queries, list of relevant sets).
        """
        if self.paper_embeddings is None:
            raise ValueError("Must compute/load embeddings first")

        self.load_model()

        from sklearn.metrics.pairwise import cosine_similarity

        relevant_sets = []
        labeled_queries = []

        for i, q in enumerate(queries, 1):
            query_text = q["query"]

            # Generate query embedding
            query_embedding = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # Compute cosine similarity
            similarities = cosine_similarity(
                query_embedding,
                self.paper_embeddings
            ).flatten()

            # Get top-k relevant papers
            top_k_indices = np.argsort(similarities)[::-1][:self.top_k_relevant]
            relevant_set = set(top_k_indices.tolist())

            # Store
            labeled_query = q.copy()
            labeled_query["relevant_papers"] = top_k_indices.tolist()
            labeled_query["num_relevant"] = len(top_k_indices)
            labeled_queries.append(labeled_query)
            relevant_sets.append(relevant_set)

            # Progress
            if i % 10 == 0 or i == len(queries):
                print(f"  Labeled query {i}/{len(queries)}")

        return labeled_queries, relevant_sets


# =============================================================================
# Model Wrappers
# =============================================================================

class TfidfModel:
    """Wrapper for TF-IDF retrieval model."""

    def __init__(self, data_path: str = "arxiv_subset.csv"):
        from baselines.tfidf_retrieval import TfidfRetriever

        print("  Loading TF-IDF model...")
        self.retriever = TfidfRetriever(data_path)
        self.retriever.load_data()
        self.retriever.build_index()
        print("  TF-IDF model ready.")

    def search(self, query: str, k: int = 5) -> List[int]:
        """Return top-k paper row indices for a query."""
        results = self.retriever.search_with_details(query, k)
        retrieved_ids = results["id"].tolist()

        # Map IDs back to row indices
        indices = []
        for paper_id in retrieved_ids:
            mask = self.retriever.data["id"] == paper_id
            idx = self.retriever.data[mask].index.tolist()
            if idx:
                indices.append(idx[0])
        return indices


class MinilmModel:
    """Wrapper for MiniLM-L3 baseline retrieval model."""

    def __init__(self, data_path: str = "arxiv_subset.csv"):
        from baselines.minilm_retrieval import MinilmRetriever

        print("  Loading MiniLM-L3 model...")
        self.retriever = MinilmRetriever(data_path)
        self.retriever.load_data()
        self.retriever.build_index()
        print("  MiniLM-L3 model ready.")

    def search(self, query: str, k: int = 5) -> List[int]:
        """Return top-k paper row indices for a query."""
        results = self.retriever.search_with_details(query, k)
        retrieved_ids = results["id"].tolist()

        # Map IDs back to row indices
        indices = []
        for paper_id in retrieved_ids:
            mask = self.retriever.data["id"] == paper_id
            idx = self.retriever.data[mask].index.tolist()
            if idx:
                indices.append(idx[0])
        return indices


class SbertFaissModel:
    """Wrapper for SBERT + FAISS retrieval model."""

    def __init__(self, data_path: str = "arxiv_subset.csv"):
        import faiss
        from sentence_transformers import SentenceTransformer
        import pandas as pd

        self.faiss = faiss

        print("  Loading SBERT + FAISS model...")
        self.data_path = data_path
        self.data = pd.read_csv(data_path, low_memory=False)

        # Load SBERT model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load or build FAISS index
        self.faiss_index_file = "faiss_index.index"
        self._load_or_build_index()
        print("  SBERT + FAISS model ready.")

    def _load_or_build_index(self):
        """Load existing FAISS index or build new one."""
        if os.path.exists(self.faiss_index_file):
            self.index = self.faiss.read_index(self.faiss_index_file)
            print(f"    Loaded existing FAISS index: {self.index.ntotal} vectors")
        else:
            print("    Building FAISS index (this may take a while)...")
            corpus = [f"{title}. {abstract}"
                     for title, abstract in zip(self.data["title"], self.data["abstract"])]
            embeddings = self.model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)

            dimension = embeddings.shape[1]
            self.index = self.faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype("float32"))

            # Save index
            self.faiss.write_index(self.index, self.faiss_index_file)
            print(f"    FAISS index built and saved: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> List[int]:
        """Return top-k paper row indices for a query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        return indices[0].tolist()


# =============================================================================
# Experiment Runner
# =============================================================================

def load_queries(queries_path: str) -> List[Dict[str, Any]]:
    """Load evaluation queries from JSON file."""
    with open(queries_path, "r") as f:
        return json.load(f)


def create_corpus(data_path: str) -> List[str]:
    """Create corpus from dataset (title + abstract)."""
    import pandas as pd
    df = pd.read_csv(data_path, low_memory=False)
    titles = df["title"].fillna("")
    abstracts = df["abstract"].fillna("")
    return (titles + " " + abstracts).tolist()


def run_model_evaluation(
    model_name: str,
    model: Any,
    queries: List[Dict[str, Any]],
    relevant_sets: List[Set[int]],
    k: int
) -> Dict[str, float]:
    """
    Evaluate a single model on all queries.

    Args:
        model_name: Name of the model.
        model: Model instance with search(query, k) method.
        queries: List of query dictionaries.
        relevant_sets: List of relevant paper sets (pseudo ground truth).
        k: Number of top results to consider.

    Returns:
        Dictionary with precision@k and MAP scores.
    """
    print(f"\nEvaluating {model_name}...")
    print("-" * 50)

    all_retrieved = []
    precision_scores = []

    for i, (q, relevant) in enumerate(zip(queries, relevant_sets), 1):
        query_text = q["query"]

        # Get retrieved papers
        retrieved = model.search(query_text, k)

        # Compute precision@k
        p_at_k = precision_at_k(retrieved, relevant, k)
        precision_scores.append(p_at_k)

        all_retrieved.append(retrieved)

        # Progress
        if i % 10 == 0 or i == len(queries):
            print(f"  Query {i}/{len(queries)}: P@{k}={p_at_k:.4f}")

    # Compute metrics
    overall_precision = np.mean(precision_scores)
    map_score = mean_average_precision(all_retrieved, relevant_sets)

    print(f"\n{model_name} Results:")
    print(f"  Precision@{k}: {overall_precision:.4f}")
    print(f"  MAP:           {map_score:.4f}")

    return {
        "precision_at_k": overall_precision,
        "map": map_score
    }


def save_results_csv(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    k: int = 5
) -> None:
    """Save evaluation results to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", f"Precision@{k}", "MAP"])

        for model_name, metrics in results.items():
            writer.writerow([
                model_name,
                f"{metrics['precision_at_k']:.4f}",
                f"{metrics['map']:.4f}"
            ])

    print(f"\nResults saved to: {output_path}")


def print_results_table(results: Dict[str, Dict[str, float]], k: int = 5) -> None:
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT RESULTS (K={k})")
    print("=" * 60)
    print(f"{'Model':<25} | {'Precision@5':<12} | {'MAP':<12}")
    print("-" * 60)

    for model_name, metrics in results.items():
        precision = metrics["precision_at_k"]
        map_score = metrics["map"]
        print(f"{model_name:<25} | {precision:<12.4f} | {map_score:<12.4f}")

    print("=" * 60)

    # Find best models
    best_precision = max(results.keys(), key=lambda m: results[m]["precision_at_k"])
    best_map = max(results.keys(), key=lambda m: results[m]["map"])

    print(f"\nBest Precision@{k}: {best_precision} ({results[best_precision]['precision_at_k']:.4f})")
    print(f"Best MAP:           {best_map} ({results[best_map]['map']:.4f})")


def run_experiments(
    queries_path: str = "evaluation/evaluation_queries.json",
    data_path: str = "arxiv_subset.csv",
    output_path: str = "evaluation/results.csv",
    k: int = 5,
    models_to_run: List[str] = None,
    top_k_relevant: int = 20
) -> Dict[str, Dict[str, float]]:
    """
    Run full evaluation experiments with pseudo relevance labeling.

    Args:
        queries_path: Path to evaluation queries JSON.
        data_path: Path to ArXiv metadata CSV.
        output_path: Path to save results CSV.
        k: Number of top results to consider.
        models_to_run: Models to evaluate ['tfidf', 'minilm_l3', 'sbert_faiss'].
        top_k_relevant: Number of papers to mark as relevant per query.

    Returns:
        Dictionary with evaluation results for each model.
    """
    import pandas as pd

    # Default: run all models
    if models_to_run is None:
        models_to_run = ["tfidf", "minilm_l3", "sbert_faiss"]

    print("=" * 60)
    print("ArXiv Paper Recommendation - Model Evaluation")
    print("=" * 60)
    print(f"Queries:         {queries_path}")
    print(f"Data:            {data_path}")
    print(f"Output:          {output_path}")
    print(f"Top-K:           {k}")
    print(f"Models:          {', '.join(models_to_run)}")
    print(f"Relevant per Q:  {top_k_relevant}")
    print("=" * 60)

    # Load queries
    print("\n[Step 1/4] Loading evaluation queries...")
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")

    # Create corpus and generate pseudo labels
    print("\n[Step 2/4] Generating pseudo relevance labels with SBERT...")
    corpus = create_corpus(data_path)

    labeler = PseudoRelevanceLabeler(
        model_name="all-MiniLM-L6-v2",
        top_k_relevant=top_k_relevant
    )

    # Check for cached embeddings
    embeddings_cache = "evaluation/sbert_embeddings.npy"
    if os.path.exists(embeddings_cache):
        labeler.load_embeddings(embeddings_cache)
    else:
        labeler.compute_paper_embeddings(corpus)
        labeler.save_embeddings(embeddings_cache)

    # Generate labels
    labeled_queries, relevant_sets = labeler.generate_labels(queries)

    print(f"\nGenerated {len(relevant_sets)} relevance sets")
    print(f"Average relevant papers per query: {top_k_relevant}")

    # Initialize models
    print("\n[Step 3/4] Initializing retrieval models...")
    models = {}

    if "tfidf" in models_to_run:
        models["TF-IDF"] = TfidfModel(data_path)

    if "minilm_l3" in models_to_run:
        models["MiniLM-L3"] = MinilmModel(data_path)

    if "sbert_faiss" in models_to_run:
        models["SBERT + FAISS"] = SbertFaissModel(data_path)

    # Evaluate each model
    print("\n[Step 4/4] Evaluating models...")
    results = {}

    for model_name, model in models.items():
        metrics = run_model_evaluation(
            model_name, model, labeled_queries, relevant_sets, k
        )
        results[model_name] = metrics

    # Print and save results
    print_results_table(results, k)
    save_results_csv(results, output_path, k)

    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run evaluation experiments for ArXiv paper recommendation models"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="evaluation/evaluation_queries.json",
        help="Path to evaluation queries JSON file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="arxiv_subset.csv",
        help="Path to ArXiv metadata CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results.csv",
        help="Path to save results CSV file"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top results to consider (default: 5)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["tfidf", "minilm_l3", "sbert_faiss"],
        default=None,
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--top-k-relevant",
        type=int,
        default=20,
        help="Number of papers to mark as relevant per query (default: 20)"
    )

    args = parser.parse_args()

    run_experiments(
        queries_path=args.queries,
        data_path=args.data,
        output_path=args.output,
        k=args.k,
        models_to_run=args.models,
        top_k_relevant=args.top_k_relevant
    )


if __name__ == "__main__":
    main()
