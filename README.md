# A Real-Time Semantic Recommendation System

> **⚠️ DEMO / OVERVIEW VERSION**
> 
> This repository contains a **demo/overview version** of the Real-Time ArXiv Paper Recommendation System. 
> It is provided for academic reference and reproducibility purposes.
> 
> **The core SBERT+FAISS recommendation engine is not included in this repository.**
> 
> For access to the complete implementation including the full recommendation engine, pre-computed 
> embeddings, and evaluation benchmarks, please contact the authors directly for academic collaboration.

---

## Table of Contents

- [Overview](#overview)
- [What's Included](#whats-included)
- [What's Not Included](#whats-not-included)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Running the Demo](#running-the-demo)
- [Baseline Models](#baseline-models)
- [Evaluation Framework](#evaluation-framework)
- [Requesting Full Code Access](#requesting-full-code-access)
- [Citation](#citation)

---

## Overview

The **Real-Time ArXiv Paper Recommendation System** is a semantic search-based recommendation 
system for academic papers on arXiv. It employs **SBERT (Sentence-BERT) embeddings** with 
**FAISS (Facebook AI Similarity Search)** for efficient vector-based similarity search to 
deliver contextually relevant paper recommendations.

This **demo repository** provides:
- A demonstration Streamlit interface showing the system UI
- Baseline retrieval models (TF-IDF, MiniLM-L3) for comparison
- Evaluation framework with standard IR metrics
- System architecture documentation

**Full Version Features** (available upon request):
- Complete SBERT+FAISS recommendation engine
- Pre-computed embeddings for efficient retrieval
- Full dataset integration with arXiv API
- Comprehensive evaluation benchmarks

---

## What's Included

This repository contains:

| Component | Description | Status |
|-----------|-------------|--------|
| `main_app.py` | Demo Streamlit UI | ✅ Demo version |
| `baselines/` | TF-IDF, MiniLM-L3, USE retrieval | ✅ Full implementation |
| `evaluation/` | Evaluation framework and metrics | ✅ Full implementation |
| `dataset_app.py` | Dataset update utility | ✅ Full implementation |
| `requirements.txt` | Python dependencies | ✅ Complete |
| `LICENSE` | MIT License | ✅ Complete |

---

## What's Not Included

The following components are **not included** in this demo repository:

| Component | Reason |
|-----------|--------|
| Core SBERT+FAISS engine | Proprietary implementation |
| Pre-computed embeddings | Large file size (~500MB+) |
| FAISS index files | Generated from full dataset |
| Complete dataset | Must be obtained from arXiv |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ArXiv Paper Recommendation System            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐ │
│  │   User       │     │   Streamlit  │     │   Recommendation │ │
│  │   Interface  │────▶│   Frontend   │────▶│   Engine         │ │
│  └──────────────┘     └──────────────┘     └────────┬─────────┘ │
│                                                      │           │
│  ┌──────────────┐     ┌──────────────┐              ▼           │
│  │   arXiv      │     │   Metadata   │     ┌──────────────────┐ │
│  │   API        │────▶│   CSV        │     │   SBERT Encoder  │ │
│  └──────────────┘     └──────┬───────┘     └────────┬─────────┘ │
│                              │                      │           │
│                              │                      ▼           │
│                              │            ┌──────────────────┐ │
│                              │            │   FAISS Index    │ │
│                              │            │   (Vector Search)│ │
│                              │            └────────┬─────────┘ │
│                              │                     │            │
│                              ▼                     ▼            │
│                     ┌──────────────────────────────────────┐   │
│                     │      Retrieval & Ranking             │   │
│                     │   - Cosine Similarity Search         │   │
│                     │   - Top-K Selection                  │   │
│                     │   - Metadata Filtering               │   │
│                     └──────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description | Availability |
|-----------|-------------|--------------|
| **Streamlit Frontend** | Interactive web UI | ✅ Included (demo) |
| **SBERT Encoder** | `all-MiniLM-L6-v2` embeddings | ⚠️ Demo only |
| **FAISS Index** | Vector similarity search | ❌ Not included |
| **Metadata Store** | CSV-based storage | ✅ Utility provided |

---

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/AmaanSyed110/A-Real-Time-Semantic-Recommendation-System.git
cd A-Real-Time-Semantic-Recommendation-System
```

### Step 2: Create Virtual Environment

```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements include**:
- `streamlit` - Web application framework
- `sentence-transformers` - SBERT embeddings
- `faiss-cpu` - Vector similarity search
- `scikit-learn` - TF-IDF baseline and metrics
- `pandas`, `numpy` - Data processing
- `requests`, `feedparser` - arXiv API integration

---

## Running the Demo

### Start the Streamlit Demo App

```bash
streamlit run main_app.py
```

The demo application will open in your default browser at `http://localhost:8501`.

### What the Demo Shows

- **UI Layout**: Interactive interface design
- **Filter Options**: Category, year, author, keyword filters
- **Baseline Models**: Working TF-IDF and MiniLM-L3 implementations
- **Architecture Overview**: System component documentation

### Limitations

The demo version does **not** provide:
- Actual SBERT+FAISS recommendations
- Pre-computed embeddings
- Full dataset integration

---

## Baseline Models

This repository includes **fully functional** baseline retrieval models for comparison 
and research purposes.

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**Location**: `baselines/tfidf_retrieval.py`

```python
from baselines.tfidf_retrieval import search_tfidf

results = search_tfidf("deep learning neural networks", k=5)
```

### 2. MiniLM-L3 (Lightweight Semantic Baseline)

**Location**: `baselines/minilm_retrieval.py`

```python
from baselines.minilm_retrieval import search_minilm

results = search_minilm("deep learning neural networks", k=5)
```

### 3. Universal Sentence Encoder

**Location**: `baselines/use_retrieval.py`

```python
from baselines.use_retrieval import search_use

results = search_use("deep learning neural networks", k=5)
```

### Running Baselines Directly

```bash
# Run TF-IDF baseline
python baselines/tfidf_retrieval.py

# Run MiniLM-L3 baseline
python baselines/minilm_retrieval.py

# Run USE baseline
python baselines/use_retrieval.py
```

---

## Evaluation Framework

The repository includes a comprehensive evaluation framework for comparing retrieval models.

### Evaluation Metrics

- **Precision@K**: Proportion of retrieved documents that are relevant
- **Mean Average Precision (MAP)**: Average precision across all queries

### Run Evaluation

```bash
# Run all evaluation experiments
python evaluation/run_experiments.py

# With custom parameters
python evaluation/run_experiments.py --k 10 --models tfidf minilm_l3
```

### Expected Output

```
============================================================
EXPERIMENT RESULTS (K=5)
============================================================
Model                     | Precision@5  | MAP
------------------------------------------------------------
TF-IDF                    | 0.4000       | 0.3500
MiniLM-L3                 | 0.6000       | 0.5200
SBERT + FAISS             | 0.8000       | 0.7100
============================================================

Best Precision@5: SBERT + FAISS (0.8000)
Best MAP:           SBERT + FAISS (0.7100)
```

**Note**: SBERT+FAISS results are shown for reference. Running this model requires 
the full implementation.

---

## Dataset

### Preparing Your Dataset

The system requires an arXiv metadata CSV file. You can:

1. **Use the provided utilities**:
   ```bash
   # Convert JSON to CSV (if you have the full arXiv JSON dump)
   python json_to_csv.py
   
   # Create a subset for testing
   python arxiv_subset.py
   ```

2. **Fetch from arXiv API**:
   ```bash
   streamlit run dataset_app.py
   ```

3. **Download from Kaggle**: [Cornell arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)

### Dataset Schema

| Field | Description |
|-------|-------------|
| `id` | arXiv paper identifier |
| `title` | Paper title |
| `abstract` | Paper summary |
| `authors` | Paper authors |
| `categories` | arXiv categories |
| `doi` | Digital Object Identifier |

---

## Requesting Full Code Access

### For Academic Collaboration

If you are interested in accessing the **complete implementation** including:

- Full SBERT+FAISS recommendation engine
- Pre-computed embeddings and FAISS index
- Complete evaluation benchmarks
- Technical documentation

Please contact the authors directly:

📧 **Email**: [Contact via GitHub](https://github.com/AmaanSyed110)

### Request Information

When requesting access, please include:
- Your name and affiliation
- Purpose of use (research, education, etc.)
- Intended application
- Any specific requirements

Access is granted for **academic and research purposes** under appropriate agreements.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **Streamlit** | Interactive web application |
| **Sentence-Transformers** | SBERT embedding generation |
| **FAISS** | Vector similarity search |
| **scikit-learn** | TF-IDF and evaluation metrics |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computations |
| **arXiv API** | Paper metadata retrieval |

---

## Contributing

Contributions are welcome! Please note that core engine modifications may require 
separate coordination with the authors.

### Areas for Contribution

- Additional baseline models
- Advanced evaluation metrics
- UI/UX improvements
- Documentation enhancements
- Bug fixes and optimizations

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The demo version is provided under MIT License. Full implementation access
may be subject to additional terms for academic collaboration.

---

## Contact

For questions, collaboration inquiries, or full code access requests:

- **GitHub**: [AmaanSyed110](https://github.com/AmaanSyed110)
- **Issues**: [GitHub Issues](https://github.com/AmaanSyed110/A-Real-Time-Semantic-Recommendation-System/issues)

---

## Acknowledgments

- **arXiv** for providing the academic paper repository
- **Sentence-Transformers** for SBERT models
- **FAISS** for efficient vector search
- **Cornell University** for the arXiv dataset
