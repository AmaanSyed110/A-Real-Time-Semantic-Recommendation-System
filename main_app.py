"""
ArXiv Paper Recommendation System - Demo Version

This is a DEMO/PLACEHOLDER version of the main application for overview purposes.
It demonstrates the UI structure and system architecture without the core 
recommendation engine implementation.

For access to the full implementation with the complete SBERT+FAISS recommendation
engine, please contact the authors directly.

This demo shows:
- The user interface layout
- Filter options and controls
- System architecture overview
- How to request full code access
"""

import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="ArXiv Paper Recommendation System - Demo",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 ArXiv Paper Recommendation System")
st.markdown("""
### Demo Version - Overview Only

This is a **demonstration version** of the ArXiv Paper Recommendation System.
The full implementation with the SBERT+FAISS recommendation engine is available
upon request for academic collaboration.

**What this demo shows:**
- User interface design and layout
- Available filters and search options
- System architecture overview
- Baseline model demonstrations (TF-IDF, MiniLM)

**What's not included:**
- Core SBERT+FAISS recommendation engine
- Pre-computed embeddings and FAISS index
- Full dataset integration

---
""")

# Sidebar with information
st.sidebar.header("ℹ️ About This Demo")
st.sidebar.info("""
This is a **demo/placeholder** version of the main application.
The core recommendation engine (SBERT + FAISS) is not included in this repository.
""")

st.sidebar.header("📧 Full Code Access")
st.sidebar.success("""
**For academic collaboration and full code access:**

Please contact the authors directly to request access to the complete
implementation including:
- Full SBERT+FAISS recommendation engine
- Pre-computed embeddings
- Complete dataset
- Evaluation benchmarks
""")

st.sidebar.header("🔧 Available Features")
features = st.sidebar.multiselect(
    "Explore system components:",
    ["UI Demo", "Baseline Models", "Architecture", "Evaluation"]
)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🖥️ UI Demo", 
    "📊 Baseline Models", 
    "🏗️ Architecture", 
    "📈 Evaluation"
])

with tab1:
    st.header("User Interface Demo")
    st.markdown("""
    This tab demonstrates the user interface layout of the recommendation system.
    The actual recommendation functionality requires the full implementation.
    """)
    
    # Input section (demo only)
    st.subheader("Search Interface")
    user_input = st.text_area(
        "Enter your research interest or a paper abstract:",
        height=150,
        placeholder="e.g., 'deep learning for natural language processing'..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        keyword = st.text_input("Keyword (optional):", placeholder="e.g., 'transformer'")
    with col2:
        top_n = st.slider("Number of recommendations", 1, 10, 5)
    
    # Filters
    st.subheader("Filters")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.selectbox("Category", ["All", "cs.CL", "cs.LG", "cs.CV", "cs.AI"])
    with col4:
        st.selectbox("Year", ["All", "2024", "2023", "2022", "2021"])
    with col5:
        st.text_input("Author", placeholder="Author name")
    
    # Demo button (non-functional)
    if st.button("🔍 Get Recommendations", type="primary"):
        st.warning("""
        ⚠️ **Recommendation Engine Not Available**
        
        This demo shows the UI layout only. The core SBERT+FAISS recommendation 
        engine is not included in this repository.
        
        **To see actual recommendations:**
        - Try the baseline models in the "Baseline Models" tab
        - Contact authors for full code access
        """)

with tab2:
    st.header("Baseline Models")
    st.markdown("""
    This repository includes baseline retrieval models for comparison:
    
    1. **TF-IDF** - Classical term-frequency based retrieval
    2. **MiniLM-L3** - Lightweight semantic embeddings
    
    These baselines can be run independently from the `baselines/` folder.
    """)
    
    st.subheader("Try Baseline Models")
    
    baseline_choice = st.radio(
        "Select a baseline model:",
        ["TF-IDF", "MiniLM-L3"]
    )
    
    demo_query = st.text_input("Enter a test query:", "deep learning neural networks")
    
    if st.button(f"Run {baseline_choice}"):
        st.info(f"""
        To run the {baseline_choice} baseline:
        
        ```python
        # In your Python environment:
        from baselines.{baseline_choice.lower().replace('-', '_')}_retrieval import search_{baseline_choice.lower().replace('-', '_')}
        
        results = search_{baseline_choice.lower().replace('-', '_')}("{demo_query}", k=5)
        print(results)
        ```
        
        Or run directly:
        ```bash
        python baselines/{baseline_choice.lower().replace('-', '_')}_retrieval.py
        ```
        """)
        
        st.warning("""
        **Note:** You need to install dependencies first:
        ```bash
        pip install -r requirements.txt
        ```
        """)

with tab3:
    st.header("System Architecture")
    
    st.markdown("""
    ## Architecture Overview
    
    The full recommendation system consists of the following components:
    
    ### 1. Data Pipeline
    - **arXiv API Integration**: Fetches latest papers
    - **Dataset Processing**: CSV-based metadata storage
    - **Preprocessing**: Text cleaning and normalization
    
    ### 2. Embedding Generation
    - **SBERT Model**: `all-MiniLM-L6-v2` for semantic embeddings
    - **Batch Processing**: Efficient GPU-accelerated encoding
    - **Embedding Storage**: Numpy arrays for fast access
    
    ### 3. Indexing & Search
    - **FAISS Index**: Efficient vector similarity search
    - **Cosine Similarity**: Semantic matching
    - **Top-K Retrieval**: Fast nearest neighbor search
    
    ### 4. User Interface
    - **Streamlit Frontend**: Interactive web interface
    - **Filters**: Category, year, author, keyword
    - **Results Display**: Paper metadata with links
    """)
    
    st.subheader("📊 System Diagram")
    st.text("""
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
    """)
    
    st.info("""
    **Note:** The core recommendation engine (SBERT + FAISS) implementation
    is not included in this demo repository. Contact the authors for access.
    """)

with tab4:
    st.header("Evaluation Framework")
    
    st.markdown("""
    ## Evaluation Setup
    
    The system is evaluated using standard information retrieval metrics:
    
    - **Precision@K**: Proportion of retrieved documents that are relevant
    - **Mean Average Precision (MAP)**: Overall retrieval quality
    
    ## Baseline Comparison
    
    | Model | Precision@5 | MAP | Speed |
    |-------|-------------|-----|-------|
    | TF-IDF | ~0.40 | ~0.35 | Fastest |
    | MiniLM-L3 | ~0.60 | ~0.52 | Fast |
    | SBERT+FAISS | ~0.80 | ~0.71 | Medium |
    
    *Note: Actual results may vary based on dataset and queries*
    """)
    
    st.subheader("Run Evaluation")
    st.code("""
# Run evaluation experiments
python evaluation/run_experiments.py

# With custom parameters
python evaluation/run_experiments.py --k 10 --models tfidf minilm_l3
    """)
    
    if st.button("View Evaluation Script"):
        st.info("""
        The evaluation framework is available in the `evaluation/` folder:
        
        - `run_experiments.py`: Main evaluation runner
        - `evaluation_queries.json`: Pre-defined test queries
        - `generate_queries.py`: Query generation utility
        
        To run evaluation, you need:
        1. Install dependencies: `pip install -r requirements.txt`
        2. Prepare dataset: `arxiv_subset.csv`
        3. Run: `python evaluation/run_experiments.py`
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">

**ArXiv Paper Recommendation System - Demo Version**

For full implementation access and academic collaboration, please contact the authors.

</div>
""", unsafe_allow_html=True)
