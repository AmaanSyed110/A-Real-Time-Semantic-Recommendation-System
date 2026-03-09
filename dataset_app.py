import streamlit as st
import requests
import feedparser
import pandas as pd
import os

# Function to fetch papers from arXiv
def fetch_arxiv_papers(query, start=0, max_results=100):
    """
    Fetch papers from the arXiv API based on a query.

    Parameters:
        query (str): Search query for arXiv (e.g., 'machine learning').
        start (int): Starting index for the query (default: 0).
        max_results (int): Number of results to fetch (default: 100, max: 2000).

    Returns:
        pd.DataFrame: A DataFrame containing fetched paper details.
    """
    base_url = "http://export.arxiv.org/api/query?"
    full_query = f"search_query=all:{query}&start={start}&max_results={max_results}"

    # Send the request
    response = requests.get(base_url + full_query)
    if response.status_code != 200:
        st.error("Failed to fetch data from the arXiv API. Please try again.")
        return pd.DataFrame()

    # Parse the response
    feed = feedparser.parse(response.text)

    # Extract details
    papers = []
    for entry in feed.entries:
        paper = {
            "id": entry.id.split("/")[-1],
            "title": entry.title,
            "abstract": entry.summary.replace('\n', ' '),
            "categories": ", ".join(tag.term for tag in entry.tags),
            "update_date": entry.updated
        }
        papers.append(paper)

    # Convert to DataFrame
    return pd.DataFrame(papers)

def update_dataset(new_data, existing_file):
    """
    Update the existing dataset with new data and save it.

    Parameters:
        new_data (pd.DataFrame): DataFrame containing new paper details.
        existing_file (str): Path to the existing CSV dataset.

    Returns:
        pd.DataFrame: Updated dataset.
    """
    if os.path.exists(existing_file):
        # Load the existing dataset
        existing_data = pd.read_csv(existing_file)
        # Append new data and drop duplicates based on 'id'
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset="id")
    else:
        # If no existing dataset, use the new data
        updated_data = new_data

    # Save the updated dataset
    updated_data.to_csv(existing_file, index=False)
    return updated_data

# Streamlit app
def main():
    st.title("ArXiv Dataset Updater")
    st.write("Update your existing research paper dataset with the latest papers from arXiv.")

    # Input fields
    search_query = st.text_input("Enter your search query (e.g., 'machine learning')", "")
    max_results = st.number_input("Number of results to fetch", min_value=1, max_value=200, value=50)
    existing_file = st.text_input("Path to your existing dataset (CSV file)", "arxiv_metadata.csv")
    update_button = st.button("Update Dataset")

    if update_button and search_query:
        # Display the number of existing papers in the dataset
        if os.path.exists(existing_file):
            existing_data = pd.read_csv(existing_file)
            existing_count = len(existing_data)
            st.write(f"Number of existing papers in dataset: {existing_count}")
        else:
            existing_count = 0
            st.write("No existing dataset found, starting with an empty dataset.")

        with st.spinner("Fetching papers from arXiv..."):
            new_papers = fetch_arxiv_papers(search_query, max_results=max_results)

        if not new_papers.empty:
            st.success(f"Fetched {len(new_papers)} new papers!")

            # Update the existing dataset
            with st.spinner("Updating the dataset..."):
                updated_dataset = update_dataset(new_papers, existing_file)

            st.success(f"Dataset updated! Total papers in the file: {len(updated_dataset)}")

            # Show newly fetched papers only (not the full updated dataset)
            st.write("Newly Fetched Papers:")
            st.dataframe(new_papers)

            # Provide download option for updated dataset
            csv_data = updated_dataset.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Updated Dataset",
                data=csv_data,
                file_name="updated_dataset.csv",
                mime="text/csv"
            )
        else:
            st.warning("No new papers found for the given query. Dataset remains unchanged.")

if __name__ == "__main__":
    main()
