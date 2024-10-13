import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import warnings
import requests
from bs4 import BeautifulSoup

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Streamlit setup
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research Tool using LLM")

# Initialize session state variables
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'urls_processed' not in st.session_state:
    st.session_state.urls_processed = False
if 'url_fields' not in st.session_state:
    st.session_state.url_fields = ['url1']
if 'document_sources' not in st.session_state:
    st.session_state.document_sources = []

# Add dynamic URL input fields
def add_url_field():
    new_field = f'url{len(st.session_state.url_fields) + 1}'
    st.session_state.url_fields.append(new_field)

st.sidebar.header("Input URLs")
for field in st.session_state.url_fields:
    st.sidebar.text_input(f"Enter {field}:", key=field)

if st.sidebar.button("Add URL"):
    add_url_field()

# Fetch content from a URL
def fetch_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('body')
        return main_content.get_text() if main_content else ""
    except Exception as e:
        st.warning(f"Error fetching content from {url}: {str(e)}")
        return ""

# Process URLs and create embeddings
def process_urls(urls):
    progress_bar = st.progress(0)
    st.sidebar.write("Processing the URLs...")

    try:
        documents = []
        for i, url in enumerate(urls):
            content = fetch_url_content(url)
            if content:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
                split_docs = splitter.create_documents([content])

                # Add the URL as metadata to each split document
                for doc in split_docs:
                    doc.metadata = {"source": url}
                    documents.append(doc)

            progress_bar.progress((i + 1) / len(urls) * 0.5)

        if not documents:
            raise ValueError("No valid content could be fetched from the provided URLs.")

        # Generate embeddings and create FAISS index
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.faiss_index = FAISS.from_documents(documents, embedding_model)

        # Store URLs for selection in the dropdown
        st.session_state.document_sources = urls

        progress_bar.progress(1.0)
        st.sidebar.success("Processing completed!")
        st.session_state.urls_processed = True
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.session_state.urls_processed = False
    finally:
        progress_bar.empty()

# Analyze button logic
if st.sidebar.button("Analyze"):
    urls = [st.session_state[field] for field in st.session_state.url_fields if st.session_state[field].strip()]
    if not urls:
        st.error("Please enter at least one valid URL.")
    else:
        process_urls(urls)

# Question-Answer interface
st.header("Ask Your Questions")
selected_url = st.selectbox("Select the Article to Query:", st.session_state.document_sources)
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not st.session_state.urls_processed:
        st.error("Please analyze the URLs first.")
    elif not query:
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Generating the answer..."):
                # Retrieve relevant documents using similarity search
                results = st.session_state.faiss_index.similarity_search(query, k=10)

                # Filter results based on the selected URL
                filtered_results = [doc for doc in results if doc.metadata.get('source') == selected_url]

                if filtered_results:
                    context = "\n".join([doc.page_content for doc in filtered_results])

                    # Generate answer using Groq API
                    groq_api_key = "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"
                    client = Groq(api_key=groq_api_key)

                    completion = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided context from news articles."},
                            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nPlease answer the question based on the given context. If the answer is not in the context, say 'I don't have enough information to answer this question accurately.'"}
                        ],
                        temperature=0.5,
                        max_tokens=1024,
                        top_p=1,
                        stream=False,
                        stop=None
                    )

                    answer = completion.choices[0].message.content
                    st.markdown(f"**Question:** {query}")
                    st.markdown(f"**Answer:** {answer}")
                    st.markdown(f"**Source:** [Link to Article]({selected_url})")
                else:
                    st.error("No relevant information found to answer the question for the selected article.")
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {str(e)}")

# Display processing status
if st.session_state.urls_processed:
    st.sidebar.success("URLs have been processed. You can ask multiple questions without re-analyzing.")
else:
    st.sidebar.info("Please input URLs and click 'Analyze' to process the news articles.")
