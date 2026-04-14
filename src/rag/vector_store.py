"""
Vector store setup and retrieval for the Study Coach RAG pipeline.
Uses ChromaDB with sentence-transformers embeddings.
"""
import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Embedding model (runs locally, no API needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Configure logging
logger = logging.getLogger(__name__)


def get_embeddings():
    """Get the HuggingFace embedding function."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_knowledge_base():
    """
    Load all .txt and .md files from data/knowledge_base/,
    split into chunks, and store in ChromaDB.
    """
    logger.info(f"📚 Loading documents from {KB_DIR}...")

    if not os.path.exists(KB_DIR):
        logger.warning(f"⚠️ Knowledge base directory not found: {KB_DIR}")
        return None

    # Load .txt files
    documents = []
    for fname in os.listdir(KB_DIR):
        fpath = os.path.join(KB_DIR, fname)
        if fname.endswith((".txt", ".md")):
            loader = TextLoader(fpath, encoding="utf-8")
            documents.extend(loader.load())

    if not documents:
        logger.warning("⚠️ No documents found in knowledge base.")
        return None

    logger.info(f"   Loaded {len(documents)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"   Split into {len(chunks)} chunks")

    # Create vector store
    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    logger.info(f"✅ Vector store created at {CHROMA_DIR}")
    return vectordb


def get_retriever(k=5):
    """
    Get a retriever from the persisted ChromaDB store.
    If the store doesn't exist, ingest the knowledge base first.
    """
    embeddings = get_embeddings()

    if not os.path.exists(CHROMA_DIR):
        logger.info("Vector store not found. Ingesting knowledge base...")
        vectordb = ingest_knowledge_base()
        if vectordb is None:
            raise RuntimeError("No knowledge base to ingest.")
    else:
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    return vectordb.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    ingest_knowledge_base()
