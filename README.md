# Document Question-Answering System
## Overview
This project implements a document question-answering system using OpenAI's embedding and chat models combined with ChromaDB, a vector database. The workflow ingests text documents, splits them into manageable chunks, generates embeddings, stores them in a vector database, retrieves relevant chunks based on user queries, and generates concise answers using a large language model (LLM). The goal is to provide accurate, context-grounded responses to user questions based on the content of provided documents, such as answering queries about the Public Financial Management Act 2016 with the PEFA Framework.
Concept and Workflow
The system follows a Retrieval-Augmented Generation (RAG) approach, structured in four key steps:

### 1, Document Ingestion: Loads .txt files from a specified directory (./Article) and splits them into chunks (1000 characters with 20-character overlap) to ensure manageable text segments for processing.
### 2, Indexing and Storage: Generates embeddings for each chunk using OpenAI’s text-embedding-3-small model and stores them in ChromaDB for efficient retrieval.
### 3, Retrieval: Uses dense retrieval to fetch the top relevant chunks (default: 2) from ChromaDB based on a user’s query.
### 4, Generation: Combines retrieved chunks into a context, constructs a prompt, and uses OpenAI’s gpt-3.5-turbo model to generate a concise answer (limited to three sentences).

The idea is to create a scalable, robust system for answering questions grounded in document content, leveraging vector search for relevance and LLMs for natural language responses.
Features

Robust File Handling: Supports .txt files with automatic encoding detection using chardet to handle various text encodings.
Efficient Processing: Batch processing for embedding generation and database upserts to minimize API calls and improve performance.
Error Handling: Comprehensive checks for file/directory existence, permissions, API errors, and invalid inputs.
Logging: Detailed logging for debugging and monitoring execution.
Cross-Platform: Uses os.path for path handling to ensure compatibility across operating systems.

## Prerequisites

Python 3.9
Required packages:pip install chromadb openai python-dotenv chardet


An OpenAI API key (stored in a .env file as OPENAI_API_KEY).
A directory named Article containing .txt files with relevant content.

## Setup

Clone the repository:git clone <repository-url>
cd <repository-directory>

Install dependencies:pip install -r requirements.txt

Or manually:pip install chromadb openai python-dotenv chardet

Create a .env file in the project root with your OpenAI API key:OPENAI_API_KEY=your-openai-api-key

Place .txt files in the ./Article directory.

Usage
Run the script to process documents and answer a sample query:
python document_qa.py

## The script will:

Load and chunk documents from ./Article.
Generate and store embeddings in ChromaDB (chroma_persistent_storage).
Answer the example query: "State the Public Financial Management Act 2016 with the PEFA Framework in 100 word limit."

## To customize:

Modify DIRECTORY_PATH for a different document directory.
Adjust CHUNK_SIZE, CHUNK_OVERLAP, or MAX_CONTEXT_LENGTH in the script for different chunking or context limits.
Change the question variable for a different query.

## Example Output
INFO:__main__:Loading documents from directory
INFO:__main__:Loaded 5 documents
INFO:__main__:Splitting document1.txt into 3 chunks
...
INFO:__main__:Generating embeddings
INFO:__main__:Inserting chunks into database
INFO:__main__:Querying documents
INFO:__main__:Answer: The Public Financial Management Act 2016 of [country] governs fiscal discipline, transparency, and accountability in public finance, aligning with the PEFA Framework’s principles for assessing public financial management performance. It mandates strategic budgeting, expenditure control, and reporting to enhance efficiency. PEFA evaluates its implementation through indicators like budget credibility and audit effectiveness.

## Limitations/Known Issue

Supports only .txt files; other formats require additional loaders.
Relies on OpenAI API, which incurs costs and requires internet access.
Hardcoded models (text-embedding-3-small, gpt-3.5-turbo) may need updates if deprecated.
Does not use Langchain, which could enhance modularity (see Future Improvements).

## Future Improvements

Integrate Langchain for document loading, chunking, and prompt templating to align with advanced RAG workflows.
Support additional file formats (e.g., PDF, Markdown) using appropriate loaders.
Add a command-line interface for dynamic query input.
Implement configurable storage paths and database reset options.

