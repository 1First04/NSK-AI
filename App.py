#Import packages
import os
import logging
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from openai.error import OpenAIError
from chromadb.utils import embedding_functions
import chardet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import chromadb
    from openai import OpenAI
    from dotenv import load_dotenv
    import chardet
except ImportError as e:
    raise ImportError("Required packages (chromadb, openai, python-dotenv, chardet) are not installed. Run 'pip install chromadb openai python-dotenv chardet'")

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
DIRECTORY_PATH = "./Article"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20
MAX_CONTEXT_LENGTH = 3000  # Approximate token limit (1 token-4 chars)

# Initialize OpenAI and Chroma clients
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name=EMBEDDING_MODEL
)
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)
client = OpenAI(api_key=openai_key)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    logger.info("Loading documents from directory")
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "rb") as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)["encoding"] or "utf-8"
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    documents.append({"id": filename, "text": file.read()})
            except UnicodeDecodeError:
                logger.warning(f"Could not decode {filename} with {encoding}, skipping")
    if not documents:
        raise ValueError(f"No readable .txt files found in {directory_path}")
    return documents

# Function to split text into chunks
def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        while end > start and text[end-1] not in " \n\t":
            end -= 1
        if end == start:
            end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Function to generate embeddings using OpenAI API
def get_openai_embedding(texts):
    try:
        response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [data.embedding for data in response.data]
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI embedding error: {str(e)}")

# Function to query documents
def query_documents(question, n_results=2):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Query must be a non-empty string")
    logger.info("Querying documents")
    results = collection.query(query_texts=[question], n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    if not relevant_chunks:
        logger.warning("No relevant chunks found for the query")
    return relevant_chunks

# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        f"\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI chat completion error: {str(e)}")

# Main execution
try:
    # Load documents
    documents = load_documents_from_directory(DIRECTORY_PATH)
    logger.info(f"Loaded {len(documents)} documents")

    # Split documents into chunks
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        logger.info(f"Splitting {doc['id']} into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

    # Generate embeddings in batch
    logger.info("Generating embeddings")
    texts = [doc["text"] for doc in chunked_documents]
    embeddings = get_openai_embedding(texts)
    for doc, embedding in zip(chunked_documents, embeddings):
        doc["embedding"] = embedding

    # Batch upsert to Chroma
    logger.info("Inserting chunks into database")
    collection.upsert(
        ids=[doc["id"] for doc in chunked_documents],
        documents=[doc["text"] for doc in chunked_documents],
        embeddings=[doc["embedding"] for doc in chunked_documents]
    )

    # Example query and response
    question = "State the Public Financial Management Act 2016 with the PEFA Framework in 100 word limit."
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    logger.info(f"Answer: {answer}")

except Exception as e:
    logger.error(f"Error: {str(e)}")
