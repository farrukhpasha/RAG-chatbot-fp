# =======================================================================
#        85% confidence, Testing feedback, role maintained, no code [Working]
# =======================================================================
# Copied from Test_feedback.py #


# import logging
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import ollama
# import faiss
# import numpy as np
# import json
# import os

# # Constants
# EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
# LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'
# FEEDBACK_FILE = "feedback.json"
# VECTOR_DB = []
# THRESHOLD_CUTOFF = 0.85
# faiss_index = None


# # Configure Logging
# logging.basicConfig(
#     level=logging.INFO,  # Use logging.DEBUG for more details
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.StreamHandler()  # Log to console
#     ]
# )

# logger = logging.getLogger(__name__)


# # Initialize FastAPI
# app = FastAPI()

# class QueryRequest(BaseModel):
#     query: str

# class FeedbackRequest(BaseModel):
#     user_input: str
#     response: str
#     feedback: str
#     correction: str = None

# # Load Dataset
# def load_dataset(file_path):
#     """Load dataset from a text file."""
#     try:
#         with open(file_path, 'r', encoding="utf-8") as file:
#             content = file.read()
#         paragraphs = content.split('\n\n')
#         return [p.strip() for p in paragraphs if p.strip()]
#     except FileNotFoundError:
#         return []

# # Generate Embeddings and Add to FAISS
# def add_paragraphs_to_database(paragraphs):
#     """Generate embeddings and add to vector database."""
#     global VECTOR_DB
#     try:
#         response = ollama.embed(model=EMBEDDING_MODEL, input=paragraphs)
#         if 'embeddings' not in response or not response['embeddings']:
#             return

#         for paragraph, embedding in zip(paragraphs, response['embeddings']):
#             VECTOR_DB.append((paragraph, embedding))
#     except Exception as e:
#         print(f"Error generating embeddings: {e}")

# def build_faiss_index():
#     """Build FAISS index."""
#     global faiss_index
#     try:
#         if not VECTOR_DB:
#             raise ValueError("Vector database is empty.")

#         embedding_dim = len(VECTOR_DB[0][1])
#         faiss_index = faiss.IndexFlatL2(embedding_dim)
#         embeddings = np.array([embedding for _, embedding in VECTOR_DB]).astype('float32')
#         faiss_index.add(embeddings)
#     except Exception as e:
#         print(f"Error building FAISS index: {e}")

# # Retrieve Knowledge
# def retrieve(query, top_n=3):
#     """Retrieve most relevant results from FAISS."""
#     global faiss_index
#     if faiss_index is None:
#         return []

#     try:
#         response = ollama.embed(model=EMBEDDING_MODEL, input=query)
#         query_embedding = np.array(response['embeddings'][0]).astype('float32').reshape(1, -1)
#         distances, indices = faiss_index.search(query_embedding, top_n)

#         results = [(VECTOR_DB[idx][0], 1 - distance) for idx, distance in zip(indices[0], distances[0]) if (1 - distance) >= THRESHOLD_CUTOFF]

#         if not results:
#             return [("⚠️ I don’t have enough context to answer this.", 0.0)]

#         return results
#     except Exception as e:
#         print(f"Error retrieving data: {e}")
#         return []

# # Chatbot Interaction
# INSTRUCTION_PROMPT = (
#     "You are an intelligent chatbot specializing in leave policies.\n"
#     "Adhere strictly to the provided context and do not introduce any information not explicitly mentioned."
# )

# def chatbot_interaction(user_input):
#     """Chatbot response based on retrieved knowledge."""
#     retrieved_knowledge = retrieve(user_input)
#     if not retrieved_knowledge:
#         return "No relevant context found."

#     context = '\n'.join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
#     messages = [
#         {'role': 'system', 'content': INSTRUCTION_PROMPT + "\nRelevant context:\n" + context},
#         {'role': 'user', 'content': user_input},
#     ]
    
#     try:
#         stream = ollama.chat(model=LANGUAGE_MODEL, messages=messages, stream=True)
#         response = ''.join(chunk['message']['content'] for chunk in stream)
#         return response
#     except Exception as e:
#         return f"Error generating response: {e}"

# # Save Feedback
# def save_feedback(user_input, response, feedback, correction=None):
#     """Save user feedback for future improvements."""
#     feedback_data = {
#         "user_input": user_input,
#         "response": response,
#         "feedback": feedback,
#         "correction": correction if feedback == "W" else None
#     }

#     with open(FEEDBACK_FILE, "a", encoding="utf-8") as file:
#         json.dump(feedback_data, file, indent=4) 


import logging
import os
import json
import numpy as np
import faiss
import ollama
from fastapi import FastAPI
from pydantic import BaseModel

# Define log file path
LOG_FILE = "faiss_service.log"

# Ensure the log directory exists (optional)
log_dir = os.path.dirname(LOG_FILE)
if log_dir and not os.path.exists(log_dir) and log_dir != "":
    os.makedirs(log_dir)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")  # Log to file
    ]
)

# Define logger variable
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'
FEEDBACK_FILE = "feedback.json"
VECTOR_DB = []
THRESHOLD_CUTOFF = 0.85
faiss_index = None

app = FastAPI()

class FeedbackRequest(BaseModel):
     user_input: str
     response: str
     feedback: str
     correction: str = None


# Pydantic Model for API Request
class QueryRequest(BaseModel):
    query: str

# Step 1: Load Dataset
def load_dataset(file_path):
    """Load dataset from a text file and split it into paragraphs."""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
        paragraphs = content.split('\n\n')
        logger.info(f"✅ Loaded {len(paragraphs)} paragraphs from {file_path}")
        return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    except FileNotFoundError:
        logger.error(f"❌ Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return []

# Step 2: Initialize FAISS Index
def add_paragraphs_to_database(paragraphs):
    """Generate embeddings and add to vector database."""
    global VECTOR_DB
    try:
        logger.info("Generating embeddings for dataset...")
        response = ollama.embed(model=EMBEDDING_MODEL, input=paragraphs)
        if 'embeddings' not in response or not response['embeddings']:
            logger.error("❌ Error: No embeddings returned.")
            return

        for paragraph, embedding in zip(paragraphs, response['embeddings']):
            VECTOR_DB.append((paragraph, embedding))
        logger.info(f"✅ Added {len(paragraphs)} paragraphs to database.")
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")

def build_faiss_index():
    """Build FAISS index with paragraph embeddings."""
    global faiss_index
    try:
        if not VECTOR_DB:
            raise ValueError("Vector database is empty.")
        
        embedding_dim = len(VECTOR_DB[0][1])
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        embeddings = np.array([embedding for _, embedding in VECTOR_DB]).astype('float32')
        faiss_index.add(embeddings)
        logger.info("✅ FAISS index built successfully.")
    except Exception as e:
        logger.error(f"❌ Error building FAISS index: {e}")

# Step 3: Retrieve Knowledge
def retrieve(query, top_n=3):
    """Retrieve the top N most relevant paragraphs using FAISS with confidence filtering."""
    global faiss_index
    if faiss_index is None:
        logger.error("❌ Error: FAISS index not built.")
        return []

    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        if 'embeddings' not in response or not response['embeddings']:
            logger.error("❌ Error: Failed to retrieve embeddings.")
            return []

        query_embedding = np.array(response['embeddings'][0]).astype('float32').reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, top_n)

        results = [
            (VECTOR_DB[idx][0], 1 - distance)
            for idx, distance in zip(indices[0], distances[0])
            if (1 - distance) >= THRESHOLD_CUTOFF
        ]

        if not results:
            logger.warning("⚠️ No relevant context found for query. Score=",1 - distance)
            return [("⚠️ I don’t have enough context to answer this.", 0.0)]

        return results
    except Exception as e:
        logger.error(f"❌ Error retrieving data: {e}")
        return []

# Step 4: Chatbot Response
INSTRUCTION_PROMPT = (
    "You are an intelligent chatbot specializing in leave policies.\n"
    "Adhere strictly to the provided context and do not introduce any information not explicitly mentioned."
)

def chatbot_interaction(user_input):
    """Handles chatbot interaction based on user input."""
    retrieved_knowledge = retrieve(user_input)
    if not retrieved_knowledge:
        return "No relevant context found."

    context = '\n'.join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
    messages = [
        {'role': 'system', 'content': INSTRUCTION_PROMPT + "\nRelevant context:\n" + context},
        {'role': 'user', 'content': user_input},
    ]
    
    try:
        stream = ollama.chat(model=LANGUAGE_MODEL, messages=messages, stream=True)
        response = ''.join(chunk['message']['content'] for chunk in stream)
        logger.info("✅ Response generated successfully.")
        return response.replace("\n", "<br>")  # Format newlines for web
    except Exception as e:
        logger.error(f"❌ Error generating response: {e}")
        return f"❌ Error: {e}"

# Save Feedback
def save_feedback(user_input, response, feedback, correction=None):
    """Save user feedback for future improvements."""
    feedback_data = {
        "user_input": user_input,
        "response": response,
        "feedback": feedback,
        "correction": correction if feedback == "W" else None
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as file:
        json.dump(feedback_data, file, indent=4) 

# API Endpoints
@app.get("/")
def root():
    logger.info("🏠 Root endpoint accessed.")
    return {"message": "FAISS AI Web Service is running"}

@app.post("/query")
def query_api(request: QueryRequest):
    logger.info(f"📥 Received query: {request.query}")
    response = chatbot_interaction(request.query)
    logger.info(f"📤 Sending response: {response}")
    return {"response": response.replace("\n", "<br>") } # Replace newlines with <br>

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    save_feedback(request.user_input, request.response, request.feedback, request.correction)
    return {"message": "Feedback submitted successfully"}

# Load dataset and build FAISS index on startup
@app.on_event("startup")
def startup():
    dataset = load_dataset("All_leaves.txt")
    if dataset:
        add_paragraphs_to_database(dataset)
        build_faiss_index()
