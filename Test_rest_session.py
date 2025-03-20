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
import uuid
from fastapi import FastAPI, HTTPException
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
sessions = {}

app = FastAPI()

class FeedbackRequest(BaseModel):
    user_input: str
    response: str
    feedback: str
    correction: str = None

class QueryRequest(BaseModel):
    session_id: str
    query: str

# Root endpoint
@app.get("/")
def root():
    logger.info("🏠 Root endpoint accessed.")
    return {"message": "FAISS AI Web Service is running"}

# Start a new session
@app.post("/start_session")
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"history": []}
    logger.info(f"✅ New session started: {session_id}")
    return {"session_id": session_id}

# End a session
@app.delete("/end_session")
def end_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"✅ Session ended: {session_id}")
        return {"message": "Session ended successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# Load Dataset
@app.on_event("startup")
def startup():
    dataset = load_dataset("All_leaves.txt")
    if dataset:
        add_paragraphs_to_database(dataset)
        build_faiss_index()

def load_dataset(file_path):
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

def add_paragraphs_to_database(paragraphs):
    global VECTOR_DB
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=paragraphs)
        if 'embeddings' not in response or not response['embeddings']:
            logger.error("❌ Error: No embeddings returned.")
            return

        for paragraph, embedding in zip(paragraphs, response['embeddings']):
            VECTOR_DB.append((paragraph, embedding))
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")

def build_faiss_index():
    global faiss_index
    try:
        if not VECTOR_DB:
            raise ValueError("Vector database is empty.")
        embedding_dim = len(VECTOR_DB[0][1])
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        embeddings = np.array([embedding for _, embedding in VECTOR_DB]).astype('float32')
        faiss_index.add(embeddings)
    except Exception as e:
        logger.error(f"❌ Error building FAISS index: {e}")

def retrieve(query, top_n=3):
    global faiss_index
    if faiss_index is None:
        return []

    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        query_embedding = np.array(response['embeddings'][0]).astype('float32').reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, top_n)

        results = [(VECTOR_DB[idx][0], 1 - distance) for idx, distance in zip(indices[0], distances[0]) if (1 - distance) >= THRESHOLD_CUTOFF]
        return results if results else [("⚠️ I don’t have enough context to answer this.", 0.0)]
    except Exception as e:
        logger.error(f"❌ Error retrieving data: {e}")
        return []

def chatbot_interaction(session_id, user_input):
    retrieved_knowledge = retrieve(user_input)
    if not retrieved_knowledge:
        return "No relevant context found."

    context = '\n'.join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
    messages = sessions[session_id]["history"] + [
        {'role': 'system', 'content': "You are an intelligent chatbot specializing in leave policies.\n" + "Relevant context:\n" + context},
        {'role': 'user', 'content': user_input},
    ]

    try:
        stream = ollama.chat(model=LANGUAGE_MODEL, messages=messages, stream=True)
        response = ''.join(chunk['message']['content'] for chunk in stream)
        sessions[session_id]["history"].append({'role': 'assistant', 'content': response})
        return response.replace("\n", "<br>")
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
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    response = chatbot_interaction(request.session_id, request.query)
    return {"response": response.replace("\n", "<br>")}

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    save_feedback(request.user_input, request.response, request.feedback, request.correction)
    return {"message": "Feedback submitted successfully"}

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

