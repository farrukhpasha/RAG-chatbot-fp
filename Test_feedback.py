# =======================================================================
#        85% confidence, Testing feedback, role maintained, no code [Working]
# =======================================================================


import ollama
import faiss
import numpy as np
import json
import os

# Constants
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'
FEEDBACK_FILE = "feedback.json"
VECTOR_DB = []
THRESHOLD_CUTOFF = 0.85
faiss_index = None

# Step 1: Load Dataset
def load_dataset(file_path):
    """Load dataset from a text file and split it into paragraphs."""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
        paragraphs = content.split('\n\n')
        print(f'✅ Loaded {len(paragraphs)} paragraphs from {file_path}')
        return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

# Step 2: Initialize FAISS Index
def add_paragraphs_to_database(paragraphs):
    """Generate embeddings and add to vector database."""
    global VECTOR_DB
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=paragraphs)
        if 'embeddings' not in response or not response['embeddings']:
            print("❌ Error: No embeddings returned.")
            return

        for paragraph, embedding in zip(paragraphs, response['embeddings']):
            VECTOR_DB.append((paragraph, embedding))
        print(f'✅ Added {len(paragraphs)} paragraphs to database.')
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")

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
        print("✅ FAISS index built successfully.")
    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")

# Step 3: Retrieve Knowledge
def retrieve(query, top_n=3):
    """Retrieve the top N most relevant paragraphs using FAISS with confidence filtering."""
    global faiss_index
    if faiss_index is None:
        print("❌ Error: FAISS index not built.")
        return []

    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        if 'embeddings' not in response or not response['embeddings']:
            print("❌ Error: Failed to retrieve embeddings.")
            return []

        query_embedding = np.array(response['embeddings'][0]).astype('float32').reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, top_n)

        # Define strict confidence threshold
        threshold = THRESHOLD_CUTOFF
        results = [(VECTOR_DB[idx][0], 1 - distance) for idx, distance in zip(indices[0], distances[0]) if (1 - distance) >= threshold]

        if not results:
            return [("⚠️ I don’t have enough context to answer this. Please refer to the policy document.", 0.0)]  # Ensuring tuple format

        return results
    except Exception as e:
        print(f"❌ Error retrieving data: {e}")
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
        return response
    except Exception as e:
        return f"❌ Error generating response: {e}"

# Step 5: Feedback System
def save_feedback(user_input, response, feedback, correction=None):
    """Save user feedback and update FAISS if correction is provided."""
    feedback_data = {
        "user_input": user_input,
        "response": response,
        "feedback": feedback,
        "correction": correction if feedback == "W" else None
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as file:
        json.dump(feedback_data, file, indent=4)

    print("✅ Feedback saved!")

    if feedback == "W" and correction:
        add_corrected_response_to_faiss(user_input, correction)

def add_corrected_response_to_faiss(query, corrected_response):
    """Embed corrected response and update FAISS."""
    global VECTOR_DB, faiss_index

    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=[corrected_response])
        if not response or 'embeddings' not in response or not response['embeddings']:
            print("❌ Error generating embeddings for correction.")
            return

        new_embedding = response['embeddings'][0]
        VECTOR_DB.append((corrected_response, new_embedding))  

        # Rebuild FAISS index
        embedding_dim = len(new_embedding)
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        embeddings = np.array([emb for _, emb in VECTOR_DB]).astype('float32')
        faiss_index.add(embeddings)

        print("✅ FAISS index updated with correction!")

    except Exception as e:
        print(f"❌ Error updating FAISS index: {e}")

# Step 6: Prepare Fine-Tuning Data
def prepare_fine_tuning_data():
    """Prepare fine-tuning data from feedback stored as a list."""
    if not os.path.exists(FEEDBACK_FILE) or os.path.getsize(FEEDBACK_FILE) == 0:
        print("❌ No feedback data found.")
        return

    with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)  # Now, it correctly reads a JSON list
        except json.JSONDecodeError as e:
            print(f"❌ Error reading feedback file: {e}")
            return

    fine_tuning_data = [
        {"input": entry["user_input"], "expected_output": entry["correction"]}
        for entry in data if entry["feedback"] == "W" and entry["correction"]
    ]

    with open("fine_tuning_data.json", "w", encoding="utf-8") as file:
        json.dump(fine_tuning_data, file, indent=4)

    print("✅ Fine-tuning data prepared successfully!")

# Step 7: CLI Interface
if __name__ == "__main__":
    print("\n===== AKUgpt - Leave Policy Chatbot =====")
    dataset = load_dataset("All_leaves.txt")
    if dataset:
        add_paragraphs_to_database(dataset)
        build_faiss_index()
        # prepare_fine_tuning_data()

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        response = chatbot_interaction(user_query)
        print(f"AKUgpt: {response}")

        feedback = input("[C]orrect / [W]rong ?: ")
        if feedback == "W":
            correction = input("Provide the correct response: ")
            save_feedback(user_query, response, feedback, correction)
        else:
            save_feedback(user_query, response, feedback)
