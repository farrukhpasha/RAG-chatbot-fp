import ollama
import faiss
import numpy as np
import sys

# Step 1: Load the text file and split it into paragraphs
def load_dataset(file_path):
    """Load the dataset from a text file and split it into paragraphs."""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
        paragraphs = content.split('\n\n')  # Split into paragraphs by double newlines
        print(f'✅ Loaded {len(paragraphs)} paragraphs from {file_path}')
        return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found. Please check the path.")
        return []
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

# Step 2: Initialize Vector Database and FAISS Index
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'
VECTOR_DB = []
faiss_index = None

def add_paragraphs_to_database(paragraphs):
    """Generate embeddings for paragraphs and add to the vector database."""
    global VECTOR_DB
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=paragraphs)
        if not response or 'embeddings' not in response or not response['embeddings']:
            print("❌ Error: No embeddings returned from embedding service.")
            return
        
        VECTOR_DB = [(paragraph, embedding) for paragraph, embedding in zip(paragraphs, response['embeddings'])]
        print(f'✅ Added {len(paragraphs)} paragraphs to the vector database.')
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")

def build_faiss_index():
    """Build the FAISS index with paragraph embeddings."""
    global faiss_index
    try:
        if not VECTOR_DB:
            raise ValueError("Vector database is empty. Cannot build FAISS index.")
        
        embedding_dim = len(VECTOR_DB[0][1])
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        embeddings = np.array([embedding for _, embedding in VECTOR_DB]).astype('float32')
        faiss_index.add(embeddings)
        print("✅ FAISS index built successfully.")
    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")

def retrieve(query, top_n=3):
    """Retrieve the top N most relevant paragraphs using FAISS."""
    global faiss_index
    if faiss_index is None:
        print("❌ Error: FAISS index is not built. Please build it before retrieval.")
        return []

    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        if not response or 'embeddings' not in response or not response['embeddings']:
            print("❌ Error: Failed to retrieve embeddings for query.")
            return []

        query_embedding = np.array(response['embeddings'][0]).astype('float32').reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, top_n)

        # Define a confidence threshold (adjust as needed)
        threshold = 0.8  
        results = [(VECTOR_DB[idx][0], 1 - distance) for idx, distance in zip(indices[0], distances[0]) if (1 - distance) >= threshold]

        return results
    except Exception as e:
        print(f"❌ Error retrieving data: {e}")
        return []

# Step 3: Chatbot Interaction
INSTRUCTION_PROMPT = (
    "You are an intelligent and professional chatbot specializing in leave policies.\n"
    "You must strictly adhere to the provided context.\n"
    "Do not introduce any information not explicitly mentioned in the retrieved content.\n"
    "If the context does not provide an answer, reply with: 'I can only answer questions related to the leave policy from the provided document.'"
)

def chatbot_interaction(user_input):
    """Handles chatbot interaction based on user input."""
    retrieved_knowledge = retrieve(user_input)

    if not retrieved_knowledge:
        return "⚠️ I can only answer questions related to the leave policy from the provided document."

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
        return f"❌ Error generating chatbot response: {e}"

# Step 4: Command-Line Interface
if __name__ == "__main__":
    print("\n===== 🤖 AKUgpt - Leave Policy Chatbot =====")
    file_path = 'All_leaves.txt'  # Path to the text file
    dataset = load_dataset(file_path)

    if dataset:
        add_paragraphs_to_database(dataset)
        build_faiss_index()
    
    print("\n💡 Type 'exit' to quit the chatbot.")
    while True:
        try:
            user_query = input("\n🟢 You: ")
            if user_query.lower() == 'exit':
                print("👋 Goodbye!")
                sys.exit(0)
            response = chatbot_interaction(user_query)
            print(f"🤖 AKUgpt: {response}")
        except KeyboardInterrupt:
            print("\n👋 Exiting chatbot. Goodbye!")
            sys.exit(0)
