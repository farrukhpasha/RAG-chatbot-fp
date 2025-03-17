import ollama

# Step 1: Load the text file and split it into paragraphs
def load_dataset(file_path):
    """Load the dataset from a text file and split it into paragraphs."""
    with open(file_path, 'r') as file:
        content = file.read()  # Read the entire content of the file
    paragraphs = content.split('\n\n')  # Split into paragraphs by double newlines
    print(f'Loaded {len(paragraphs)} paragraphs')
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]  # Remove empty paragraphs and extra spaces


# Step 2: Define the Vector Database and Embedding Model
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []  # Store tuples of (paragraph, embedding)


def add_paragraph_to_database(paragraph):
    """Generates an embedding for the paragraph and adds it to the vector database."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=paragraph)
    embedding = response['embeddings'][0]  # Extract the first embedding
    VECTOR_DB.append((paragraph, embedding))


# Step 3: Cosine Similarity Function
def cosine_similarity(a, b):
    """Calculates cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


# Step 4: Retrieve Relevant Paragraphs
def retrieve(query, top_n=3):
    """Retrieves the top N most relevant paragraphs based on cosine similarity."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    query_embedding = response['embeddings'][0]
    similarities = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in VECTOR_DB
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity in descending order
    return similarities[:top_n]


# Step 5: Chatbot Interaction
def chatbot_interaction():
    """Handles chatbot interaction based on user input."""
    input_query = input('Ask me a question: ')
    retrieved_knowledge = retrieve(input_query)

    print('\nRetrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = (
        f"You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. Don't make up any new information:\n"
        + '\n'.join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
    )

    print('\nChatbot response:')
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)


# Step 6: Main Functionality
def main():
    """Main function to load dataset, process paragraphs, and interact with the chatbot."""
    file_path = 'Annual Leaves.txt'  # Path to the text file
    dataset = load_dataset(file_path)

    # Add paragraphs to the vector database
    for i, paragraph in enumerate(dataset):
        add_paragraph_to_database(paragraph)
        print(f'Added paragraph {i + 1}/{len(dataset)} to the database')

    # Start chatbot interaction
    chatbot_interaction()


if __name__ == "__main__":
    main()
