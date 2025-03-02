from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize Flask app
app = Flask(__name__)

# Sample documents (replace with your own data)
documents = [
    "SQL (Structured Query Language) is used to manage and manipulate relational databases.",
    "Relational databases store data in tables with rows and columns, enforcing schema-based structure.",
    "Popular SQL databases include MySQL, PostgreSQL, SQLite, Oracle, and Microsoft SQL Server.",
    "SQL databases use ACID (Atomicity, Consistency, Isolation, Durability) properties to ensure reliable transactions.",
    "SQL queries use commands like SELECT, INSERT, UPDATE, and DELETE to interact with data.",
    "SQL is ideal for applications needing complex queries, transactions, and data integrity."
]

# Step 1: Embed documents using a retriever model
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
document_embeddings = retriever.encode(documents, convert_to_tensor=True).cpu().numpy()

# Step 2: Build a FAISS index for efficient search
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Step 3: Define a retriever function
def retrieve_documents(query, k=2):
    query_embedding = retriever.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Step 4: Load a generator model (e.g., GPT-2)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
generator = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 5: Generate an answer using retrieved context
def generate_answer(query, max_length=100):
    # Retrieve relevant documents
    context_docs = retrieve_documents(query)
    context = " ".join(context_docs)
    
    # Create prompt with context and query
    prompt = f"Answer concisely based on the context below.\nContext: {context}\nQuestion: {query}\nAnswer:"
    
    # Generate answer
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0,  # Avoid repetitive output
        no_repeat_ngram_size=1    # Prevent repeated phrases
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle queries
@app.route('/ask', methods=['POST'])
def ask():
    # Get the query from the request
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Generate the answer
    answer = generate_answer(query)
    
    # Return the answer as JSON
    return jsonify({"query": query, "answer": answer})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)