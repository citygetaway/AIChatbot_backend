# this is the main file for the RAG system, 
# processing the user input and returning the answer
# using the knowledge in vector database as the context
# --- backend/app.py ---
from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
from embeddings import get_relevant_passages

import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

CORS(app)  # allow requests from frontend


@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')

    if not question:
        return jsonify({"error": "Missing question"}), 400

    # Get relevant passages
    context = get_relevant_passages(question)
    # print(f"Context: {context}")

    # Build prompt
    prompt = f"""
    You are a helpful assistant for Montreal tourism. Answer questions based only on the context below.
    If the question is unrelated to Montreal tourism, respond:
    'I only understand Montreal tourism.'

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "user", "content": prompt}])

    return jsonify({"answer": response.choices[0].message.content.strip()})

if __name__ == '__main__':
    app.run(debug=True)
