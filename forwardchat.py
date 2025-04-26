# this node.js code forward the question from the front end to OpenAI API 
# and get the answer. 

from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)  # allow requests from frontend

  # make sure you set this in .env or your terminal

@app.route('/ask', methods=['POST'])
def ask_openai():
    data = request.get_json()
    question = data.get("question", "")

    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "user", "content": question}
        ])
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
