# This script is a Flask application that 
# uses OpenAI's API to extract structured information from 
# user queries related to tourism in Montreal.
# the extracted information is used for planning activities

from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

custom_prompt = """
You are a travel assistant for Montreal tourism. If the question is unrelated to Montreal tourism, respond:
I only understand Montreal tourism. 

The user is asking about tourism activities. Your task is to extract structured information from their message and fill in the following fields. If a field is not mentioned, leave it blank.

Respond only in JSON format like this:
{
  "location": {
    "city": "",
    "street": "",
    "from_location": "",
    "to_location": ""
  },
  "time": {
    "date_from": "",
    "date_to": "",
    "hour_from": "",
    "hour_to": "",
    "duration": ""
  },
  "budget": "",
  "activity_type": []
}

Field Descriptions:
 - "location": 
  - "city": name of the city if mentioned.
  - "street": street name or local area if specified.
  - "from_location" and "to_location": if the user mentions a route or movement.
 - "time": 
  - "date_from" and "date_to": specific date range if mentioned.
  - "hour_from" and "hour_to": time of day.
  - "duration": duration of the activity (e.g., "3 hours", "half day").
 - "budget": classify as "economic", "normal", or "luxury" based on language (e.g., “cheap” → economic, “mid-range” → normal, “high-end” → luxury).
 - "activity_type": identify one or more of the following: "transport", "sightseeing", "entertainment", "accommodation", "meals".



User Input:
"""
@app.route('/', methods=['GET'])
def hello_world():
    return "Hello, World!"
@app.route('/ask', methods=['POST'])
def ask_openai():
    data = request.get_json()
    question = data.get("question", "")

    try:
        full_prompt = custom_prompt + question.strip()
        response = client.chat.completions.create(
            model="gpt-4",  # gpt-3.5-turbo You can also try "" if available
            messages=[{"role": "user", "content": full_prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


# Field Descriptions:
# - "location": 
#  - "city": name of the city if mentioned.
#  - "street": street name or local area if specified.
#  - "from_location" and "to_location": if the user mentions a route or movement.
# - "time": 
#  - "date_from" and "date_to": specific date range if mentioned.
#  - "hour_from" and "hour_to": time of day.
#  - "duration": duration of the activity (e.g., "3 hours", "half day").
# - "budget": classify as "economic", "normal", or "luxury" based on language (e.g., “cheap” → economic, “mid-range” → normal, “high-end” → luxury).
# - "activity_type": identify one or more of the following: "transport", "sightseeing", "entertainment", "accommodation", "meals".
