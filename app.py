import os
import json
import pathlib
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

#use environment variable for credentials
def load_credentials():
    #try to load credentials from environment variable
    credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if credentials_json:
        #write credentials to a temporary file
        credentials_path = pathlib.Path(__file__).parent / 'temp_credentials.json'
        with open(credentials_path, 'w') as f:
            f.write(credentials_json)
        return str(credentials_path)
    return None

#set up credentials
credentials_path = load_credentials()
if credentials_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

app = Flask(__name__)
app.secret_key = os.urandom(24)  #session management
CORS(app, supports_credentials=True)  #enable CORS with credentials support

#store conversations by session ID
conversation_store = {}

#function to communicate with Google Vertex AI
def generate_response(user_message, conversation_id):
    client = genai.Client(
        vertexai=True,
        project="typ-23-03",
        location="europe-west4",
    )
    
    model = "gemini-2.0-flash-001"
    
    #system instruction for child-friendly assistant
    si_text = """You are a child-friendly and helpful assistant. You should identify instances of gender and racial bias in conversations and provide age-appropriate feedback that encourages inclusivity. When a bias is detected, respond by explaining why it is wrong and offer alternative inclusive suggestions. Begin by asking appropriate questions that may result in a biased response from children (implicit or explicit). After 4 questions, ask a few final questions to see if the child's bias (if any) has decreased after reading your responses. Avoid using more complex language and sentences, since the responses will be read by children ages 5-10."""
    
    #get or initialize conversation history
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = []
    
    conversation_history = conversation_store[conversation_id]
    
    #create content with conversation history and current user message
    contents = []
    
    #add conversation history
    for entry in conversation_history:
        if entry["role"] == "user":
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=entry["text"])]
            ))
        else:  # Model response
            contents.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=entry["text"])]
            ))
    
    #add current user message
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)]
    ))
    
    #configure generation parameters
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=1024,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            )
        ],
        system_instruction=[types.Part.from_text(text=si_text)],
    )
    
    #generate response
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    #update conversation history
    conversation_store[conversation_id].append({"role": "user", "text": user_message})
    conversation_store[conversation_id].append({"role": "assistant", "text": response.text})
    
    #keep conversation history to a reasonable length (last 10 exchanges)
    if len(conversation_store[conversation_id]) > 20:
        conversation_store[conversation_id] = conversation_store[conversation_id][-20:]
    
    return response.text

#API endpoint to handle requests from react frontend
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    conversation_id = request.json.get("conversationId", None)
    
    # Create new conversation ID if not provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    ai_response = generate_response(user_input, conversation_id)
    
    return jsonify({
        "response": ai_response,
        "conversationId": conversation_id
    })

#cleanup for temporary credentials file
@app.teardown_appcontext
def cleanup_credentials(exception=None):
    credentials_path = pathlib.Path(__file__).parent / 'temp_credentials.json'
    if credentials_path.exists():
        os.unlink(credentials_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

