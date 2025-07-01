from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

# Initialize FastAPI app
app = FastAPI()

# Load open-source chatbot model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Input message format
class Message(BaseModel):
    message: str

# Health check endpoint
@app.get("/")
def root():
    return {"message": "Chatbot is running"}

# Chat endpoint
@app.post("/chat")
def chat(user_input: Message):
    result = chatbot(user_input.message, max_length=100, pad_token_id=50256)
    return {"response": result[0]["generated_text"]}

# Run app if this file is executed directly (optional for local dev)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
