from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Define a Pydantic model to validate the request payload
class Message(BaseModel):
    message: str

# Load the Hugging Face model
chatbot = pipeline("conversational")

@app.post("/chat")
async def chat(message: Message):
    user_message = message.message
    # Get the chatbot response
    response = chatbot(user_message)
    return {"response": response[0]['generated_text']}


