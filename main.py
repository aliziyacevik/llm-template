from fastapi import FastAPI
from pydantic import BaseModel
from model import get_response

app = FastAPI()

class UserMessage(BaseModel):
    message: str

@app.post("/chat/")
async def chat(user_message: UserMessage):
    response = get_response(user_message.message)
    return {"response": response}

