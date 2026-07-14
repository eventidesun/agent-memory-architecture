from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import get_response
from logger import init_log

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_log()

class MessageRequest(BaseModel):
    participant_id: str
    session: int
    condition: str
    message: str
    room: int = 1
    first_message: bool = False

@app.post("/chat")
def chat(request: MessageRequest):
    response = get_response(
        participant_id=request.participant_id,
        session=request.session,
        condition=request.condition,
        user_message=request.message,
        room=request.room,
        first_message=request.first_message
    )
    return {"response": response}

@app.get("/health")
def health():
    return {"status": "ok"}