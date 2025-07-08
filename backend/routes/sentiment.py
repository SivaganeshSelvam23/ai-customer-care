from fastapi import APIRouter
from pydantic import BaseModel
from backend.utils.inference import predict_sentiment

router = APIRouter()

class Message(BaseModel):
    text: str

@router.post("/predict")
def get_sentiment(message: Message):
    result = predict_sentiment(message.text)
    return {"sentiment": result}
