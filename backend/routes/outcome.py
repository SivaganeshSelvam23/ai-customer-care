from fastapi import APIRouter
from pydantic import BaseModel
from backend.utils.infer_outcome import predict_outcome

router = APIRouter()

class DialogRequest(BaseModel):
    dialog: list[str]  # List of turns like ["Customer: ...", "Agent: ..."]

@router.post("/predict")
def get_outcome(payload: DialogRequest):
    predicted = predict_outcome(payload.dialog)
    return {"outcome": predicted}
