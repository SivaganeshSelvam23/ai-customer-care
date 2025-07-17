from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import random

from backend.db.database import SessionLocal
from backend.db.schema import User, ChatSession, Message
from app.sentiment_engine import predict_emotion

router = APIRouter(tags=["Session"])

@router.post("/start-session")
def start_session(customer_id: int):
    db: Session = SessionLocal()
    try:
        agents = db.query(User).filter_by(role="agent").all()
        if not agents:
            raise HTTPException(status_code=400, detail="No agents available")

        agent = random.choice(agents)
        new_session = ChatSession(customer_id=customer_id, agent_id=agent.id)
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return {
            "session_id": new_session.id,
            "agent_id": agent.id,
            "message": "Session started"
        }
    finally:
        db.close()

@router.post("/end-session")
def end_session(session_id: int):
    db: Session = SessionLocal()
    try:
        session = db.query(ChatSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.status == "ended":
            return {"message": "Session already ended"}

        session.status = "ended"
        db.commit()
        return {"message": "Session ended"}
    finally:
        db.close()

class MessageInput(BaseModel):
    session_id: int
    sender_id: int
    sender: str
    text: str

@router.post("/send")
def send_message(payload: MessageInput):
    db = SessionLocal()
    try:
        print(f"➡️ Received message: {payload.text}")
        emotion = predict_emotion(payload.text)
        print(f"🔍 Predicted emotion: {emotion}")

        new_msg = Message(
            session_id=payload.session_id,
            sender=payload.sender,
            emotion=emotion,
            timestamp=datetime.utcnow()
        )
        # Manually add dynamic column `text` and `sender_id` if present
        setattr(new_msg, "text", payload.text)
        setattr(new_msg, "sender_id", payload.sender_id)

        db.add(new_msg)
        db.commit()
        db.refresh(new_msg)
        print(f"✅ Message saved: {new_msg.id}")
        return {"message": "Message sent", "emotion": emotion}
    except Exception as e:
        print(f"❌ Error in /send: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/messages/{session_id}")
def get_messages(session_id: int):
    db: Session = SessionLocal()
    try:
        messages = db.query(Message).filter_by(session_id=session_id).order_by(Message.timestamp).all()
        return [
            {
                "id": m.id,
                "sender": m.sender,
                "text": m.text,
                "emotion": m.emotion,
                "timestamp": m.timestamp.strftime("%Y-%m-%d %H:%M"),
                "sender_id": getattr(m, "sender_id", None)
            } for m in messages
        ]
    finally:
        db.close()
