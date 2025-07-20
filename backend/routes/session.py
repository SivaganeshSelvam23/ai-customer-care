from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import random

from backend.db.database import SessionLocal
from backend.db.schema import User, ChatSession, Message
from app.sentiment_engine import predict_emotion

from backend.routes.logger import log_agent_session_summary
from backend.db.schema import Message


router = APIRouter(tags=["Session"])

@router.post("/start-session")
def start_session(customer_id: int):
    db: Session = SessionLocal()
    try:
        existing = db.query(ChatSession).filter_by(customer_id=customer_id, status="active").first()
        if existing:
            agent = db.query(User).filter_by(id=existing.agent_id).first()
            return {
                "session_id": existing.id,
                "agent_id": agent.id,
                "agent_name": agent.name,
                "message": "Existing session reused"
            }

        agents = db.query(User).filter_by(role="agent").all()
        if not agents:
            raise HTTPException(status_code=400, detail="No agents available")

        agent = random.choice(agents)
        new_session = ChatSession(
            customer_id=customer_id,
            agent_id=agent.id,
            status="active"
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)

        return {
            "session_id": new_session.id,
            "agent_id": agent.id,
            "agent_name": agent.name,
            "message": "New session started"
        }
    finally:
        db.close()


@router.get("/assigned/{agent_id}")
def get_assigned_sessions(agent_id: int):
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).filter_by(agent_id=agent_id, status="active").all()
        result = []
        for s in sessions:
            customer = db.query(User).filter_by(id=s.customer_id).first()
            result.append({
                "session_id": s.id,
                "customer_id": s.customer_id,
                "customer_name": customer.name,
                "started_at": s.created_at.strftime("%Y-%m-%d %H:%M"),
            })
        return result
    finally:
        db.close()

@router.post("/end-session")
def end_session(session_id: int, customer_id: int):
    db: Session = SessionLocal()
    try:
        session = db.query(ChatSession).filter_by(id=session_id, status="active").first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or already ended")

        if session.customer_id != customer_id:
            raise HTTPException(status_code=403, detail="Only the customer can end this session")

        # Get messages before deletion
        messages = db.query(Message).filter_by(session_id=session_id).order_by(Message.timestamp).all()

        # Prepare message log
        message_log = [
            {
                "text": m.text,
                "sender": m.sender,
                "emotion": m.emotion
            } for m in messages
        ]

        # Log the session to agent_logs
        log_agent_session_summary(
            agent_id=session.agent_id,
            customer_id=customer_id,
            session_id=session.id,
            outcome="resolved",  # or from model if dynamic
            start_time=session.created_at.strftime("%Y-%m-%d %H:%M"),
            end_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            messages=message_log,
            ner_spans=[]  # You can populate this later
        )

        # Delete messages and end session
        db.query(Message).filter_by(session_id=session_id).delete()
        session.status = "ended"
        db.commit()

        return {"message": "Session ended, logged, and messages deleted"}
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
        emotion = predict_emotion(payload.text)
        new_msg = Message(
            session_id=payload.session_id,
            sender=payload.sender,
            text=payload.text,
            emotion=emotion,
            timestamp=datetime.utcnow()
        )
        db.add(new_msg)
        db.commit()
        db.refresh(new_msg)
        return {"message": "Message sent", "emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/messages/{session_id}")
def get_messages(session_id: int):
    db = SessionLocal()
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
