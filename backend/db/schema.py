from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from datetime import datetime
from sqlalchemy.sql import func
from backend.db.database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    username = Column(String, unique=True)
    password = Column(String)
    role = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("users.id"))
    agent_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))  # fixed FK
    sender = Column(String)
    text = Column(Text)
    emotion = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
