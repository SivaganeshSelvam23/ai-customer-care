from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.schema import User
from backend.db.database import SessionLocal
from backend.auth import hash_password, verify_password, create_jwt_token

router = APIRouter()

class RegisterRequest(BaseModel):
    name: str
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/register")
def register(payload: RegisterRequest):
    db: Session = SessionLocal()
    try:
        if db.query(User).filter_by(username=payload.username).first():
            raise HTTPException(status_code=400, detail="Username already exists")
        user = User(
            name=payload.name,
            username=payload.username,
            password=hash_password(payload.password),
            role="customer"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"message": "User registered successfully"}
    finally:
        db.close()

@router.post("/login")
def login(payload: LoginRequest):
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter_by(username=payload.username).first()
        if not user or not verify_password(payload.password, user.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_jwt_token(user.id, user.role)
        return {
            "token": token,
            "user": {"id": user.id, "name": user.name, "role": user.role}
        }
    finally:
        db.close()

@router.get("/{user_id}")
def get_user(user_id: int):
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"id": user.id, "name": user.name, "role": user.role}
    finally:
        db.close()
