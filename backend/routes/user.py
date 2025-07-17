from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from backend.db.schema import User
from backend.db.database import SessionLocal
from backend.auth import hash_password, verify_password, create_jwt_token

router = APIRouter()

@router.post("/register")
def register(name: str, username: str, password: str):
    db: Session = SessionLocal()
    try:
        if db.query(User).filter_by(username=username).first():
            raise HTTPException(status_code=400, detail="Username already exists")
        user = User(name=name, username=username, password=hash_password(password), role="customer")
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"message": "User registered successfully"}
    finally:
        db.close()

@router.post("/login")
def login(username: str, password: str):
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter_by(username=username).first()
        if not user or not verify_password(password, user.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_jwt_token(user.id, user.role)
        return {
            "token": token,
            "user": {"id": user.id, "name": user.name, "role": user.role}
        }
    finally:
        db.close()
