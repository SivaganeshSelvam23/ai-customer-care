import bcrypt
import jwt
import time
from fastapi import HTTPException

SECRET_KEY = "your-very-secret-key"
ALGORITHM = "HS256"
TOKEN_EXPIRE_SEC = 3600  # 1 hour

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_jwt_token(user_id: int, role: str):
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": time.time() + TOKEN_EXPIRE_SEC
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_jwt_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
