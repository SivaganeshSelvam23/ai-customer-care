from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.sentiment import router as sentiment_router
from backend.routes.user import router as user_router
from backend.routes.session import router as session_router
from backend.routes.outcome import router as outcome_router
from backend.routes.logs import router as logs_router
from backend.db.schema import Base
from backend.db.database import engine

Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="AI-Powered Customer Care",
    description="Real-time sentiment analysis + chat session API",
    version="1.0.0"
)

# 🌐 Enable CORS for Streamlit frontend
origins = ["*"]  # You can restrict this later to localhost or Streamlit Cloud URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Include all routes
app.include_router(sentiment_router, prefix="/api/sentiment")
app.include_router(outcome_router, prefix="/api/outcome")
app.include_router(user_router, prefix="/api/user")
app.include_router(session_router, prefix="/api/session")
app.include_router(logs_router, prefix="/api/logs")
@app.get("/")
def root():
    return {"message": "AI Customer Care API is live"}
