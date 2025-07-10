from fastapi import FastAPI
from backend.routes.sentiment import router as sentiment_router

app = FastAPI(
    title="AI-Powered Customer Care",
    description="Real-time sentiment analysis API using RoBERTa",
    version="1.0.0"
)

# sentiment prediction routes
app.include_router(sentiment_router, prefix="/api/sentiment")
