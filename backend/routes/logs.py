from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from backend.db.database import SessionLocal
from backend.db.schema import AgentLog

router = APIRouter(tags=["Logs"])

@router.get("/agent/{agent_id}")
def get_agent_logs(agent_id: int):
    db: Session = SessionLocal()
    try:
        logs = db.query(AgentLog).filter_by(agent_id=agent_id).order_by(AgentLog.start_time.desc()).all()
        return [
            {
                "customer_name": log.customer_name,
                "session_id": log.session_id,
                "start_time": log.start_time,
                "end_time": log.end_time,
                "outcome": log.outcome,
                "ner_summary": log.ner_summary,
                "messages": log.messages,
            } for log in logs
        ]
    finally:
        db.close()
