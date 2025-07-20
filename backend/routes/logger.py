import json
from backend.db.schema import AgentLog, User
from backend.db.database import SessionLocal

def log_agent_session_summary(agent_id, customer_id, session_id, outcome, messages, ner_spans, start_time, end_time):
    db = SessionLocal()
    try:
        agent = db.query(User).filter(User.id == agent_id).first()
        customer = db.query(User).filter(User.id == customer_id).first()

        ner_summary = ", ".join([f"{ner['label']} \"{ner['text']}\"" for ner in ner_spans])
        messages_json = json.dumps(messages, ensure_ascii=False)

        log = AgentLog(
            agent_id=agent.id,
            agent_name=agent.name,
            customer_id=customer.id,
            customer_name=customer.name,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            outcome=outcome,
            ner_summary=ner_summary,
            messages=messages_json
        )
        db.add(log)
        db.commit()
    finally:
        db.close()
