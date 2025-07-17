import sqlite3
import bcrypt
from datetime import datetime

users = [
    ("Admin", "admin", "admin123", "admin"),
    ("Agent One", "agent1", "agent123", "agent"),
    ("Agent Two", "agent2", "agent123", "agent"),
    ("Agent Three", "agent3", "agent123", "agent")
]

def hash_pw(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

conn = sqlite3.connect("data/users.db")
cur = conn.cursor()

for name, username, pw, role in users:
    hashed = hash_pw(pw)
    cur.execute("""
        INSERT INTO users (name, username, password, role, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (name, username, hashed, role, datetime.now()))

conn.commit()
conn.close()
print("✅ Seeded admin and agents.")
