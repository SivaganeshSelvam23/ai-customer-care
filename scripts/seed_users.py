import os
import sqlite3
import bcrypt
from datetime import datetime

# Ensure correct folder exists
os.makedirs("backend/db", exist_ok=True)

users = [
    ("Admin", "admin", "admin123", "admin"),
    ("Agent One", "agent1", "agent123", "agent"),
    ("Agent Two", "agent2", "agent123", "agent"),
    ("Agent Three", "agent3", "agent123", "agent")
]

def hash_pw(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

# 🔧 Correct DB path
conn = sqlite3.connect("backend/db/users.db")
cur = conn.cursor()

# Create users table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT,
    created_at TEXT
)
""")

# Insert users
for name, username, pw, role in users:
    hashed = hash_pw(pw)
    cur.execute("""
        INSERT OR IGNORE INTO users (name, username, password, role, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (name, username, hashed, role, datetime.now()))

conn.commit()
conn.close()
print("✅ Seeded admin and agents.")
