import sqlite3

conn = sqlite3.connect("backend/db/users.db")
cur = conn.cursor()
cur.execute("DELETE FROM chat_sessions;")
cur.execute("DELETE FROM sqlite_sequence WHERE name='chat_sessions';")
conn.commit()
conn.close()
