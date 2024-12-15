import sqlite3
import bcrypt
from datetime import datetime

# Initialize the SQLite database connection
conn = sqlite3.connect('chatbot_app.db')
cursor = conn.cursor()
def history_table():
    try:
        conn = sqlite3.connect('chatbot_app.db')
        cursor = conn.cursor()
        # Create a table for storing user history if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                response TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("Database and table initialized successfully.")
    except Exception as e:
        print(f"Error initializing the database: {e}")
def user_table():
    try:
        conn = sqlite3.connect('chatbot_app.db')
        cursor = conn.cursor()
        # Create a table for storing user data if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''') 
        conn.commit()
        conn.close()
        print("User database and table initialized successfully.")
    except Exception as e:
        print(f"Error initializing the user database: {e}")
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
