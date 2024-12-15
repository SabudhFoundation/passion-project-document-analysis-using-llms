import sqlite3
import gradio as gr
import pytz
import bcrypt

current_user_id = None  # Global variable for logged-in user ID


def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)


# Function to register a new user
def register_user(username, password):
    conn = sqlite3.connect('chatbot_app.db')
    cursor = conn.cursor()
    
    # Check if username already exists
    cursor.execute("SELECT * FROM user WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return "Error: Username already exists."
    
    # Hash password and store
    hashed_password = hash_password(password)
    cursor.execute("INSERT INTO user (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()
    return "Registration successful! Please log in."


def handle_login(username, password, login_tab, register_tab, main_app_row):
    global current_user_id
    conn = sqlite3.connect('chatbot_app.db')
    cursor = conn.cursor()
    
    # Check if the user exists
    cursor.execute("SELECT user_id, password FROM user WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return "Error: User not found.", login_tab, register_tab, main_app_row

    user_id, hashed_password = user
    if verify_password(password, hashed_password):  # Assume verify_password is defined elsewhere
        current_user_id = user_id  # Set logged-in user ID
        print(f"User {username} logged in with ID {user_id}.")
        return "Login successful!", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    else:
        return "Error: Incorrect password.", login_tab, register_tab, main_app_row
    
def fetch_user_history():
    global current_user_id
    if not current_user_id:
        return "Error: No user is logged in. Cannot fetch history."
    
    conn = sqlite3.connect('chatbot_app.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, query, response
        FROM user_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
    """, (current_user_id,))
    results = cursor.fetchall()
    conn.close()

    if not results:
        return "No history found for this user."

    # Format history display
    return "\n\n".join([
        f"{timestamp}\nQuery: {query}\nResponse: {response}"
        for timestamp, query, response in results
    ])


# Function to log user interactions
def log_user_history(query, response):
    global current_user_id
    if not current_user_id:
        return "Error: No user is logged in. Cannot store history."

    conn = sqlite3.connect('chatbot_app.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_history (user_id, timestamp, query, response)
        VALUES (?, datetime('now'), ?, ?)
    """, (current_user_id, query, response))
    conn.commit()
    conn.close()
    return "History logged successfully."

correct Credentials", gr.update(), gr.update(), gr.update()




