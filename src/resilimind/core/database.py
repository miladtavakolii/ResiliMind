import sqlite3
import hashlib
from pathlib import Path
from typing import Optional

# Define the path for the SQLite database file within the 'data' directory
DB_PATH: Path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "resilimind.db"

def init_db() -> None:
    """
    Initializes the SQLite database and creates the 'users' table if it does not exist.
    Ensures the parent directories exist before creating the database file.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def _hash_password(password: str) -> str:
    """
    Hashes a plain-text password using the SHA-256 algorithm.

    Args:
        password (str): The plain-text password provided by the user.

    Returns:
        str: The SHA-256 hashed password string.
    """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username: str, password: str) -> bool:
    """
    Registers a new user in the database.

    Args:
        username (str): The desired username.
        password (str): The plain-text password.

    Returns:
        bool: True if the registration is successful, False if the username already exists.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username.strip(), _hash_password(password))
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Triggered if the username violates the UNIQUE constraint in the database
        return False

def authenticate_user(username: str, password: str) -> Optional[int]:
    """
    Authenticates a user by verifying their username and password match in the database.

    Args:
        username (str): The provided username.
        password (str): The provided plain-text password.

    Returns:
        Optional[int]: The user's ID if authentication is successful, None otherwise.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE username = ? AND password_hash = ?",
            (username.strip(), _hash_password(password))
        )
        result: Optional[tuple] = cursor.fetchone()
        return result[0] if result else None
