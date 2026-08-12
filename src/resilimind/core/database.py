import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

# Define the path for the SQLite database file within the 'data' directory
DB_PATH: Path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "resilimind.db"

def init_db() -> None:
    """
    Initializes the SQLite database and creates necessary tables ('users' and 'resilience_logs')
    if they do not exist. Ensures parent directories exist prior to database connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        
        # Create users table for authentication
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create resilience_logs table for tracking node status history per user
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resilience_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                node_id TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
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


def save_resilience_log(
    user_id: int, 
    node_id: str, 
    status: str, 
    confidence: float, 
    reasoning: Optional[str] = None
) -> None:
    """
    Saves a single resilience assessment evaluation log for a specific user.

    Args:
        user_id (int): The ID of the authenticated user.
        node_id (str): The graph node identifier (e.g., 'IND_ECO_01').
        status (str): The evaluated status ('GREEN', 'YELLOW', 'RED').
        confidence (float): The confidence score assigned by the Assessor Agent.
        reasoning (Optional[str]): The underlying reasoning text provided by the LLM.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO resilience_logs (user_id, node_id, status, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, node_id, status.upper(), confidence, reasoning)
        )
        conn.commit()


def get_user_resilience_history(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves the recent resilience evaluation logs for a specific user sorted by timestamp.

    Args:
        user_id (int): The ID of the user whose logs are being queried.
        limit (int): Maximum number of log entries to return (default: 20).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing log entry details.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            SELECT node_id, status, confidence, reasoning, created_at
            FROM resilience_logs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_user_latest_node_statuses(user_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent resilience status log for each unique active node for a specific user.

    Args:
        user_id (int): The ID of the user.

    Returns:
        List[Dict[str, Any]]: List of dictionary items representing the latest log per node.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            SELECT rl.node_id, rl.status, rl.confidence, rl.reasoning, rl.created_at
            FROM resilience_logs rl
            INNER JOIN (
                SELECT node_id, MAX(created_at) as max_date
                FROM resilience_logs
                WHERE user_id = ?
                GROUP BY node_id
            ) latest ON rl.node_id = latest.node_id AND rl.created_at = latest.max_date
            WHERE rl.user_id = ?
            ORDER BY rl.created_at DESC
            """,
            (user_id, user_id)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
