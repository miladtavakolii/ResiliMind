import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash

from .config import settings

# Initialize module logger
logger = logging.getLogger(__name__)

# Initialize Argon2 Password Hasher (OWASP recommended standard)
ph = PasswordHasher()

# Define the path for the SQLite database file
DB_PATH: Path = settings.user_db_path

def init_db() -> None:
    """
    Initializes the SQLite database and creates necessary tables ('users' and 'resilience_logs')
    if they do not exist. Ensures parent directories exist prior to database connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Database] Initializing database at {DB_PATH}")
    
    try:
        with get_connection() as conn:
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
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resilience_logs_user_created
                ON resilience_logs(user_id, created_at DESC)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resilience_logs_user_node_created
                ON resilience_logs(user_id, node_id, created_at DESC)
                """
            )
            
            conn.commit()
            logger.debug("[Database] Tables 'users' and 'resilience_logs' verified/created.")
    except sqlite3.Error as e:
        logger.error(f"[Database] Failed to initialize tables: {e}")
        raise

def get_connection() -> sqlite3.Connection:
    conn = get_connection()
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def register_user(username: str, password: str) -> bool:
    """
    Registers a new user in the database using secure Argon2id password hashing.

    Args:
        username (str): The desired username.
        password (str): The plain-text password.

    Returns:
        bool: True if the registration is successful, False if the username already exists.
    """
    try:
        hashed_password = ph.hash(password)
        with get_connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username.strip(), hashed_password)
            )
            conn.commit()
        logger.info(f"[Database] Successfully registered user: {username}")
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"[Database] Registration failed: Username '{username}' already exists.")
        return False
    except sqlite3.Error as e:
        logger.error(f"[Database] Registration error for '{username}': {e}")
        return False


def authenticate_user(username: str, password: str) -> Optional[int]:
    """
    Authenticates a user by securely verifying their password against the stored Argon2 hash.

    Args:
        username (str): The provided username.
        password (str): The provided plain-text password.

    Returns:
        Optional[int]: The user's ID if authentication is successful, None otherwise.
    """
    with get_connection() as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username.strip(),)
        )
        result: Optional[tuple] = cursor.fetchone()
        
        if result:
            user_id, stored_hash = result
            try:
                if ph.verify(stored_hash, password):
                    logger.info(f"[Database] User '{username}' authenticated successfully.")
                    return user_id
            except (VerifyMismatchError, InvalidHash):
                logger.warning(f"[Database] Password mismatch for user: {username}")
        else:
            logger.warning(f"[Database] Authentication attempt failed: User '{username}' not found.")
            
        return None


def save_resilience_log(
    user_id: int, 
    node_id: str, 
    category: str,
    status: str, 
    score: int,
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
    try:
        with get_connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO resilience_logs (user_id, node_id, category, status, score, confidence, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, node_id, category, status.upper(), score, confidence, reasoning)
            )
            conn.commit()
        logger.debug(f"[Database] Saved resilience log for user {user_id} on node {node_id}.")
    except sqlite3.Error as e:
        logger.error(f"[Database] Failed to save resilience log for user {user_id}: {e}")


def get_user_resilience_history(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves the recent resilience evaluation logs for a specific user sorted by timestamp.

    Args:
        user_id (int): The ID of the user whose logs are being queried.
        limit (int): Maximum number of log entries to return (default: 20).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing log entry details.
    """
    with get_connection() as conn:
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
        logger.debug(f"[Database] Retrieved {len(rows)} history logs for user {user_id}.")
        return [dict(row) for row in rows]


def get_user_latest_node_statuses(user_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent resilience status log for each unique active node for a specific user.

    Args:
        user_id (int): The ID of the user.

    Returns:
        List[Dict[str, Any]]: List of dictionary items representing the latest log per node.
    """
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            SELECT node_id, category, status, score, confidence, reasoning, created_at
            FROM (
                SELECT node_id, category, status, score, confidence, reasoning, created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id, node_id
                        ORDER BY created_at DESC, id DESC
                    ) AS row_num
                FROM resilience_logs
                WHERE user_id = ?
            )
            WHERE row_num = 1
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        logger.debug(f"[Database] Retrieved {len(rows)} latest node statuses for user {user_id}.")
        return [dict(row) for row in rows]

def get_user_node_timeline(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetches the chronological history of node assessments for a user
    to build a true temporal memory for the Advisor Agent.

    Args:
        user_id (int): The ID of the user.
        limit (int): Maximum number of log entries to return (default: 50).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing chronological log entries.
    """
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT node_id, status, score, created_at
            FROM resilience_logs
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = list(reversed(cursor.fetchall()))
        logger.debug(f"[Database] Retrieved {len(rows)} timeline records for user {user_id}.")
        return [dict(row) for row in rows]
