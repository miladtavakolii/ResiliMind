import sqlite3
import logging
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

# Define the path for the SQLite database file
DB_PATH: Path = settings.user_db_path


def get_connection() -> sqlite3.Connection:
    """Creates a new SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """
    Initializes the SQLite database and creates necessary tables ('users' and 'resilience_logs')
    if they do not exist. Ensures parent directories exist prior to database connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[Database] Initializing database at %s", DB_PATH)
    
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
        logger.error("[Database] Failed to initialize tables: %s", e)
        raise
