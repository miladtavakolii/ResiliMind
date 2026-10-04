import sqlite3
import logging
from typing import Optional
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash

from .connection import get_connection

logger = logging.getLogger(__name__)

# Initialize Argon2 Password Hasher (OWASP recommended standard)
ph = PasswordHasher()


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
        logger.info("[Database] Successfully registered user: %s", username)
        return True
    except sqlite3.IntegrityError:
        logger.warning("[Database] Registration failed: Username '%s' already exists.", username)
        return False
    except sqlite3.Error as e:
        logger.error("[Database] Registration error for '%s': %s", username, e)
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
                    logger.info("[Database] User '%s' authenticated successfully.", username)
                    return user_id
            except (VerifyMismatchError, InvalidHash):
                logger.warning("[Database] Password mismatch for user: %s", username)
        else:
            logger.warning("[Database] Authentication attempt failed: User '%s' not found.", username)
            
        return None
