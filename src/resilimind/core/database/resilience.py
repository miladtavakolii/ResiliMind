import sqlite3
import logging
from typing import Optional, List, Dict, Any

from .connection import get_connection

logger = logging.getLogger(__name__)


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
        logger.debug("[Database] Saved resilience log for user %s on node %s.", user_id, node_id)
    except sqlite3.Error as e:
        logger.exception("[Database] Failed to save resilience log for user %s", user_id)
        raise


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
        logger.debug("[Database] Retrieved %s history logs for user %s.", len(rows), user_id)
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
        logger.debug("[Database] Retrieved %s latest node statuses for user %s.", len(rows), user_id)
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
        logger.debug("[Database] Retrieved %s timeline records for user %s.", len(rows), user_id)
        return [dict(row) for row in rows]
