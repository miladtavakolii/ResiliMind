from __future__ import annotations

import json
import time
import logging
from typing import Any

from google import genai


logger = logging.getLogger(__name__)


class GeminiJudge:
    """
    Gemini based evaluator for qualitative response assessment.

    Uses a separate LLM call to judge advisor responses.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        retries: int = 5,
        delay: float = 3.0,
    ) -> None:

        self.client = genai.Client(
            api_key=api_key,
        )

        self.model = model
        self.retries = retries
        self.delay = delay


    def evaluate(
        self,
        prompt: str,
    ) -> dict[str, Any]:

        for attempt in range(
            1,
            self.retries + 1,
        ):
            try:

                response = (
                    self.client
                    .models
                    .generate_content(
                        model=self.model,
                        contents=prompt,
                        config={
                            "temperature": 0.1,
                        },
                    )
                )

                return json.loads(
                    response.text
                )

            except Exception as exc:

                logger.warning(
                    "Judge failed attempt %s/%s: %s",
                    attempt,
                    self.retries,
                    exc,
                )

                if attempt < self.retries:
                    time.sleep(
                        self.delay * attempt
                    )

        raise RuntimeError(
            "Gemini judge failed after retries"
        )
