"""
Base class for notification providers.
"""
from abc import ABC, abstractmethod
from typing import List

class BaseNotifier(ABC):
    """Abstract base class for all notification providers."""

    @abstractmethod
    def send(self, message: str, recipients: List[str]) -> dict:
        """
        Sends a message to a list of recipients.

        Args:
            message: The text message to send.
            recipients: A list of recipient identifiers (e.g., phone numbers).

        Returns:
            A dictionary containing the status of the operation.
            Example: {"status": "success", "details": "Message sent to 3 recipients."}
                     {"status": "error", "details": "Authentication failed."}
        """
        pass
