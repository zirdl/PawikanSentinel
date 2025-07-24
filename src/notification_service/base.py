from abc import ABC, abstractmethod

class BaseNotificationService(ABC):
    """
    Abstract base class for notification services.
    """

    @abstractmethod
    def send_alert(self, message: str) -> bool:
        """
        Sends an alert.

        Args:
            message (str): The alert message to send.

        Returns:
            bool: True if the alert was sent successfully, False otherwise.
        """
        pass
