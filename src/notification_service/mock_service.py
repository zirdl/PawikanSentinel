from .base import BaseNotificationService

class MockNotificationService(BaseNotificationService):
    """
    A mock notification service that prints alerts to the console.
    """

    def send_alert(self, message: str) -> bool:
        """
        Prints the alert message to the console.

        Args:
            message (str): The alert message to send.

        Returns:
            bool: Always returns True.
        """
        print(f"[ALERT] {message}")
        return True
