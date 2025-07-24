import time
from collections import deque
from src.notification_service.base import BaseNotificationService

class AlertDelivery:
    """
    Manages the delivery of alerts, with a queue and retry logic.
    """

    def __init__(self, notification_service: BaseNotificationService, max_retries=3, retry_delay=5):
        """
        Initializes the AlertDelivery system.

        Args:
            notification_service (BaseNotificationService): An object that inherits from BaseNotificationService.
            max_retries (int): The maximum number of retries for a failed alert.
            retry_delay (int): The delay in seconds between retries.
        """
        self.notification_service = notification_service
        self.alert_queue: deque[tuple[str, int]] = deque()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def enqueue_alert(self, message: str):
        """
        Adds an alert message to the queue.

        Args:
            message (str): The alert message to send.
        """
        if message:
            self.alert_queue.append((message, 0))  # (message, retry_count)

    def process_queue(self):
        """
        Processes the alert queue, sending alerts and handling retries.
        """
        if not self.alert_queue:
            return

        message, retry_count = self.alert_queue.popleft()

        if self.notification_service.send_alert(message):
            print(f"Alert sent successfully: {message}")
        else:
            print(f"Failed to send alert: {message}")
            if retry_count < self.max_retries:
                print(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                self.alert_queue.append((message, retry_count + 1))
            else:
                print(f"Failed to send alert after {self.max_retries} retries: {message}")

if __name__ == '__main__':
    from src.notification_service.mock_service import MockNotificationService

    # --- Example Usage ---

    # 1. Initialize the alert delivery system with the mock service
    mock_service = MockNotificationService()
    alert_delivery = AlertDelivery(mock_service, max_retries=3, retry_delay=1)

    # 2. Enqueue an alert
    alert_delivery.enqueue_alert("A sea turtle has been detected.")

    # 3. Process the queue until it's empty
    while alert_delivery.alert_queue:
        alert_delivery.process_queue()
