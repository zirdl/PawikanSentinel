from datetime import datetime, timedelta
import time

class AlertGenerator:
    """
    Generates alerts based on detection events, with deduplication.
    """

    def __init__(self, deduplication_window_minutes: int):
        """
        Initializes the AlertGenerator.

        Args:
            deduplication_window_minutes (int): The time window in minutes for deduplicating alerts.
        """
        self.deduplication_window = timedelta(minutes=deduplication_window_minutes)
        self.last_alert_time = None

    def should_send_alert(self) -> bool:
        """
        Determines if a new alert should be sent based on the deduplication window.

        Returns:
            bool: True if a new alert should be sent, False otherwise.
        """
        if self.last_alert_time is None:
            return True
        
        if datetime.now() - self.last_alert_time > self.deduplication_window:
            return True
        
        return False

    def generate_alert(self, tracked_objects: dict) -> str:
        """
        Generates an alert message based on the tracked objects.

        Args:
            tracked_objects (dict): A dictionary of tracked objects from the ObjectTracker.

        Returns:
            str: The formatted alert message.
        """
        num_turtles = len(tracked_objects)
        if num_turtles == 0:
            return ""

        if self.should_send_alert():
            self.last_alert_time = datetime.now()
            if num_turtles == 1:
                return "A sea turtle has been detected."
            else:
                return f"{num_turtles} sea turtles have been detected."
        
        return ""

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Initialize the alert generator with a 1-minute deduplication window
    alert_generator = AlertGenerator(deduplication_window_minutes=1)

    # 2. Simulate some detection events
    print("Simulating detection events...")

    # First detection
    tracked_objects = {0: (10, 10)}
    alert_message = alert_generator.generate_alert(tracked_objects)
    print(f"- {datetime.now()}: {alert_message}")

    # Second detection (within the deduplication window)
    time.sleep(30)
    tracked_objects = {0: (12, 12), 1: (100, 100)}
    alert_message = alert_generator.generate_alert(tracked_objects)
    print(f"- {datetime.now()}: {alert_message}")

    # Third detection (after the deduplication window)
    time.sleep(31)
    tracked_objects = {0: (15, 15)}
    alert_message = alert_generator.generate_alert(tracked_objects)
    print(f"- {datetime.now()}: {alert_message}")
