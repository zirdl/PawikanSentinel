"""
Throttling and cooldown logic.
"""
import time

class CooldownManager:
    """Manages simple in-memory cooldowns."""

    def __init__(self, key: str, cooldown_seconds: int):
        """
        Args:
            key: A unique key for the cooldown (e.g., 'notification', 'gallery_save').
            cooldown_seconds: The duration of the cooldown in seconds.
        """
        self.key = key
        self.cooldown_seconds = cooldown_seconds
        self.last_event_time = 0  # Stores the timestamp of the last event

    def is_ready(self) -> bool:
        """Checks if the cooldown period has passed."""
        current_time = time.monotonic()
        return (current_time - self.last_event_time) >= self.cooldown_seconds

    def trigger(self) -> None:
        """Records that an event has occurred, resetting the cooldown timer."""
        self.last_event_time = time.monotonic()
