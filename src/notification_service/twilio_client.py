import os
import time
from collections import deque
from twilio.rest import Client # type: ignore
from src.config import ConfigManager
from src.notification_service.base import BaseNotificationService

class TwilioClient(BaseNotificationService):
    """
    A wrapper for the Twilio API to send SMS messages with rate limiting.
    """

    def __init__(self, config: ConfigManager):
        """
        Initializes the TwilioClient with configuration from ConfigManager.
        """
        self.account_sid = config.get("TWILIO", "ACCOUNT_SID")
        self.auth_token = config.get("TWILIO", "AUTH_TOKEN")
        self.twilio_phone_number = config.get("TWILIO", "PHONE_NUMBER")
        self.recipient_phone_number = config.get("TWILIO", "RECIPIENT_NUMBER")
        self.messaging_service_sid = config.get("TWILIO", "MESSAGING_SERVICE_SID")
        self.max_messages_per_hour = config.get_int("TWILIO", "MAX_MESSAGES_PER_HOUR", 5)

        if not all([self.account_sid, self.auth_token, self.recipient_phone_number]) or not (self.twilio_phone_number or self.messaging_service_sid):
            raise ValueError("Twilio credentials not fully configured in config.ini or environment variables.")

        self.client = Client(self.account_sid, self.auth_token)
        self.message_timestamps: deque = deque()

    def _prune_timestamps(self):
        """Removes timestamps older than one hour."""
        one_hour_ago = time.time() - 3600
        while self.message_timestamps and self.message_timestamps[0] < one_hour_ago:
            self.message_timestamps.popleft()

    def send_alert(self, message: str) -> bool:
        """
        Sends an SMS message using the Twilio API, respecting rate limits.

        Args:
            message (str): The message to send.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        self._prune_timestamps()

        if len(self.message_timestamps) >= self.max_messages_per_hour:
            print("Rate limit exceeded. SMS not sent.")
            return False

        try:
            message_payload = {
                "to": self.recipient_phone_number,
                "body": message,
            }
            if self.messaging_service_sid and self.messaging_service_sid != "YOUR_MESSAGING_SERVICE_SID_HERE":
                message_payload["messaging_service_sid"] = self.messaging_service_sid
            elif self.twilio_phone_number:
                message_payload["from_"] = self.twilio_phone_number
            else:
                raise ValueError("No valid sender configured (phone number or messaging service SID).")

            self.client.messages.create(**message_payload)
            self.message_timestamps.append(time.time())
            return True
        except Exception as e:
            print(f"Failed to send SMS: {e}")
            return False
