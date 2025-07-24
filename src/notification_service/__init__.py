"""
Notification Service for sending alerts via Twilio.
"""

from twilio.rest import Client # type: ignore

class NotificationService:
    """
    A service to send SMS alerts using the Twilio API.
    """
    def __init__(self, sid, token, phone_number):
        """
        Initializes the NotificationService.

        Args:
            sid (str): Twilio Account SID.
            token (str): Twilio Auth Token.
            phone_number (str): Twilio phone number for sending messages.
        """
        self.client = Client(sid, token)
        self.phone_number = phone_number

    def send_alert(self, message, recipient):
        """
        Sends an SMS alert to a recipient.

        Args:
            message (str): The message to send.
            recipient (str): The recipient's phone number.
        """
        try:
            message = self.client.messages.create(
                body=message,
                from_=self.phone_number,
                to=recipient
            )
            print(f"Alert sent successfully to {recipient}: {message.sid}")
        except Exception as e:
            print(f"Failed to send alert: {e}")

