import os
from twilio.rest import Client # type: ignore

class TwilioClient:
    """
    A wrapper for the Twilio API to send SMS messages.
    """

    def __init__(self):
        """
        Initializes the TwilioClient.
        """
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.environ.get("TWILIO_PHONE_NUMBER")
        self.recipient_phone_number = os.environ.get("RECIPIENT_PHONE_NUMBER")

        if not all([self.account_sid, self.auth_token, self.twilio_phone_number, self.recipient_phone_number]):
            raise ValueError("Twilio environment variables not set.")

        self.client = Client(self.account_sid, self.auth_token)

    def send_sms(self, message: str) -> bool:
        """
        Sends an SMS message using the Twilio API.

        Args:
            message (str): The message to send.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        try:
            self.client.messages.create(
                to=self.recipient_phone_number,
                from_=self.twilio_phone_number,
                body=message
            )
            return True
        except Exception as e:
            print(f"Failed to send SMS: {e}")
            return False
