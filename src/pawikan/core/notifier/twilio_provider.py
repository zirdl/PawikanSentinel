"""
Notifier for sending SMS via the Twilio API.
"""
from typing import List
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from .base import BaseNotifier

class TwilioNotifier(BaseNotifier):
    """Sends SMS using the Twilio API."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        if not all([account_sid, auth_token, from_number]):
            raise ValueError("Twilio credentials (account_sid, auth_token, from_number) are required.")
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number

    def send(self, message: str, recipients: List[str]) -> dict:
        """
        Sends an SMS to a list of phone numbers.

        Args:
            message: The message to send.
            recipients: A list of phone numbers in E.164 format.

        Returns:
            A dictionary with the operation status.
        """
        if not recipients:
            return {"status": "warning", "details": "No recipients provided."}

        success_count = 0
        errors = []
        for to_number in recipients:
            try:
                self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=to_number
                )
                success_count += 1
            except TwilioRestException as e:
                error_msg = f"Failed to send to {to_number}: {e.msg}"
                print(error_msg)
                errors.append(error_msg)

        if success_count == len(recipients):
            return {
                "status": "success",
                "details": f"Successfully sent SMS to {success_count} recipients.",
            }
        else:
            return {
                "status": "partial_failure",
                "details": f"Sent to {success_count}/{len(recipients)}. Errors: {'; '.join(errors)}",
            }
