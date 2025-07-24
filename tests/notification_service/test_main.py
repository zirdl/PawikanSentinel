import unittest
from unittest.mock import patch, MagicMock
import time

from src.notification_service.twilio_client import TwilioClient
from src.config import ConfigManager

class TestTwilioClient(unittest.TestCase):

    def setUp(self):
        # Create a dummy config.ini for testing
        with open('config.ini', 'w') as f:
            f.write("[TWILIO]\n")
            f.write("ACCOUNT_SID = ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
            f.write("AUTH_TOKEN = your_auth_token\n")
            f.write("PHONE_NUMBER = +1234567890\n")
            f.write("RECIPIENT_NUMBER = +1987654321\n")
            f.write("MAX_MESSAGES_PER_HOUR = 2\n")
        
        self.config = ConfigManager('config.ini')
        self.twilio_client = TwilioClient(self.config)

    def tearDown(self):
        import os
        os.remove('config.ini')

    @patch('src.notification_service.twilio_client.Client')
    def test_send_sms_rate_limit(self, mock_twilio_client_class):
        # Mock the Twilio client instance and its messages.create method
        mock_twilio_instance = MagicMock()
        mock_create = MagicMock()
        mock_twilio_instance.messages.create = mock_create
        mock_twilio_client_class.return_value = mock_twilio_instance

        # Re-initialize the client to use the mock
        self.twilio_client = TwilioClient(self.config)

        # Send messages up to the limit
        self.assertTrue(self.twilio_client.send_sms("Test message 1"))
        self.assertTrue(self.twilio_client.send_sms("Test message 2"))

        # Try to send one more message - it should be rate limited
        self.assertFalse(self.twilio_client.send_sms("Test message 3"))

        # Verify that the Twilio API was called only twice
        self.assertEqual(mock_create.call_count, 2)

        # Now, let's simulate some time passing (an hour)
        # We can manipulate the internal timestamps for testing purposes
        self.twilio_client.message_timestamps[0] = time.time() - 3601
        self.twilio_client.message_timestamps[1] = time.time() - 3601

        # Now we should be able to send another message
        self.assertTrue(self.twilio_client.send_sms("Test message 4"))
        self.assertEqual(mock_create.call_count, 3)

if __name__ == '__main__':
    unittest.main()
