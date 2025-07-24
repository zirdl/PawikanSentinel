import sys
from src.config import ConfigManager
from src.notification_service.twilio_client import TwilioClient

def send_test_sms():
    """
    Initializes the Twilio client and sends a single test SMS message.
    """
    try:
        print("Initializing configuration...")
        config = ConfigManager(config_file='/home/gio/Projects/PawikanSentinel/config.ini')

        messaging_service_sid = config.get("TWILIO", "MESSAGING_SERVICE_SID")
        if messaging_service_sid and messaging_service_sid != "YOUR_MESSAGING_SERVICE_SID_HERE":
            print(f"Using Messaging Service SID: {messaging_service_sid}")
        else:
            print("Using Twilio Phone Number.")
        
        print("Initializing Twilio client...")
        twilio_client = TwilioClient(config)
        
        message = "This is a test message from Pawikan Sentinel (via Messaging Service SID)."
        print(f"Sending test SMS: '{message}'")
        
        success = twilio_client.send_sms(message)
        
        if success:
            print("Test SMS sent successfully!")
        else:
            print("Failed to send test SMS. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    send_test_sms()
