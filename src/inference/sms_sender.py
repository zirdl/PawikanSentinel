import os
import requests
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SemaphoreSMSSender:
    """
    SMS sender implementation using Semaphore API instead of Twilio.
    This is more cost-effective for Philippine deployments.
    """
    
    def __init__(self):
        self.api_key = os.getenv("SEMAPHORE_API_KEY")
        self.sender_name = os.getenv("SEMAPHORE_SENDER_NAME", "PawikanSentinel")
        self.cooldown_period = int(os.getenv("SMS_NOTIFICATION_COOLDOWN", "10")) * 60  # Convert minutes to seconds
        self.last_sms_times = {}  # Track last SMS time per contact
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            logger.warning("Semaphore SMS not configured - SEMAPHORE_API_KEY not set")
    
    def is_enabled(self) -> bool:
        """Check if Semaphore SMS is properly configured and enabled."""
        return self.enabled
    
    def _send_single_sms(self, message: str, number: str) -> bool:
        """
        Send a single SMS message using Semaphore API.
        
        Args:
            message: The message to send
            number: The phone number to send to (format: 09xxxxxxxxx for Philippines)
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.error("Semaphore not enabled - cannot send SMS")
            return False
        
        try:
            # Prepare the API parameters
            params = {
                'apikey': self.api_key,
                'sendername': self.sender_name,
                'message': message,
                'number': number
            }
            
            # Make the API request to Semaphore
            response = requests.post('https://semaphore.co/api/v4/messages', data=params)
            
            if response.status_code == 200:
                result = response.json()
                message_id = result.get('message_id', 'unknown')
                logger.info(f"SMS sent to {number}: Message ID {message_id}")
                return True
            else:
                logger.error(f"Failed to send SMS to {number}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Exception when sending SMS to {number}: {e}")
            return False
    
    def send_sms_notification(self, phone_numbers: List[str], message_body: str) -> Dict[str, bool]:
        """
        Send SMS notifications to multiple contacts with cooldown.
        
        Args:
            phone_numbers: List of phone numbers to send the message to
            message_body: The message to send
            
        Returns:
            Dict mapping phone numbers to success status
        """
        if not self.enabled:
            logger.error("Semaphore not enabled - cannot send SMS notifications")
            return {num: False for num in phone_numbers}
        
        results = {}
        current_time = time.time()
        
        for phone_number in phone_numbers:
            # Check cooldown period for this contact
            last_sms_time = self.last_sms_times.get(phone_number, 0)
            
            if current_time - last_sms_time < self.cooldown_period:
                logger.info(f"SMS cooldown active for {phone_number}. Skipping notification.")
                results[phone_number] = False
                continue
            
            # Send the SMS
            success = self._send_single_sms(message_body, phone_number)
            results[phone_number] = success
            
            if success:
                # Update last SMS time for this contact
                self.last_sms_times[phone_number] = current_time
            else:
                logger.error(f"Failed to send SMS to {phone_number}")
        
        return results

    def send_detailed_notification(self, phone_numbers: List[str], 
                                 class_name: str, 
                                 confidence: float, 
                                 timestamp: str) -> Dict[str, bool]:
        """
        Send a detailed notification about a detection.
        
        Args:
            phone_numbers: List of phone numbers to send the message to
            class_name: Name of the detected class (e.g., "pawikan")
            confidence: Confidence level of the detection
            timestamp: Timestamp of detection
            
        Returns:
            Dict mapping phone numbers to success status
        """
        message_body = f"Pawikan Sentinel Alert: {class_name} detected with {confidence:.2f} confidence at {timestamp}"
        return self.send_sms_notification(phone_numbers, message_body)