import os
import requests
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class IprogSMSSender:
    """
    SMS sender implementation using iprog API instead of Semaphore.
    This is the new SMS provider for the application.
    """
    
    def __init__(self):
        self.api_token = os.getenv("IPROG_API_TOKEN")
        self.sender_name = os.getenv("IPROG_SENDER_NAME", "PawikanSentinel")
        self.cooldown_period = int(os.getenv("SMS_NOTIFICATION_COOLDOWN", "10")) * 60  # Convert minutes to seconds
        self.last_sms_times = {}  # Track last SMS time per contact
        self.enabled = bool(self.api_token)
        
        if not self.enabled:
            logger.warning("iprog SMS not configured - IPROG_API_TOKEN not set")
    
    def is_enabled(self) -> bool:
        """Check if iprog SMS is properly configured and enabled."""
        return self.enabled
    
    def _normalize_phone_number(self, phone_number: str) -> str:
        """
        Normalize phone number to international format (63 prefix for Philippines).
        Handles various formats like 09xxxxxxxxx, +639xxxxxxxxx, 639xxxxxxxxx.
        """
        # Remove any non-digit characters
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Handle different formats
        if clean_number.startswith('09'):
            # Convert 09xxxxxxxxx to 639xxxxxxxxx
            clean_number = '63' + clean_number[1:]
        elif clean_number.startswith('+63'):
            # Convert +639xxxxxxxxx to 639xxxxxxxxx
            clean_number = clean_number[1:]
        elif clean_number.startswith('63') and len(clean_number) == 11:
            # Already in correct format
            pass
        elif clean_number.startswith('9') and len(clean_number) == 10:
            # Convert 9xxxxxxxxx to 639xxxxxxxxx
            clean_number = '63' + clean_number
        
        return clean_number
    
    def _send_single_sms(self, message: str, number: str) -> bool:
        """
        Send a single SMS message using iprog API.
        
        Args:
            message: The message to send
            number: The phone number to send to (format: 09xxxxxxxxx or 639xxxxxxxxx for Philippines)
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.error("iprog not enabled - cannot send SMS")
            return False
        
        try:
            # Normalize phone number to international format
            normalized_number = self._normalize_phone_number(number)
            
            # Prepare the API parameters - using POST with form data based on documentation
            api_url = "https://sms.iprogtech.com/api/v1/sms_messages"
            data = {
                'api_token': self.api_token,
                'message': message,
                'phone_number': normalized_number
            }
            
            # Make the API request to iprog - using POST with form data as per documentation
            response = requests.post(api_url, data=data)
            
            if response.status_code in [200, 201]:
                result = response.json()
                logger.info(f"SMS sent to {number} (normalized: {normalized_number}): Response {result}")
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
            logger.error("iprog not enabled - cannot send SMS notifications")
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