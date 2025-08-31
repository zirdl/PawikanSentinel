"""
Notifier for sending SMS via a serial GSM modem.
"""
import serial
from typing import List
from .base import BaseNotifier

class GsmNotifier(BaseNotifier):
    """Sends SMS using AT commands over a serial connection."""

    def __init__(self, device: str, baudrate: int):
        self.device = device
        self.baudrate = baudrate
        self.serial_conn = None
        # NOTE: This is a simplified implementation. A robust one would handle
        # AT command responses, errors, and timeouts much more carefully.
        # The hardware has not been acquired yet.

    def _connect(self):
        try:
            self.serial_conn = serial.Serial(self.device, self.baudrate, timeout=5)
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to GSM modem at {self.device}: {e}")

    def _send_at_command(self, command: str, expected_response: str = "OK"):
        if not self.serial_conn or not self.serial_conn.is_open:
            self._connect()
        
        self.serial_conn.write((command + '\r\n').encode())
        response = self.serial_conn.read(100).decode(errors='ignore')
        if expected_response not in response:
            raise IOError(f"GSM modem command '{command}' failed. Response: {response}")

    def send(self, message: str, recipients: List[str]) -> dict:
        """
        Sends an SMS to multiple recipients.
        
        Warning: Hardware not yet acquired. This is a placeholder implementation.
        """
        if not recipients:
            return {"status": "warning", "details": "No recipients provided."}

        try:
            self._connect()
            self._send_at_command('AT') # Check connection
            self._send_at_command('AT+CMGF=1') # Set to text mode

            success_count = 0
            for number in recipients:
                try:
                    self._send_at_command(f'AT+CMGS="{number}"', expected_response=">")
                    self.serial_conn.write(message.encode())
                    self.serial_conn.write(b'\x1a') # Ctrl+Z to send
                    # A real implementation would wait for +CMGS response here
                    success_count += 1
                except (IOError, serial.SerialException) as e:
                    print(f"Failed to send SMS to {number}: {e}")
            
            if self.serial_conn:
                self.serial_conn.close()

            return {
                "status": "success",
                "details": f"Attempted to send SMS to {success_count}/{len(recipients)} recipients.",
            }
        except (ConnectionError, IOError, serial.SerialException) as e:
            return {"status": "error", "details": str(e)}
