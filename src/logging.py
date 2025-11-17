import logging
import re
from typing import Any


class SensitiveDataFilter(logging.Filter):
    """
    Custom filter that removes sensitive data from log messages.
    """
    
    def __init__(self, name=''):
        super().__init__(name)
        # Define patterns for sensitive data that should be masked
        self.sensitive_patterns = [
            # API keys (common formats)
            (r'(\b(?:api_?key|secret|token|password|pwd)\s*[:=]\s*)([^\s\'"&,<>]+)', r'\1[REDACTED]'),
            # Phone numbers (various formats)
            (r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE REDACTED]'),
            # Email-like patterns in context of sensitive data
            (r'(\b(?:email|mail)\s*[:=]\s*)([^\s\'"&,<>@]+@[^\s\'"&,<>]+)', r'\1[EMAIL REDACTED]'),
            # General pattern for values that look like secrets
            (r'(["\']?(?:api_?key|secret|token|password|pwd|auth)["\']?\s*[:=]\s*)["\']?([^\s\'"&,<>]+)["\']?', r'\1[REDACTED]'),
        ]
        
        # Compile the regex patterns for better performance
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), replacement) 
                                  for pattern, replacement in self.sensitive_patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        # Apply sensitive data filtering to the message
        for pattern, replacement in self.compiled_patterns:
            record.msg = pattern.sub(replacement, str(record.msg))
        return True  # Always allow the record to pass through


def setup_sensitive_data_logging():
    """
    Set up sensitive data logging by adding the filter to all loggers.
    """
    sensitive_filter = SensitiveDataFilter()
    
    # Add the filter to all existing handlers of the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(sensitive_filter)
    
    # Also return the filter so it can be added to specific loggers if needed
    return sensitive_filter