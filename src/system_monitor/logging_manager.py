import logging
import logging.handlers

class LoggingManager:
    """
    Manages structured logging for the application.
    """

    def __init__(self, log_file: str, log_level=logging.INFO):
        """
        Initializes the LoggingManager.

        Args:
            log_file (str): The path to the log file.
            log_level: The minimum log level to record.
        """
        self.logger = logging.getLogger("PawikanSentinel")
        self.logger.setLevel(log_level)

        # Create a rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1024 * 1024, backupCount=5
        )

        # Create a formatter and add it to the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)

    def get_logger(self):
        """
        Returns the configured logger instance.

        Returns:
            The logger instance.
        """
        return self.logger

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Initialize the logging manager
    logging_manager = LoggingManager("pawikan_sentinel.log")
    logger = logging_manager.get_logger()

    # 2. Log some messages
    logger.info("Application starting...")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    print("Check the pawikan_sentinel.log file for the log messages.")
