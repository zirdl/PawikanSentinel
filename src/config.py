import configparser
import os

class ConfigManager:
    """
    Manages application configuration, loading from a file and environment variables.
    """

    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self._load_config()

    def _load_config(self):
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            print(f"Warning: Configuration file '{self.config_file}' not found. Using defaults and environment variables.")

    def get(self, section, option, default=None):
        """
        Gets a configuration value. Tries environment variable first, then config file, then default.
        """
        env_var_name = f"{section.upper()}_{option.upper()}"
        if env_var_name in os.environ:
            return os.environ[env_var_name]
        
        if self.config.has_option(section, option):
            return self.config.get(section, option)
        
        return default

    def get_int(self, section, option, default=None):
        return int(self.get(section, option, default))

    def get_float(self, section, option, default=None):
        return float(self.get(section, option, default))

    def get_boolean(self, section, option, default=None):
        return self.config.getboolean(section, option) if self.config.has_option(section, option) else default


if __name__ == '__main__':
    # Example Usage:
    # Create a dummy config.ini for testing
    with open('config.ini', 'w') as f:
        f.write("[APP]\n")
        f.write("RTSP_URL = rtsp://test.camera.com/stream\n")
        f.write("MODEL_PATH = ./models/best.tflite\n")
        f.write("CONFIDENCE_THRESHOLD = 0.7\n")
        f.write("DEDUPLICATION_WINDOW_MINUTES = 5\n")
        f.write("[TWILIO]\n")
        f.write("ACCOUNT_SID = ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        f.write("AUTH_TOKEN = your_auth_token\n")
        f.write("PHONE_NUMBER = +1234567890\n")
        f.write("RECIPIENT_NUMBER = +1987654321\n")
        f.write("SAVE_ANNOTATED_FRAMES = False\n")
        f.write("ANNOTATED_FRAMES_DIR = /tmp/pawikan_sentinel/annotated_frames\n")

    # Test loading from file
    config = ConfigManager('config.ini')
    print(f"RTSP URL from config: {config.get('APP', 'RTSP_URL')}")
    print(f"Confidence Threshold from config: {config.get_float('APP', 'CONFIDENCE_THRESHOLD')}")

    # Test environment variable override
    os.environ['APP_RTSP_URL'] = 'rtsp://env.camera.com/stream'
    print(f"RTSP URL from env: {config.get('APP', 'RTSP_URL')}")
    del os.environ['APP_RTSP_URL'] # Clean up env var

    # Test default value
    print(f"Non-existent option (default): {config.get('APP', 'NON_EXISTENT', 'default_value')}")

    # Clean up dummy config file
    os.remove('config.ini')
