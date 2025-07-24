import cv2
import time
from typing import Optional, Tuple
from src.frame_processor.preprocessing import preprocess_frame

class RTSPClient:
    """
    Manages the connection to an RTSP camera stream and captures frames.
    """

    def __init__(self, rtsp_url: str):
        """
        Initializes the RTSP client.

        Args:
            rtsp_url (str): The URL of the RTSP stream.
        """
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False

    def connect(self) -> bool:
        """
        Connects to the RTSP stream.

        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                print(f"Error: Could not open RTSP stream at {self.rtsp_url}")
                self.is_connected = False
                return False
            self.is_connected = True
            print("Successfully connected to RTSP stream.")
            return True
        except Exception as e:
            print(f"An exception occurred while trying to connect: {e}")
            self.is_connected = False
            return False

    def read_frame(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """
        Reads a single frame from the stream.

        Returns:
            Optional[Tuple[bool, MatLike]]: A tuple containing a success flag and the frame,
                                             or None if not connected.
        """
        if not self.is_connected or self.cap is None:
            print("Error: Not connected to the stream.")
            return False, None
        
        success, frame = self.cap.read()
        return success, frame

    def release(self):
        """
        Releases the video capture object and disconnects from the stream.
        """
        if self.cap is not None:
            self.cap.release()
        self.is_connected = False
        print("RTSP stream connection released.")

    def reconnect(self, max_retries=5, delay=2) -> bool:
        """
        Attempts to reconnect to the stream with exponential backoff.

        Args:
            max_retries (int): The maximum number of reconnection attempts.
            delay (int): The initial delay between retries in seconds.

        Returns:
            bool: True if reconnection is successful, False otherwise.
        """
        self.release()
        print("Attempting to reconnect...")
        for i in range(max_retries):
            if self.connect():
                return True
            print(f"Reconnect attempt {i + 1}/{max_retries} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        print("Failed to reconnect after multiple attempts.")
        return False

if __name__ == '__main__':
    # --- Example Usage ---
    # IMPORTANT: Replace this with the actual URL of your RTSP camera.
    # For testing without a camera, you can use a local video file path.
    # Example: "rtsp://username:password@camera_ip:554/stream_path"
    RTSP_URL = "path/to/your/video.mp4" 

    client = RTSPClient(RTSP_URL)

    if client.connect():
        try:
            frame_count = 0
            while True:
                success, frame = client.read_frame()
                
                if not success:
                    print("Failed to read frame. Attempting to reconnect...")
                    if not client.reconnect():
                        break
                    continue

                # --- Your processing logic would go here ---
                # For this example, we'll preprocess the frame and display it.
                if frame is not None:
                    preprocessed = preprocess_frame(frame, (640, 480))
                    cv2.imshow("RTSP Stream", frame)
                frame_count += 1
                
                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            print(f"Processed {frame_count} frames.")

        except KeyboardInterrupt:
            print("Stream reading interrupted by user.")
        finally:
            # Clean up
            client.release()
            cv2.destroyAllWindows()
    else:
        print("Could not connect to the RTSP stream. Please check the URL and network connection.")
