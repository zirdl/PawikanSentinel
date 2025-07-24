# How to Test the Pawikan Sentinel Application

This guide provides two methods for testing the turtle detection application. The first method is a full end-to-end test that simulates a live camera feed, while the second method is a quicker way to test the detection logic on a local file.

---

## Method 1: Simulated Live Test (End-to-End)

This is the most comprehensive method. You will use a pre-recorded video of a turtle and stream it to your application using the RTSP protocol. This will test the entire pipeline, from video input to the Twilio notification.

### What You'll Need:
1.  A video file with a sea turtle (e.g., `turtle_test.mp4`). You can find suitable videos on YouTube or stock footage sites.
2.  **VLC Media Player** installed on your computer to act as the RTSP server.

### Steps:

**1. Get a Turtle Video:**
   Download a video of a sea turtle. For example, search on YouTube for "sea turtle nesting video" and download one. Save it as `turtle_test.mp4`.

**2. Set up the RTSP Stream with VLC:**
   a. Open VLC Media Player.
   b. Go to `Media` -> `Stream`.
   c. In the "File" tab, click `+ Add...` and select your `turtle_test.mp4` file.
   d. Click the `Stream` button at the bottom.
   e. In the "Destinations" screen, select `RTSP` from the dropdown and click `Add`.
   f. For "Path", enter `/test`. The default port is `8554`.
   g. Click `Next`. In the "Transcoding Options", you can leave the default profile (`Video - H.264 + MP3 (MP4)`) and click `Next`.
   h. Click `Stream`. VLC will now be streaming your video file on an RTSP feed at `rtsp://127.0.0.1:8554/test`.

**3. Configure Pawikan Sentinel:**
   Update your `config.ini` file to point to your local VLC stream:

   ```ini
   [APP]
   RTSP_URL = rtsp://127.0.0.1:8554/test
   ...
   ```

**4. Run the Application:**
   With the VLC stream running, start the Pawikan Sentinel application:
   ```bash
   source .venv/bin/activate
   python3 src/main.py
   ```
   The application will connect to the VLC stream, process the video, and should send a Twilio notification when a turtle is detected.

---

## Method 2: Direct Image/Video File Processing

This method involves modifying the `main.py` script to read directly from a local video file or a single image. This is useful for quickly testing the ML model and detection logic without the need to set up an RTSP stream.

### Steps:

**1. Modify `main.py`:**
   You will need to comment out the RTSP client and replace it with a `cv2.VideoCapture` object that reads from your local file.

   Open `src/main.py` and make the following changes:

   ```python
   # ... imports ...

   def main():
       # ... config loading ...

       # === START MODIFICATION ===

       # Comment out the RTSP client initialization
       # rtsp_client = RTSPClient(RTSP_URL)
       # if not rtsp_client.connect():
       #     logger.error(f"Failed to connect to RTSP stream at {RTSP_URL}. Exiting.")
       #     return

       # Add this line to read from a local video file
       video_path = 'path/to/your/turtle_test.mp4' # <--- CHANGE THIS
       video_capture = cv2.VideoCapture(video_path)

       # === END MODIFICATION ===


       # ... ML Inference Engine initialization ...
       # ... Detection Analyzer initialization ...
       # ... Alert Manager initialization ...

       try:
           # === START MODIFICATION ===
           # while True:
           while video_capture.isOpened():
           # === END MODIFICATION ===

               # === START MODIFICATION ===
               # Read frame from video file
               success, frame = video_capture.read()
               # === END MODIFICATION ===

               if not success:
                   # logger.warning("Failed to read frame. Attempting to reconnect...")
                   # if not rtsp_client.reconnect():
                   #     logger.error("Failed to reconnect to RTSP stream. Exiting.")
                   #     break
                   # continue
                   print("End of video file.")
                   break # Exit loop at the end of the video

               # ... rest of the processing loop ...

       # ... except KeyboardInterrupt ...
       finally:
           # rtsp_client.release()
           video_capture.release() # Release the video capture object
           cv2.destroyAllWindows()
           logger.info("Pawikan Sentinel application terminated.")

   if __name__ == "__main__":
       main()
   ```

**2. Run the Modified Script:**
   Save the changes to `src/main.py` and run it:
   ```bash
   source .venv/bin/activate
   python3 src/main.py
   ```
   The application will process your video file frame by frame and send a notification upon detection.

**Important:** Remember to revert the changes to `src/main.py` before deploying the application to the Raspberry Pi.
