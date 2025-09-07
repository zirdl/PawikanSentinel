### `inference.md`

**Your Role**: AI Engineer

* RTSP worker:
  * Connects to camera RTSP.
  * Sends frames → Roboflow inference server (running in Docker).
  * Logs results into `detections` table.
  * Saves snapshot images for gallery.
* Runs as background service (separate thread or process).
* Configurable polling interval.

