# Local Development & Testing Guide

This guide explains how to set up, develop, and test Pawikan Sentinel on your local machine without a Raspberry Pi or a real RTSP camera.

## 🛠️ Environment Setup

1.  **Install `uv`**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    ```bash
    uv sync --extra inference
    ```

3.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

## 🌐 Running the Web Application

To run the web app in development mode with **auto-reload** (restarts on code changes):

```bash
./scripts/manage.sh dev
```
Access the UI at: [http://localhost:8000](http://localhost:8000)

## 🧪 Running Model Tests

1.  **Model Availability Smoke Test**:
    ```bash
    bash scripts/test_model.sh
    ```

2.  **Single Image Inference Test**:
    ```bash
    bash scripts/test_image.sh test.jpg
    ```

## 🎥 Simulation via VLC Media Player

This is the recommended way to test the model's accuracy on real video files without a physical camera.

1.  **Open VLC** and go to **Media > Stream...**
2.  **Add your video file** (e.g., `turtle_sample.mp4`) and click **Stream**.
3.  Click **Next**, then in "New Destination", select **RTSP** and click **Add**.
4.  Specify a path (e.g., `/test`) and a port (default `8554`).
5.  Click **Next**, then uncheck "Activate Transcoding" (to save CPU) and click **Next**.
6.  Click **Stream**. VLC is now your "RTSP Camera".

### Testing the Stream
To verify the model can "see" your VLC stream:
```bash
uv run python src/inference/rtsp_infer.py --rtsp-url rtsp://localhost:8554/test
```
The script will display the stream and save `last_detection.jpg` in the project root if it finds a turtle.

## 🔧 Contributing Workflows

1.  **Feature Branching**: Always work on a branch: `git checkout -b feature/your-feature-name`.
2.  **UI Development**: The web templates use Tailwind CSS. If you modify styles, build them using `npm run build`.
3.  **Testing**: Before committing, ensure `scripts/test_model.sh` passes.
