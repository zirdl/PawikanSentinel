# Pawikan Sentinel

This project develops a web-based management application for the Pawikan Sentinel Project, aimed at supporting sea turtle monitoring and conservation. It runs on a Raspberry Pi and integrates with an ML inference service to provide a user-friendly interface for managing analytics, system settings, and detection data.

## Key Features

*   **Secure Admin Dashboard**: Manage system settings, API secrets, and personnel contacts.
*   **Analytics & Visualization**: View detection event trends and data.
*   **Live Dashboard**: Display live RTSP feed with camera and inference status.
*   **Detection Frame Gallery**: Browse annotated detection images.
*   **Personnel & Notification Management**: Configure contacts and notification settings.
*   **Comprehensive Logging**: Access structured JSON logs for diagnostics.

## System Architecture

The Pawikan Sentinel system operates on a Raspberry Pi and is composed of two primary services:

1.  **Inference Service**: This background service continuously processes video streams from an RTSP camera. It uses a TFLite Model to detect turtles, then stores detection data in a SQLite Database, saves annotated images to a Detection Gallery, and records events in System Logs. It also triggers SMS Notifications for critical events.

2.  **Web Application**: This FastAPI-based web interface provides an Admin Web Interface for users. It interacts with the SQLite Database to manage settings and view analytics, accesses the Detection Gallery to display images, and retrieves information from System Logs. It also allows for the management of SMS Notification settings.

Both services run concurrently on the Raspberry Pi, sharing data through the centralized SQLite Database, Detection Gallery, and System Logs.

