# Pawikan Sentinel

AI-powered sea-turtle detection on Raspberry Pi 4B with SMS alerts.

## Overview

The Pawikan Sentinel is a real-time sea turtle detection system designed for conservation efforts. The system uses a Raspberry Pi 4B with an infrared camera to detect nesting sea turtles and automatically alert conservation teams via SMS. The solution employs edge-optimized machine learning with YOLOv5n and multi-stage transfer learning to achieve high accuracy detection within strict hardware constraints.

## Dataset Sources

The model was trained on a combination of the following datasets:

*   **GTST-2023**: [https://www.kaggle.com/datasets/sabasylvester/global-sea-turtle-dataset-2023-gtst-2023](https://www.kaggle.com/datasets/sabasylvester/global-sea-turtle-dataset-2023-gtst-2023)
*   **SeaTurtleID2022**: [https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022)

## Project Pipeline

```mermaid
graph TD
    subgraph "On-Site Hardware"
        A[/"📹\nInfrared Camera"/];
    end

    subgraph "Raspberry Pi: The Brain"
        direction LR
        B["Frame Processor\n(Grabs video frames)"];
        C["AI Turtle Detection\n(YOLOv5n TFLite)"];
        D["Detection Analyzer\n(Filters results, avoids false alarms)"];
        E["Smart Alert Manager\n(Sends one alert per event)"];
    end

    subgraph "Cloud Services"
        F[/"☁️\nTwilio API"/];
    end

    subgraph "Conservation Team"
        G[/"📱\nTeam's Phones"/];
    end

    subgraph "Development & Training (Offline)"
        H["Model Training\n(Google Colab, Python)"];
    end

    A -- "Live RTSP Video Stream" --> B;
    B -- "Frames" --> C;
    C -- "Detection Data (Boxes, Scores)" --> D;
    D -- "Confirmed Turtle Sighting" --> E;
    E -- "API Call (HTTP Request)" --> F;
    F -- "Sends SMS Notification" --> G;
    H -. "Optimized .tflite Model" .-> C;

```