# Requirements Document

## Introduction

The Pawikan Sentinel is a real-time sea turtle detection system designed for conservation efforts. The system uses a Raspberry Pi 4B with an infrared camera to detect nesting sea turtles and automatically alert conservation teams via SMS. The solution employs edge-optimized machine learning with YOLOv5n and multi-stage transfer learning to achieve high accuracy detection within strict hardware constraints.

## Requirements

### Requirement 1

**User Story:** As a conservation team member, I want to receive automatic SMS alerts when sea turtles are detected, so that I can respond quickly to protect nesting turtles.

#### Acceptance Criteria

1. WHEN a sea turtle is detected in the camera feed THEN the system SHALL send an SMS alert within 30 seconds
2. WHEN multiple turtles are detected simultaneously THEN the system SHALL send a single consolidated alert message
3. WHEN the same turtle remains in view THEN the system SHALL NOT send duplicate alerts within a 10-minute window
4. IF SMS delivery fails THEN the system SHALL retry up to 3 times with exponential backoff

### Requirement 2

**User Story:** As a conservation researcher, I want the system to accurately detect sea turtles in infrared camera feeds, so that we minimize false alarms and missed detections.

#### Acceptance Criteria

1. WHEN processing infrared camera frames THEN the system SHALL achieve 85% or higher detection accuracy
2. WHEN analyzing video feeds THEN the system SHALL maintain false positive rates below 10%
3. WHEN a turtle is present in the frame THEN the system SHALL detect it within 2 seconds of appearance
4. IF lighting conditions change THEN the system SHALL maintain consistent detection performance

### Requirement 3

**User Story:** As a system administrator, I want to train and deploy optimized ML models efficiently, so that the detection system can be updated and improved over time.

#### Acceptance Criteria

1. WHEN training models in Google Colab THEN the system SHALL complete full training pipeline within 8 GPU sessions
2. WHEN converting models to TensorFlow Lite THEN the system SHALL achieve 3-4x inference speedup on Raspberry Pi
3. WHEN deploying new models THEN the system SHALL validate performance before replacing production model
4. IF model performance degrades THEN the system SHALL automatically rollback to previous stable version

### Requirement 4

**User Story:** As a conservation project manager, I want the system to handle multiple data sources and training stages, so that detection accuracy improves over time with domain-specific learning.

#### Acceptance Criteria

1. WHEN downloading datasets THEN the system SHALL use curl commands to fetch GTST-2023 and SeaTurtleID2022 from Kaggle
2. WHEN preparing datasets THEN the system SHALL combine and process them into a structure compatible with YOLOv5 training scripts
3. WHEN training with GTST-2023 dataset THEN the system SHALL process all annotated frames from the NIR images
4. WHEN augmenting with SeaTurtleID2022 data THEN the system SHALL integrate and convert the format appropriately
5. WHEN performing multi-stage transfer learning THEN the system SHALL progress through these specific stages:
   - Stage 1: Load YOLOv5n COCO pretrained weights
   - Stage 2: Fine-tune on general wildlife/animal datasets
   - Stage 3: Specialize on turtle datasets (GTST-2023 + SeaTurtleID2022)
   - Stage 4: Domain adaptation for RGB to NIR transition
6. IF training data is corrupted or incomplete THEN the system SHALL validate data integrity before training

### Requirement 5

**User Story:** As a field operator, I want the system to integrate seamlessly with existing camera infrastructure, so that deployment requires minimal hardware changes.

#### Acceptance Criteria

1. WHEN connecting to RTSP cameras THEN the system SHALL establish stable video streams within 10 seconds
2. WHEN processing video feeds THEN the system SHALL handle standard RTSP protocols and codecs
3. IF network connectivity is lost THEN the system SHALL buffer detections and send alerts when connection is restored
4. WHEN camera feed quality degrades THEN the system SHALL adjust processing parameters automatically