# Pawikan Sentinel: Web Dashboard Future Implementation Plan

This document outlines the proposed architecture and implementation plan for a future web dashboard feature. This feature will provide a user interface for managing the system, viewing detection data, and monitoring the live camera feed.

## 1. Goal

To create a web-based dashboard that allows authorized users (conservation team members and administrators) to:
-   View a live feed from the RTSP camera.
-   See a history of turtle detection events.
-   Manage a list of conservation team members and their contact phone numbers.
-   Securely log in and out of the system.

## 2. Proposed Architecture

The web dashboard will be a separate application that communicates with the Raspberry Pi detection script. It will consist of a backend server, a database, and a frontend user interface.

```mermaid
graph TD
    subgraph Raspberry Pi
        A[RTSP Camera] --> B(Detection Script);
    end

    subgraph Web Server (Cloud or On-Premise)
        D[Web Backend API];
        E[Database];
        F[Web Frontend];
    end

    subgraph End User
        G[Web Browser];
        H[Conservation Team Phone];
    end

    B -->|1. Send Detection Data (HTTP POST)| D;
    B -->|2. Send SMS (via SIM)| H;
    D <-->|3. Read/Write Data| E;
    F <-->|4. Fetch Data via API| D;
    G -->|5. Views Dashboard| F;
    F -->|6. Displays Live Feed| A;
```

### Core Components

1.  **Web Backend API:**
    -   **Responsibility:** Handle user authentication, manage data, and provide API endpoints for the frontend. It will receive detection data from the Raspberry Pi script.
    -   **Proposed Technology:** Python with **FastAPI** or **Flask**.

2.  **Database:**
    -   **Responsibility:** Store user credentials, team member contact information, and a log of all detection events.
    -   **Proposed Technology:** **PostgreSQL** for robustness, or **SQLite** for simplicity.

3.  **Web Frontend:**
    -   **Responsibility:** Provide the user interface that runs in a web browser. It will display the live feed, data dashboards, and management forms.
    -   **Proposed Technology:** A modern JavaScript framework like **React** or **Vue.js** for a dynamic user experience.

## 3. Key Features

-   **User Authentication:** Secure login/registration for administrators and team members.
-   **Team Management:** A view for administrators to add, edit, or remove team members and their phone numbers.
-   **Detection Dashboard:** A dashboard showing a log of recent turtle detections, including timestamps and potentially saved images.
-   **Live Video Feed:** An embedded video player that displays the live stream from the RTSP camera.

## 4. Implementation Roadmap (Post-Core App)

This work can begin after the current headless Raspberry Pi application is complete and tested.

1.  **Phase 1: Backend & Database Setup**
    -   Initialize a new FastAPI/Flask project.
    -   Design and create the database schema (users, contacts, detections).
    -   Implement API endpoints for user authentication (login, registration).

2.  **Phase 2: Detection Logging**
    -   Create an API endpoint for the Raspberry Pi to submit detection events to.
    -   Modify the Raspberry Pi's `main.py` script to send an HTTP POST request to this endpoint upon each new detection.

3.  **Phase 3: Frontend Development**
    -   Set up a new React/Vue.js project.
    -   Build UI components for login, the main dashboard, and user management.
    -   Connect the frontend to the backend API to display data.

4.  **Phase 4: Live Feed Integration**
    -   Investigate and implement a solution for streaming the RTSP feed to the web. This might involve a media server or a direct connection, and is often the most complex part.

By stalling this implementation, we can focus on delivering the core, high-priority detection and alerting system first.
