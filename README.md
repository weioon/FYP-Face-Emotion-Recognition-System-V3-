# EmotionWave - Live Class Monitoring System using Face Emotion Recognition

EmotionWave is a web application designed to analyze and interpret emotions from facial expressions in real-time. It aims to provide insights into emotional states, particularly in educational or interactive settings, to help understand engagement and tailor approaches accordingly.

## Core Features

*   **User Authentication:** Secure registration and login for users.
*   **Real-time Emotion Detection:** Captures video from the user's webcam and analyzes facial expressions to detect emotions.
*   **Comprehensive Dashboard:** Displays:
    *   Dominant emotions detected.
    *   An "Emotional Journey" showing emotion distribution over the beginning, middle, and end of a session.
    *   Textual interpretation of the emotional patterns.
    *   Educational recommendations based on the detected emotions.
*   **Recording History:** Users can save their emotion analysis sessions and review them later.
*   **Detailed Session Review:** View detailed charts and analyses for past recordings.

## How It Works

1.  **User Interaction (Frontend):**
    *   Users register or log in via the React frontend.
    *   Authenticated users can start an emotion detection session. The frontend captures webcam video frames.
2.  **API Communication:**
    *   Video frames (as images) are sent to the FastAPI backend API.
3.  **Backend Processing:**
    *   The backend receives the image data.
    *   **Face Detection:** A YOLOv8 model (`yolov8n.pt`) is used to detect faces in the image.
    *   **Emotion Analysis:** For each detected face, the `DeepFace` library is utilized to analyze and predict emotions.
    *   The backend aggregates emotion data, generates interpretations, and formulates educational recommendations.
4.  **Displaying Results (Frontend):**
    *   The analysis results are sent back to the React frontend.
    *   The `EmotionDashboard` component displays the information using charts (Chart.js) and textual summaries.
5.  **Session Recording:**
    *   Users can opt to save the session data, which is stored in the database via the backend.
    *   Past recordings can be accessed and reviewed through the `RecordingHistory` and `RecordingDetail` components.

## Technology Stack

**Frontend:**
*   **Language:** JavaScript
*   **Library/Framework:** React (Create React App)
*   **HTTP Client:** Axios
*   **Routing:** React Router
*   **Charting:** Chart.js
*   **Styling:** CSS, Font Awesome

**Backend:**
*   **Language:** Python
*   **Framework:** FastAPI
*   **Face Detection:** YOLOv8 (via `ultralytics` library)
*   **Emotion Analysis:** `DeepFace` library
*   **Database ORM:** SQLAlchemy
*   **Authentication:** OAuth2 with JWT tokens (`python-jose`, `passlib[bcrypt]`)
*   **Image Processing:** OpenCV

**Database:**
*   SQLite (default, for ease of development, e.g., `emotion_recognition.db`)
*   PostgreSQL (supported, requires `psycopg2-binary` and proper `DATABASE_URL` configuration)

**Containerization & Deployment:**
*   **Containerization:** Docker, Docker Compose
*   **Web Server/Proxy (for frontend):** Nginx

## Project Structure

```
.
├── Backend/                  # FastAPI application, emotion analysis logic
│   ├── api.py                # Main API endpoints
│   ├── realtime_emotion.py   # Core emotion detection class using DeepFace & YOLO
│   ├── models.py             # SQLAlchemy database models
│   ├── db.py                 # Database session and engine setup
│   ├── requirements.txt      # Python dependencies
│   ├── .env                  # Environment variables for backend (DATABASE_URL, SECRET_KEY, etc.)
│   └── yolov8n.pt            # YOLOv8 model file
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/       # React components (Login, EmotionDetector, Dashboard, etc.)
│   │   ├── App.js            # Main application component with routing
│   │   └── App.css           # Global styles
│   ├── public/
│   ├── Dockerfile            # Dockerfile for frontend
│   ├── nginx.conf            # Nginx configuration for serving frontend
│   └── .env                  # Environment variables for frontend (REACT_APP_API_URL)
├── docker-compose.yml        # Docker Compose configuration to run services
├── Dockerfile                # Dockerfile for backend
├── yolov8n.pt                # YOLOv8 model file (can be a duplicate or centrally located)
└── README.md                 # This file
```

## Setup and Installation

### Prerequisites
*   Docker
*   Docker Compose

### Running the Application

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Configure Environment Variables:**
    *   **Backend:** Create a `.env` file in the `Backend/` directory. Example:
        ```env
        # Backend/.env
        DATABASE_URL="sqlite:///./emotion_recognition.db" # For SQLite
        # DATABASE_URL="postgresql://user:password@host:port/database" # For PostgreSQL
        SECRET_KEY=your_very_secret_key_for_jwt
        ALGORITHM=HS256
        ACCESS_TOKEN_EXPIRE_MINUTES=30
        ```
    *   **Frontend:** Create a `.env` file in the `frontend/` directory. Example:
        ```env
        # frontend/.env
        REACT_APP_API_URL=http://localhost:8000/api
        ```
        (Adjust `REACT_APP_API_URL` if your backend runs on a different port or domain when accessed from the frontend container/browser).

3.  **Build and Run with Docker Compose:**
    From the root directory of the project:
    ```bash
    docker-compose up --build
    ```

4.  **Access the application:**
    *   Frontend: Open your browser and navigate to `http://localhost:3000` (or the port mapped in `docker-compose.yml` for the frontend service, often port 80 if Nginx is directly exposed).
    *   Backend API (for testing): Accessible at `http://localhost:8000` (or the port mapped for the backend service).

## Notes

*   The system uses `DeepFace` for emotion analysis in the primary backend API (`Backend/realtime_emotion.py`).
*   A custom TensorFlow/Keras model (`Backend/Final_model.h5`) and training script (`train_model.py`) exist in the repository, which seems to be part of an alternative implementation or experimental script (`app.py`) and is not the primary model used by the main FastAPI backend.
*   Ensure the `yolov8n.pt` model file is available in the `Backend/` directory (or accessible as per paths in the code) for face detection.
