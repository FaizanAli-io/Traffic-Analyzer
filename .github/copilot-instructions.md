# Traffic Analyzer - AI Assistant Instructions

## Project Overview

A cloud-based traffic analysis system that processes video files using GPU instances on Lambda Labs. The system consists of a React frontend and Flask backend that orchestrates remote GPU processing for vehicle detection and motion analysis.

## Architecture & Key Components

### Backend (`backend/script.py`)

- **Flask API server** that manages Lambda Labs cloud GPU instances
- **Remote execution pipeline**: Launches instances → Uploads video → Runs DETR model → Downloads results
- **Authentication**: Uses environment variables (`USERID`, `PASSWORD`) for login validation
- **Instance management**: Tracks active instances in `instance_info.txt` file
- **SSH automation**: Uses paramiko/SCP for remote file operations

### Frontend (`frontend/src/`)

- **React + Vite** with React Router for SPA navigation
- **Authentication flow**: Login → Protected Dashboard with localStorage session management
- **Component structure**: Login, Dashboard, ProtectedRoute components in dedicated folders
- **API integration**: Uses `VITE_BACKEND_SERVER` env var, defaults to `localhost:5000`

### GPU Processing (`backend/detr_motion.py`)

- **DETR transformer model** for vehicle detection on remote GPU instances
- **Timestamp extraction** using OCR (pytesseract) from video frames
- **Motion filtering** to reduce false positives
- Deployed to Lambda Labs instances with `remote-requirements.txt`

## Development Workflows

### Quick Start Commands

```bash
# Backend (requires Python virtual env at backend/.venv)
run-be.bat

# Frontend
run-fe.bat
```

### Environment Setup

- **Backend**: Requires `.env` file with `LAMBDA_API_KEY`, `SSH_KEY_PATH`, `USERID`, `PASSWORD`
- **Frontend**: Optional `VITE_BACKEND_SERVER` for non-localhost backend
- **Remote GPU**: Separate requirements in `remote-requirements.txt` (torch, transformers, opencv)

## Critical Patterns & Conventions

### API Communication

- All frontend components use `const API_BASE = import.meta.env.VITE_BACKEND_SERVER || "http://localhost:5000"`
- Backend serves at port 5000 with CORS enabled
- RESTful endpoints: `/login`, `/lambda/*`, `/upload-video`, `/lambda/download-output`

### State Management

- **No global state library** - uses React hooks and localStorage
- **Authentication state**: Stored in localStorage as `userSession` JSON object
- **Instance tracking**: Backend persists instance info in text file, not database

### File Handling

- **Upload workflow**: Frontend → Backend → Remote GPU instance
- **Download workflow**: Remote GPU → Backend downloads dir → Frontend download
- **Fixed naming**: Remote video always renamed to `input_video_4.mp4`

### Remote GPU Orchestration

```python
# Typical workflow in backend/script.py
1. Launch Lambda Labs instance (launch_instance)
2. Setup environment via SSH (setup_environment_and_run)
3. Upload video via SCP (upload_video_to_instance)
4. Execute processing (run_script_on_instance)
5. Download results (download_output_from_instance)
6. Cleanup/terminate instance
```

### Component Architecture

- **Route protection**: `ProtectedRoute` wrapper checks localStorage authentication
- **Logout handling**: Clears localStorage and updates parent App state via callback
- **Error boundaries**: Each component handles its own API errors, no global error handling

## Key Integration Points

### Lambda Labs API

- Instance management via REST API (`LAMBDA_BASE_URL`)
- SSH key authentication for instance access
- Firewall rulesets configuration in backend constants

### File System Dependencies

- `backend/downloads/` for processed video output
- `backend/traffic.pem` SSH private key file
- `instance_info.txt` for persistence across backend restarts

### Environment Variables

```bash
# Required backend .env
LAMBDA_API_KEY=your_api_key
SSH_KEY_PATH=./traffic.pem
USERID=your_username
PASSWORD=your_password

# Optional frontend .env
VITE_BACKEND_SERVER=http://your-backend:5000
```

## Testing & Debugging

- **No automated tests** - manual testing via web interface
- **Backend logs**: Flask debug mode, check terminal output
- **Remote debugging**: SSH into Lambda instance, check `/home/ubuntu/myjob/` directory
- **Frontend debugging**: Browser dev tools, network tab for API calls

## Common Gotchas

- **Windows-specific**: Uses `.bat` files and backslash paths
- **Remote timeouts**: Long-running GPU processing may timeout connections
- **Instance cleanup**: Always terminate instances to avoid costs
- **Video format**: Processing script expects specific video formats/codecs
- **Memory management**: Large videos may cause OOM on GPU instances
