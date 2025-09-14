import os
from dotenv import load_dotenv

# Load env first (same as script.py)
load_dotenv()

# Lambda Cloud
LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
LAMBDA_BASE_URL = "https://cloud.lambdalabs.com/api/v1"

REGION_NAME = "us-east-3"
SSH_KEY_NAMES = ["Desktop-Pc"]
INSTANCE_NAME = "traffic-analysis"
INSTANCE_TYPE_NAME = "gpu_8x_a100"
FIREWALL_RULESETS = [{"id": "7093760c5d4e4df2bf8584ae791526c4"}]

# SSH config
SSH_USERNAME = "ubuntu"
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH")

# Directories anchored at backend root
# backend/refactored/config.py → backend (parent of this file's dir)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS_DIR = os.path.join(ROOT_DIR, "downloads")
INPUT_VIDEO_DIR = os.path.join(ROOT_DIR, "input-video")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Files
OUTPUT_DEFAULT_NAME = "output_detr_motion_filtered.mp4"
INSTANCE_INFO_FILE = os.path.join(ROOT_DIR, "instance_info.txt")

# Auth
ENV_USERNAME = os.getenv("USERID", "")
ENV_PASSWORD = os.getenv("PASSWORD", "")

# Remote
REMOTE_DIR = "/home/ubuntu/myjob"
REMOTE_VENV = f"{REMOTE_DIR}/venv"
REMOTE_SCRIPT_NAME = "detr_motion.py"
REMOTE_REQ_NAME = "requirements.txt"
REMOTE_VIDEO_NAME = "input_video_4.mp4"

# Behavior toggles
RUN_AFTER_SETUP = False
FIXED_VIDEO_BASENAME = "input_video_4.mp4"
