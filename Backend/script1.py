import os
import requests
import time
import socket
import paramiko
from scp import SCPClient
from flask import Flask, jsonify, request, send_file
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
LAMBDA_BASE_URL = "https://cloud.lambdalabs.com/api/v1"

INSTANCE_TYPE_NAME = "gpu_1x_gh200"
REGION_NAME = "us-east-3"
SSH_KEY_NAMES = ["Desktop-Pc"]
INSTANCE_NAME = "traffic-analysis-1"
FIREWALL_RULESETS = [{"id": "7093760c5d4e4df2bf8584ae791526c4"}]

# SSH config (update with your key and username)
SSH_USERNAME = "ubuntu"   # usually "ubuntu"
SSH_KEY_PATH = "C:/Users/hamza/Desktop/Desktop-Pc.pem"

# Directory configuration
DOWNLOADS_DIR = os.path.join(os.getcwd(), "downloads")
INPUT_VIDEO_DIR = os.path.join(os.getcwd(), "input-video")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# File names
OUTPUT_DEFAULT_NAME = "output_detr_motion_filtered.mp4"
INSTANCE_INFO_FILE = os.path.join(os.getcwd(), "instance_info.txt")

# Authentication
ENV_USERNAME = os.getenv("USERID", "")
ENV_PASSWORD = os.getenv("PASSWORD", "")

# Remote configuration
REMOTE_DIR = "/home/ubuntu/myjob"
REMOTE_VENV = f"{REMOTE_DIR}/venv"
REMOTE_SCRIPT_NAME = "detr_motion.py"
REMOTE_REQ_NAME = "requirements.txt"
REMOTE_VIDEO_NAME = "input_video_4.mp4"

# Toggle running after setup
RUN_AFTER_SETUP = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# ===============================
# INSTANCE INFO MANAGEMENT
# ===============================

def save_instance_info(instance_id, ip):
    """Save instance ID and IP to a text file."""
    try:
        with open(INSTANCE_INFO_FILE, "w") as f:
            f.write(f"instance_id={instance_id}\n")
            f.write(f"ip={ip}\n")
    except Exception as e:
        print(f"Warning: Could not save instance info to {INSTANCE_INFO_FILE}: {e}")

def load_instance_info():
    """Load instance ID and IP from text file. Returns (instance_id, ip) or (None, None)."""
    try:
        if os.path.exists(INSTANCE_INFO_FILE):
            with open(INSTANCE_INFO_FILE, "r") as f:
                lines = f.read().strip().split("\n")
                info = {}
                for line in lines:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        info[key.strip()] = value.strip()
                return info.get("instance_id"), info.get("ip")
    except Exception as e:
        print(f"Warning: Could not load instance info from {INSTANCE_INFO_FILE}: {e}")
    return None, None

def get_saved_ip():
    """Get the saved IP from instance info file."""
    _, ip = load_instance_info()
    return ip


# ===============================
# LAMBDA CLOUD API HELPERS
# ===============================

def cloud_get(path: str):
    """Make GET request to Lambda Cloud API."""
    if not LAMBDA_API_KEY:
        return type('MockResponse', (), {
            'ok': False,
            'status_code': 401,
            'text': '{"error": "No API key provided"}',
            'json': lambda: {"error": "No API key provided"}
        })()
    headers = {"Authorization": f"Bearer {LAMBDA_API_KEY}"}
    return requests.get(f"{LAMBDA_BASE_URL}{path}", headers=headers, timeout=30)

def cloud_post(path: str, body: dict):
    """Make POST request to Lambda Cloud API."""
    if not LAMBDA_API_KEY:
        return type('MockResponse', (), {
            'ok': False,
            'status_code': 401,
            'text': '{"error": "No API key provided"}',
            'json': lambda: {"error": "No API key provided"}
        })()
    headers = {
        "Authorization": f"Bearer {LAMBDA_API_KEY}",
        "Content-Type": "application/json"
    }
    return requests.post(f"{LAMBDA_BASE_URL}{path}", headers=headers, json=body, timeout=30)

def extract_instance_id(launch_json):
    """
    Supports common launch response shapes:
    - {"data":{"instance_ids":[id,...]}}
    - {"instances":[{"id": id, ...}]}
    - {"instance":{"id": id, ...}}
    - {"data":{"id": id, ...}}
    """
    j = launch_json or {}
    data = j.get("data") or {}

    ids = data.get("instance_ids")
    if isinstance(ids, list) and ids:
        return ids[0]

    if isinstance(j.get("instances"), list) and j["instances"]:
        return j["instances"][0].get("id")

    if isinstance(j.get("instance"), dict):
        return j["instance"].get("id")

    if isinstance(data, dict) and data.get("id"):
        return data.get("id")

    return None

def wait_for_instance_ip(instance_id: str, timeout: int = 600, interval: int = 5):
    """
    Poll GET /instances/{id} until BOTH conditions are met:
      1) status == 'active'
      2) ip (or ipv4) is present

    Allowed statuses per API: 'booting', 'active', 'unhealthy', 'terminated'.
    Returns the IP string on success; raises RuntimeError on failure/timeout.
    """
    deadline = time.time() + timeout
    sleep_s = max(1, interval)
    last_status = None
    last_ip = None

    while time.time() < deadline:
        r = cloud_get(f"/instances/{instance_id}")
        if r.ok:
            payload = r.json()
            inst = payload.get("data") or payload

            status = (inst.get("status") or "").lower()
            ip = inst.get("ip") or inst.get("ipv4")

            last_status = status
            last_ip = ip

            # success condition: ACTIVE + IP assigned
            if status == "active" and ip:
                return ip

            # terminal / error states per spec
            if status in ("terminated", "unhealthy"):
                raise RuntimeError(
                    f"Instance {instance_id} reached terminal status '{status}' "
                    f"(last_ip={ip!r})."
                )

            # otherwise: 'booting' → keep waiting
        # if not r.ok: likely transient; keep trying

        time.sleep(sleep_s)
        sleep_s = min(int(sleep_s * 1.3) or 1, 15)

    raise RuntimeError(
        f"Timed out waiting for instance {instance_id} to be 'active' with an IP. "
        f"Last seen status={last_status!r}, ip={last_ip!r}."
    )


# ===============================
# SSH CONNECTION HELPERS
# ===============================

def _ssh_key():
    """Load SSH private key (try RSA first, then Ed25519)."""
    try:
        return paramiko.RSAKey.from_private_key_file(SSH_KEY_PATH)
    except Exception:
        return paramiko.Ed25519Key.from_private_key_file(SSH_KEY_PATH)

def _mk_ssh(ip):
    """Create SSH and SCP connections."""
    key = _ssh_key()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=SSH_USERNAME, pkey=key, timeout=30)
    scp = SCPClient(ssh.get_transport())
    return ssh, scp

def _sh(ssh, cmd):
    """Run a bash command and return (code, out, err)."""
    cmd = f"bash -lc '{cmd}'"
    _, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    code = stdout.channel.recv_exit_status()
    return code, out, err

def wait_for_ssh(ip, port=22, timeout=300):
    """Wait until instance SSH port is reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((ip, port), timeout=5):
                return True
        except OSError:
            time.sleep(5)
    return False


# ===============================
# REMOTE INSTANCE SETUP
# ===============================

def create_remote_folder(ip, folder="/home/ubuntu/myjob"):
    """Connect to instance via SSH and create a folder."""
    key = _ssh_key()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=SSH_USERNAME, pkey=key, timeout=30)
    cmd = f"mkdir -p {folder}"
    ssh.exec_command(cmd)
    ssh.close()
    return folder

def bootstrap_system_and_venv(ssh):
    """Install system dependencies and create Python virtual environment."""
    # System deps
    _sh(ssh, "sudo apt-get update -y")
    _sh(ssh, "sudo apt-get install -y --no-install-recommends python3-venv python3-pip tesseract-ocr libtesseract-dev ffmpeg")
    # Project dir + venv
    _sh(ssh, f"mkdir -p {REMOTE_DIR}")
    _sh(ssh, f"python3 -m venv {REMOTE_VENV}")
    _sh(ssh, f"source {REMOTE_VENV}/bin/activate && pip install --upgrade pip setuptools wheel")

def upload_code_and_requirements_from_cwd(ssh, scp):
    """Upload ./detr_motion.py and ./remote-requirements.txt from current working dir."""
    cwd = os.getcwd()
    local_script = os.path.join(cwd, "detr_motion.py")
    local_req = os.path.join(cwd, "remote-requirements.txt")

    if not os.path.exists(local_script):
        raise FileNotFoundError(f"Expected file not found in current dir: {local_script}")
    if not os.path.exists(local_req):
        raise FileNotFoundError(f"Expected file not found in current dir: {local_req}")

    _sh(ssh, f"mkdir -p {REMOTE_DIR}")
    scp.put(local_script, f"{REMOTE_DIR}/{REMOTE_SCRIPT_NAME}")
    scp.put(local_req, f"{REMOTE_DIR}/{REMOTE_REQ_NAME}")

def pip_install_requirements(ssh):
    """Install Python requirements in the virtual environment."""
    code, out, err = _sh(ssh, f"source {REMOTE_VENV}/bin/activate && cd {REMOTE_DIR} && pip install -r {REMOTE_REQ_NAME}")
    if code != 0:
        raise RuntimeError(f"pip install -r failed\nSTDOUT:\n{out}\nSTDERR:\n{err}")

def run_script(ssh):
    """Run the main script in the virtual environment."""
    code, out, err = _sh(ssh, f"cd {REMOTE_DIR} && source {REMOTE_VENV}/bin/activate && python {REMOTE_SCRIPT_NAME}")
    if code != 0:
        raise RuntimeError(f"Script failed\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out


# ===============================
# FILE MANAGEMENT HELPERS
# ===============================

def _pick_local_video(video_filename: str | None):
    """
    Pick a local video from ./input-video.
    - If video_filename is given, use that (must exist in INPUT_VIDEO_DIR).
    - Else, pick the first .mp4 in the folder.
    Returns absolute path to the file and the chosen filename.
    """
    if not os.path.isdir(INPUT_VIDEO_DIR):
        raise FileNotFoundError(f"Input folder not found: {INPUT_VIDEO_DIR}")

    if video_filename:
        local_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Video not found: {local_path}")
        return local_path, video_filename

    # Auto-pick first .mp4
    for name in os.listdir(INPUT_VIDEO_DIR):
        if name.lower().endswith(".mp4"):
            return os.path.join(INPUT_VIDEO_DIR, name), name

    raise FileNotFoundError(f"No .mp4 files found in {INPUT_VIDEO_DIR}")

def replace_remote_file(ip: str | None = None,
                        local_script: str | None = None,
                        remote_script: str | None = None):
    """Replace a file on the remote instance."""
    target_ip = ip or get_saved_ip()
    if not target_ip:
        raise ValueError("No IP provided. Pass 'ip' or launch an instance first.")

    local_name = local_script or "detr_motion.py"
    local_path = os.path.join(os.getcwd(), local_name)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    remote_name = remote_script or REMOTE_SCRIPT_NAME
    remote_path = f"{REMOTE_DIR}/{remote_name}"

    # Ensure SSH is reachable
    if not wait_for_ssh(target_ip, timeout=600):
        raise RuntimeError(f"SSH not reachable on {target_ip}")

    ssh, scp = _mk_ssh(target_ip)
    try:
        # Ensure directory exists
        _sh(ssh, f"mkdir -p {REMOTE_DIR}")

        # Delete if exists
        _sh(ssh, f"test -f {remote_path} && rm -f {remote_path} || true")

        # Upload new file
        scp.put(local_path, remote_path)

        return {
            "status": "ok",
            "ip": target_ip,
            "remote_path": remote_path,
            "note": "File replaced successfully."
        }
    finally:
        try: scp.close()
        except Exception: pass
        try: ssh.close()
        except Exception: pass

def pull_remote_file(ip: str | None, remote_filename: str) -> str:
    """SCP a file from REMOTE_DIR on the instance to local DOWNLOADS_DIR."""
    target_ip = ip or get_saved_ip()
    if not target_ip:
        raise ValueError("No IP provided. Pass 'ip' or launch an instance first.")

    if not wait_for_ssh(target_ip, timeout=600):
        raise RuntimeError(f"SSH not reachable on {target_ip}")

    ssh, scp = _mk_ssh(target_ip)
    try:
        remote_path = f"{REMOTE_DIR}/{remote_filename}"
        local_path = os.path.join(DOWNLOADS_DIR, remote_filename)

        # ensure local folder exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # fetch the file
        scp.get(remote_path, local_path)
        return local_path
    finally:
        try: scp.close()
        except Exception: pass
        try: ssh.close()
        except Exception: pass


# ===============================
# CORE WORKFLOW FUNCTIONS
# ===============================

def upload_video_and_run(ip: str | None = None, video_filename: str | None = None):
    """
    Upload a video from ./input-video to the remote working folder and run detr_motion.py.
    - ip: the instance IP to SSH into. If None, uses saved IP from instance_info.txt.
    - video_filename: name inside ./input-video; if None, pick first .mp4 found.
    Returns a summary dict (steps, stdout tail, etc.).
    """
    target_ip = ip or get_saved_ip()
    if not target_ip:
        raise ValueError("No IP provided. Pass ip in the request body or launch an instance first.")

    # 0) pick local video
    local_path, chosen_name = _pick_local_video(video_filename)

    # 1) wait for SSH (in case IP is new)
    if not wait_for_ssh(target_ip, timeout=600):
        raise RuntimeError(f"SSH not reachable on {target_ip}")

    # 2) connect
    ssh, scp = _mk_ssh(target_ip)

    try:
        # 3) ensure project dir exists
        _sh(ssh, f"mkdir -p {REMOTE_DIR}")

        # 4) upload video (rename remotely if your script expects a specific name)
        remote_path = f"{REMOTE_DIR}/{REMOTE_VIDEO_NAME}"
        scp.put(local_path, remote_path)

        # 5) run inside venv
        cmd = (
            f"cd {REMOTE_DIR} && "
            f"source {REMOTE_VENV}/bin/activate && "
            f"pip install --no-cache-dir -r {REMOTE_REQ_NAME} && "
            f"python {REMOTE_SCRIPT_NAME}"
        )

        code, out, err = _sh(ssh, cmd)
        if code != 0:
            raise RuntimeError(f"Script failed (exit {code})\nSTDOUT:\n{out}\nSTDERR:\n{err}")

        return {
            "status": "ok",
            "ip": target_ip,
            "uploaded": {
                "local": local_path,
                "remote": remote_path
            },
            "stdout_tail": "\n".join(out.splitlines()[-50:])
        }
    finally:
        try:
            scp.close()
        except Exception:
            pass
        try:
            ssh.close()
        except Exception:
            pass


# ===============================
# FLASK ROUTES - AUTHENTICATION
# ===============================

@app.post("/login")
def auth_login():
    """
    POST JSON: {"username": "...", "password": "..."}
    Compares against USERNAME/PASSWORD from environment.
    Returns 200 on success, 401 on failure.
    """
    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "username and password required"}), 400

    if username == ENV_USERNAME and password == ENV_PASSWORD:
        return jsonify({"success": True, "message": "login ok"}), 200

    return jsonify({"success": False, "error": "invalid credentials"}), 401


# ===============================
# FLASK ROUTES - HEALTH & INFO
# ===============================

@app.get("/health")
def health():
    """Health check endpoint."""
    api_key_status = "SET" if LAMBDA_API_KEY else "NOT SET"
    api_key_preview = f"{LAMBDA_API_KEY[:8]}..." if LAMBDA_API_KEY else "None"

    return jsonify({
        "status": "ok",
        "service": "lambda-min-api",
        "api_key_status": api_key_status,
        "api_key_preview": api_key_preview
    })

@app.get("/lambda/firewall-rulesets")
def list_firewall_rulesets():
    """
    Returns all firewall rulesets from Lambda Cloud (IDs included).
    Tip: filter on region to match your instance region (e.g., us-east-3).
    """
    r = cloud_get("/firewall-rulesets")
    return (r.text, r.status_code, {"Content-Type": "application/json"})

@app.get("/lambda/instance-types")
def list_instance_types():
    """Proxy: return Lambda Labs /instance-types response as-is."""
    r = cloud_get("/instance-types")
    return (r.text, r.status_code, {"Content-Type": "application/json"})


# ===============================
# FLASK ROUTES - INSTANCE MANAGEMENT
# ===============================

@app.post("/lambda/launch-and-setup")
def launch_and_setup():
    """
    Launch → wait SSH → bootstrap (apt+venv) → upload ./detr_motion.py and ./remote-requirements.txt
    → pip install -r requirements.txt → (optional) run script
    """
    # 1) Launch
    body = {
        "region_name": REGION_NAME,
        "instance_type_name": INSTANCE_TYPE_NAME,
        "ssh_key_names": SSH_KEY_NAMES,
        "name": INSTANCE_NAME,
        "firewall_rulesets": FIREWALL_RULESETS
    }
    body = {k: v for k, v in body.items() if v not in ("", [], {})}
    r = cloud_post("/instance-operations/launch", body)
    if not r.ok:
        return (r.text, r.status_code, {"Content-Type": "application/json"})

    payload = r.json()
    instance_id = extract_instance_id(payload)
    if not instance_id:
        return jsonify({"error": "No instance id in launch response", "raw": payload}), 500

    # Poll until IP is assigned
    try:
        ip = wait_for_instance_ip(instance_id, timeout=600, interval=5)
        # Save instance info to file
        save_instance_info(instance_id, ip)
    except RuntimeError as e:
        return jsonify({"error": str(e), "instance_id": instance_id, "raw": payload}), 504

    # 2) Wait for SSH
    if not wait_for_ssh(ip, timeout=600):
        return jsonify({"error": "SSH not ready", "ip": ip}), 504

    summary = {"status": "launched+provisioning", "ip": ip, "remote_dir": REMOTE_DIR, "steps": []}
    try:
        ssh, scp = _mk_ssh(ip)

        # 3) Bootstrap
        bootstrap_system_and_venv(ssh)
        summary["steps"].append({"bootstrap": "ok"})

        # 4) Upload code + requirements from current directory
        upload_code_and_requirements_from_cwd(ssh, scp)
        summary["steps"].append({"upload": {"script": "detr_motion.py", "requirements": "remote-requirements.txt → requirements.txt"}})

        # 5) pip install -r
        pip_install_requirements(ssh)
        summary["steps"].append({"pip_install": "ok"})

        # 6) Optional: run the script
        if RUN_AFTER_SETUP:
            out = run_script(ssh)
            summary["steps"].append({"run": "ok", "stdout_tail": "\n".join(out.splitlines()[-20:])})

        scp.close()
        ssh.close()
        summary["status"] = "launched+provisioned"
        return jsonify(summary)

    except Exception as e:
        summary["status"] = "error"
        summary["message"] = str(e)
        return jsonify(summary), 500


# ===============================
# FLASK ROUTES - FILE OPERATIONS
# ===============================

@app.post("/lambda/upload-video-and-run")
def api_upload_video_and_run():
    """
    Body (JSON) optional:
      {
        "ip": "1.2.3.4",            // if omitted, uses saved IP from instance_info.txt
        "video": "input_video_4.mp4" // if omitted, picks first .mp4 in ./input-video
      }
    """
    try:
        body = request.get_json(silent=True) or {}
        ip = body.get("ip") or None
        video = body.get("video") or None

        result = upload_video_and_run(ip=ip, video_filename=video)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.post("/lambda/replace-file")
def api_replace_file():
    """
    Replace a file on the remote instance.

    JSON body (all optional):
    {
      "ip": "1.2.3.4",              # if omitted, uses saved IP from instance_info.txt
      "local_script": "detr_motion.py",  # local file in cwd (default: detr_motion.py)
      "remote_script": "detr_motion.py"  # remote filename in REMOTE_DIR (default: REMOTE_SCRIPT_NAME)
    }
    """
    try:
        body = request.get_json(silent=True) or {}
        res = replace_remote_file(
            ip=body.get("ip"),
            local_script=body.get("local_script"),
            remote_script=body.get("remote_script"),
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.get("/lambda/download-output")
def api_download_output():
    """
    Download an output file from the instance.

    Query params (optional):
      ip   = instance IP (uses saved IP from instance_info.txt if omitted)
      name = remote filename in REMOTE_DIR (default: output_detr_motion_filtered.mp4)

    Example:
      GET /lambda/download-output?ip=12.34.56.78
      GET /lambda/download-output?name=custom_output.mp4
    """
    try:
        ip = request.args.get("ip") or None
        name = request.args.get("name") or OUTPUT_DEFAULT_NAME

        local_path = pull_remote_file(ip=ip, remote_filename=name)
        if not os.path.exists(local_path):
            return jsonify({"status": "error", "message": f"File not found after download: {local_path}"}), 404

        # stream the file back to the client
        return send_file(local_path, as_attachment=True, download_name=name)
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.post("/upload-video")
def upload_video():
    """
    Upload a video file to the input-video directory.
    Expects a multipart/form-data request with 'video' file field.
    """
    try:
        if 'video' not in request.files:
            return jsonify({"status": "error", "message": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Validate file type
        allowed_extensions = {'.mp4', '.mov', '.avi'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({"status": "error", "message": "Invalid file type. Only MP4, MOV, AVI allowed"}), 400
        
        # Ensure input-video directory exists
        os.makedirs(INPUT_VIDEO_DIR, exist_ok=True)
        
        # Save file to input-video directory
        filepath = os.path.join(INPUT_VIDEO_DIR, file.filename)
        file.save(filepath)
        
        return jsonify({
            "status": "ok",
            "message": "Video uploaded successfully",
            "filename": file.filename,
            "path": filepath
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print(f"LAMBDA_API_KEY status: {'SET' if LAMBDA_API_KEY else 'NOT SET'}")
    app.run(host="0.0.0.0", port=5000, debug=True)