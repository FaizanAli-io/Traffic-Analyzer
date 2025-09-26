import os
from .config import (
    ROOT_DIR,
    REMOTE_DIR,
    REMOTE_VENV,
    DOWNLOADS_DIR,
    INPUT_VIDEO_DIR,
    REMOTE_REQ_NAME,
    REMOTE_VIDEO_NAME,
)
from .instance_info import get_saved_ip
from .ssh_utils import make_ssh, sh, wait_for_ssh


def bootstrap_system_and_venv(ssh):
    sh(ssh, "sudo apt-get update -y")
    sh(
        ssh,
        "sudo apt-get install -y --no-install-recommends python3-venv python3-pip tesseract-ocr libtesseract-dev ffmpeg",
    )
    sh(ssh, f"mkdir -p {REMOTE_DIR}")
    sh(ssh, f"python3 -m venv {REMOTE_VENV}")
    sh(
        ssh,
        f"source {REMOTE_VENV}/bin/activate && pip install --upgrade pip setuptools wheel",
    )


def upload_code_and_requirements_from_cwd(ssh, scp):
    """Upload the analyzer package (backend/analyzer) and requirements to the remote.

    We no longer upload detr_motion.py directly. Instead, we place the minimal
    package structure so the remote can run `python -m backend.analyzer`.
    """
    # Paths in local backend folder
    backend_pkg_init = os.path.join(ROOT_DIR, "__init__.py")
    analyzer_dir = os.path.join(ROOT_DIR, "analyzer")
    local_req = os.path.join(ROOT_DIR, "remote-requirements.txt")

    # Validate expected files/folders
    if not os.path.exists(backend_pkg_init):
        raise FileNotFoundError(f"Missing backend __init__.py: {backend_pkg_init}")
    if not os.path.isdir(analyzer_dir):
        raise FileNotFoundError(f"Missing analyzer package directory: {analyzer_dir}")
    if not os.path.exists(local_req):
        raise FileNotFoundError(f"Expected file not found: {local_req}")

    # Ensure remote structure exists
    sh(ssh, f"mkdir -p {REMOTE_DIR}/backend")

    # Upload backend package init and analyzer directory recursively
    scp.put(backend_pkg_init, f"{REMOTE_DIR}/backend/__init__.py")
    scp.put(analyzer_dir, f"{REMOTE_DIR}/backend/", recursive=True)

    # Upload requirements for remote pip install
    scp.put(local_req, f"{REMOTE_DIR}/{REMOTE_REQ_NAME}")


def pip_install_requirements(ssh):
    code, out, err = sh(
        ssh,
        f"source {REMOTE_VENV}/bin/activate && cd {REMOTE_DIR} && pip install -r {REMOTE_REQ_NAME}",
    )
    if code != 0:
        raise RuntimeError(f"pip install -r failed\nSTDOUT:\n{out}\nSTDERR:\n{err}")


def run_script(ssh):
    # Run analyzer as a module so Python can resolve backend/analyzer package
    code, out, err = sh(
        ssh,
        f"cd {REMOTE_DIR} && source {REMOTE_VENV}/bin/activate && python -m backend.analyzer",
    )
    if code != 0:
        raise RuntimeError(f"Script failed\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out


def _pick_local_video(video_filename: str | None):
    if not os.path.isdir(INPUT_VIDEO_DIR):
        raise FileNotFoundError(f"Input folder not found: {INPUT_VIDEO_DIR}")
    if video_filename:
        local_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Video not found: {local_path}")
        return local_path, video_filename
    for name in os.listdir(INPUT_VIDEO_DIR):
        if name.lower().endswith(".mp4"):
            return os.path.join(INPUT_VIDEO_DIR, name), name
    raise FileNotFoundError(f"No .mp4 files found in {INPUT_VIDEO_DIR}")


def cleanup_remote_files(ssh):
    try:
        sh(ssh, f"find {REMOTE_DIR} -name '*.mp4' -type f -delete")
        sh(ssh, f"find {REMOTE_DIR} -name '*.csv' -type f -delete")
        sh(ssh, f"find {REMOTE_DIR} -name '*.log' -type f -delete")
        return True
    except Exception:
        return False


def upload_video_and_run(
    ip: str | None = None,
    video_filename: str | None = None,
    direction_orientation: int = 0,
):
    target_ip = ip or get_saved_ip()
    if not target_ip:
        raise ValueError("No IP provided. Pass ip or launch an instance first.")

    local_path, _ = _pick_local_video(video_filename)

    if not wait_for_ssh(target_ip, timeout=600):
        raise RuntimeError(f"SSH not reachable on {target_ip}")

    ssh, scp = make_ssh(target_ip)
    try:
        sh(ssh, f"mkdir -p {REMOTE_DIR}")
        cleanup_remote_files(ssh)
        remote_path = f"{REMOTE_DIR}/{REMOTE_VIDEO_NAME}"
        scp.put(local_path, remote_path)

        remote_log = (
            f"{REMOTE_DIR}/detr_motion.log"  # keep legacy log name for continuity
        )
        cmd = (
            f"cd {REMOTE_DIR} && "
            f"source {REMOTE_VENV}/bin/activate && "
            f"pip install --no-cache-dir -r {REMOTE_REQ_NAME} && "
            f"python -m backend.analyzer --direction-orientation {direction_orientation} "
            f"> {remote_log} 2>&1"
        )
        code, out, err = sh(ssh, cmd)
        try:
            _, tail_out, _ = sh(ssh, f"tail -n 50 {remote_log}")
        except Exception:
            tail_out = "Could not read log file"

        if code != 0:
            raise RuntimeError(
                f"Script failed (exit {code})\nCheck logs at {remote_log}\nLast output:\n{tail_out}"
            )
        return {
            "status": "ok",
            "ip": target_ip,
            "uploaded": {"local": local_path, "remote": remote_path},
            "log_file": remote_log,
            "stdout_tail": tail_out,
            "cleanup_performed": True,
            "message": "Previous files cleaned up and new video processed successfully",
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


def pull_remote_file(ip: str | None, remote_filename: str) -> str:
    from .ssh_utils import make_ssh

    target_ip = ip or get_saved_ip()
    if not target_ip:
        raise ValueError("No IP provided. Pass 'ip' or launch an instance first.")
    if not wait_for_ssh(target_ip, timeout=600):
        raise RuntimeError(f"SSH not reachable on {target_ip}")
    ssh, scp = make_ssh(target_ip)
    try:
        remote_path = f"{REMOTE_DIR}/{remote_filename}"
        local_path = os.path.join(DOWNLOADS_DIR, remote_filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        scp.get(remote_path, local_path)
        return local_path
    finally:
        try:
            scp.close()
        except Exception:
            pass
        try:
            ssh.close()
        except Exception:
            pass


def replace_remote_file(
    ip: str | None = None,
    local_script: str | None = None,
    remote_script: str | None = None,
):
    from .config import ROOT_DIR, REMOTE_DIR

    target_ip = ip or get_saved_ip()
    if not target_ip:
        raise ValueError("No IP provided. Pass 'ip' or launch an instance first.")
    local_name = local_script or "detr_motion.py"
    local_path = os.path.join(ROOT_DIR, local_name)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")
    remote_name = remote_script or "detr_motion.py"
    remote_path = f"{REMOTE_DIR}/{remote_name}"
    if not wait_for_ssh(target_ip, timeout=600):
        raise RuntimeError(f"SSH not reachable on {target_ip}")
    ssh, scp = make_ssh(target_ip)
    try:
        sh(ssh, f"mkdir -p {REMOTE_DIR}")
        sh(ssh, f"test -f {remote_path} && rm -f {remote_path} || true")
        scp.put(local_path, remote_path)
        return {
            "status": "ok",
            "ip": target_ip,
            "remote_path": remote_path,
            "note": "File replaced successfully.",
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
