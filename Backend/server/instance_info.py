import os
from .config import INSTANCE_INFO_FILE


def save_instance_info(instance_id: str, ip: str):
    try:
        with open(INSTANCE_INFO_FILE, "w") as f:
            f.write(f"instance_id={instance_id}\n")
            f.write(f"ip={ip}\n")
    except Exception as e:
        print(f"Warning: Could not save instance info to {INSTANCE_INFO_FILE}: {e}")


def load_instance_info():
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
    _, ip = load_instance_info()
    return ip
