import time
import requests
from .config import LAMBDA_API_KEY, LAMBDA_BASE_URL


def cloud_get(path: str):
    if not LAMBDA_API_KEY:
        return type(
            "MockResponse",
            (),
            {
                "ok": False,
                "status_code": 401,
                "text": '{"error": "No API key provided"}',
                "json": lambda: {"error": "No API key provided"},
            },
        )()
    headers = {"Authorization": f"Bearer {LAMBDA_API_KEY}"}
    return requests.get(f"{LAMBDA_BASE_URL}{path}", headers=headers, timeout=30)


def cloud_post(path: str, body: dict):
    if not LAMBDA_API_KEY:
        return type(
            "MockResponse",
            (),
            {
                "ok": False,
                "status_code": 401,
                "text": '{"error": "No API key provided"}',
                "json": lambda: {"error": "No API key provided"},
            },
        )()
    headers = {
        "Authorization": f"Bearer {LAMBDA_API_KEY}",
        "Content-Type": "application/json",
    }
    return requests.post(
        f"{LAMBDA_BASE_URL}{path}", headers=headers, json=body, timeout=30
    )


def extract_instance_id(launch_json):
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


def wait_for_instance_ip(
    get_fn, instance_id: str, timeout: int = 600, interval: int = 5
):
    deadline = time.time() + timeout
    sleep_s = max(1, interval)
    last_status = None
    last_ip = None

    while time.time() < deadline:
        r = get_fn(f"/instances/{instance_id}")
        if r.ok:
            payload = r.json()
            inst = payload.get("data") or payload

            status = (inst.get("status") or "").lower()
            ip = inst.get("ip") or inst.get("ipv4")

            last_status = status
            last_ip = ip

            if status == "active" and ip:
                return ip
            if status in ("terminated", "unhealthy"):
                raise RuntimeError(
                    f"Instance {instance_id} reached terminal status '{status}' (last_ip={ip!r})."
                )
        time.sleep(sleep_s)
        sleep_s = min(int(sleep_s * 1.3) or 1, 15)

    raise RuntimeError(
        f"Timed out waiting for instance {instance_id} to be 'active' with an IP. Last seen status={last_status!r}, ip={last_ip!r}."
    )
