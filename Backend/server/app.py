import os
import tempfile
import zipfile
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from .config import (
    LAMBDA_API_KEY,
    REGION_NAME,
    SSH_KEY_NAMES,
    INSTANCE_NAME,
    INSTANCE_TYPE_NAME,
    FIREWALL_RULESETS,
    DOWNLOADS_DIR,
    INPUT_VIDEO_DIR,
    OUTPUT_DEFAULT_NAME,
    INSTANCE_INFO_FILE,
    ENV_USERNAME,
    ENV_PASSWORD,
    REMOTE_DIR,
    RUN_AFTER_SETUP,
    FIXED_VIDEO_BASENAME,
)
from .lambda_client import (
    cloud_get,
    cloud_post,
    extract_instance_id,
    wait_for_instance_ip,
)
from .ssh_utils import make_ssh, sh, wait_for_ssh
from .processing import (
    bootstrap_system_and_venv,
    upload_code_and_requirements_from_cwd,
    pip_install_requirements,
    run_script,
    upload_video_and_run as svc_upload_video_and_run,
    pull_remote_file,
)
from .instance_info import save_instance_info, load_instance_info, get_saved_ip


app = Flask(__name__)
CORS(app)


@app.post("/login")
def auth_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")
    if not username or not password:
        return (
            jsonify({"success": False, "error": "username and password required"}),
            400,
        )
    if username == ENV_USERNAME and password == ENV_PASSWORD:
        return jsonify({"success": True, "message": "login ok"}), 200
    return jsonify({"success": False, "error": "invalid credentials"}), 401


@app.get("/health")
def health():
    api_key_status = "SET" if LAMBDA_API_KEY else "NOT SET"
    api_key_preview = f"{LAMBDA_API_KEY[:8]}..." if LAMBDA_API_KEY else "None"
    return jsonify(
        {
            "status": "ok",
            "service": "lambda-min-api",
            "api_key_status": api_key_status,
            "api_key_preview": api_key_preview,
        }
    )


@app.get("/lambda/firewall-rulesets")
def list_firewall_rulesets():
    r = cloud_get("/firewall-rulesets")
    return (r.text, r.status_code, {"Content-Type": "application/json"})


@app.get("/lambda/instance-types")
def list_instance_types():
    r = cloud_get("/instance-types")
    return (r.text, r.status_code, {"Content-Type": "application/json"})


@app.post("/lambda/launch-and-setup")
def launch_and_setup():
    body = {
        "name": INSTANCE_NAME,
        "region_name": REGION_NAME,
        "ssh_key_names": SSH_KEY_NAMES,
        "firewall_rulesets": FIREWALL_RULESETS,
        "instance_type_name": INSTANCE_TYPE_NAME,
    }
    body = {k: v for k, v in body.items() if v not in ("", [], {}, None)}
    r = cloud_post("/instance-operations/launch", body)
    if not r.ok:
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    try:
        payload = r.json()
    except Exception:
        return (
            jsonify({"status": "error", "message": "Invalid JSON in launch response"}),
            500,
        )
    instance_id = extract_instance_id(payload)
    if not instance_id:
        return (
            jsonify(
                {"status": "error", "message": "No instance id in launch response"}
            ),
            500,
        )
    try:
        ip = wait_for_instance_ip(cloud_get, instance_id, timeout=600, interval=5)
        save_instance_info(instance_id, ip)
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)}), 504
    if not wait_for_ssh(ip, timeout=600):
        return jsonify({"status": "error", "message": "SSH not ready"}), 504
    try:
        ssh, scp = make_ssh(ip)
        bootstrap_system_and_venv(ssh)
        upload_code_and_requirements_from_cwd(ssh, scp)
        pip_install_requirements(ssh)
        if RUN_AFTER_SETUP:
            run_script(ssh)
        scp.close()
        ssh.close()
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify(
        {
            "ip": ip,
            "status": "ready",
            "message": "GPU setup done",
            "instance_id": instance_id,
        }
    )


@app.post("/lambda/upload-video-and-run")
def api_upload_video_and_run():
    try:
        body = request.get_json(silent=True) or {}
        ip = body.get("ip") or None
        video = body.get("video") or None
        direction = getattr(app, "direction_orientation", 0)
        result = svc_upload_video_and_run(
            ip=ip, video_filename=video, direction_orientation=direction
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/lambda/replace-file")
def api_replace_file():
    try:
        body = request.get_json(silent=True) or {}
        ip = body.get("ip")
        local_script = body.get("local_script")
        remote_script = body.get("remote_script")
        from .processing import replace_remote_file

        res = replace_remote_file(
            ip=ip, local_script=local_script, remote_script=remote_script
        )
        return jsonify(res)
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/lambda/download-output")
def api_download_output():
    try:
        ip = request.args.get("ip") or None
        file_type = (request.args.get("type") or "both").lower()

        if os.path.exists(DOWNLOADS_DIR):
            import shutil

            shutil.rmtree(DOWNLOADS_DIR)
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)

        video_local = None
        csv_local = None

        if file_type in ("video", "both"):
            video_name = OUTPUT_DEFAULT_NAME
            video_local = pull_remote_file(ip=ip, remote_filename=video_name)
            if not os.path.exists(video_local):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Video not found: {video_local}",
                        }
                    ),
                    404,
                )

        if file_type in ("csv", "both"):
            ssh, scp = make_ssh(ip or get_saved_ip())
            try:
                code, out, err = sh(
                    ssh, f"ls -1 {REMOTE_DIR}/*.csv 2>/dev/null | head -n 1"
                )
                csv_remote = out.strip()
            finally:
                try:
                    scp.close()
                except Exception:
                    pass
                try:
                    ssh.close()
                except Exception:
                    pass
            if not csv_remote:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "No CSV file found in remote directory",
                        }
                    ),
                    404,
                )
            import os as _os

            csv_name = _os.path.basename(csv_remote)
            csv_local = pull_remote_file(ip=ip, remote_filename=csv_name)
            if not os.path.exists(csv_local):
                return (
                    jsonify(
                        {"status": "error", "message": f"CSV not found: {csv_local}"}
                    ),
                    404,
                )

        if file_type == "video":
            return send_file(
                video_local,
                as_attachment=True,
                download_name=os.path.basename(video_local),
            )
        elif file_type == "csv":
            return send_file(
                csv_local, as_attachment=True, download_name=os.path.basename(csv_local)
            )
        else:
            tmp_dir = tempfile.mkdtemp(prefix="outputs_")
            zip_path = os.path.join(tmp_dir, "outputs_bundle.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if video_local:
                    zf.write(video_local, arcname=os.path.basename(video_local))
                if csv_local:
                    zf.write(csv_local, arcname=os.path.basename(csv_local))
            return send_file(
                zip_path, as_attachment=True, download_name="outputs_bundle.zip"
            )
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/upload-video")
def upload_video():
    try:
        if "video" not in request.files:
            return (
                jsonify({"status": "error", "message": "No video file provided"}),
                400,
            )
        file = request.files["video"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400
        allowed_extensions = {".mp4", ".mov", ".avi"}
        import os as _os

        orig_ext = _os.path.splitext(file.filename)[1].lower()
        if orig_ext not in allowed_extensions:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Invalid file type. Only MP4, MOV, AVI allowed",
                    }
                ),
                400,
            )
        os.makedirs(INPUT_VIDEO_DIR, exist_ok=True)
        fixed_path = os.path.join(INPUT_VIDEO_DIR, FIXED_VIDEO_BASENAME)
        file.save(fixed_path)
        return jsonify(
            {
                "status": "ok",
                "message": "Video uploaded successfully",
                "filename": FIXED_VIDEO_BASENAME,
                "path": fixed_path,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/lambda/set-direction-orientation")
def api_set_direction_orientation():
    try:
        body = request.get_json(silent=True) or {}
        action = body.get("action")
        if not action:
            return jsonify({"status": "error", "message": "Action required"}), 400
        if not hasattr(app, "direction_orientation"):
            app.direction_orientation = 0
        base_directions = ["North", "East", "South", "West"]
        if action == "rotate_clockwise":
            app.direction_orientation = (app.direction_orientation + 1) % 4
        elif action == "rotate_counterclockwise":
            app.direction_orientation = (app.direction_orientation - 1) % 4
        elif action != "get_current":
            return jsonify({"status": "error", "message": "Invalid action"}), 400
        current_directions = [
            base_directions[app.direction_orientation],
            base_directions[(app.direction_orientation + 1) % 4],
            base_directions[(app.direction_orientation + 2) % 4],
            base_directions[(app.direction_orientation + 3) % 4],
        ]
        mapping = {
            "top": current_directions[0],
            "right": current_directions[1],
            "bottom": current_directions[2],
            "left": current_directions[3],
        }
        return jsonify(
            {
                "status": "ok",
                "orientation": app.direction_orientation,
                "mapping": mapping,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def terminate_saved_instance():
    from .instance_info import load_instance_info

    saved_id, _ip = load_instance_info()
    if not saved_id:
        raise ValueError("No saved instance found. Check instance_info.txt file.")
    r = cloud_post("/instance-operations/terminate", {"instance_ids": [saved_id]})
    if not r.ok:
        raise RuntimeError(f"Failed to terminate instance {saved_id}: {r.text}")
    return r.json(), saved_id, _ip


@app.post("/lambda/terminate-instance")
def api_terminate_instance():
    try:
        response_data, instance_id, ip = terminate_saved_instance()
        try:
            if os.path.exists(INSTANCE_INFO_FILE):
                os.remove(INSTANCE_INFO_FILE)
        except Exception as e:
            print(f"Warning: Could not remove instance info file: {e}")
        return jsonify(
            {
                "status": "ok",
                "message": "Instance terminated successfully",
                "instance_id": instance_id,
                "ip": ip,
                "lambda_response": response_data,
            }
        )
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}),
            500,
        )


@app.get("/lambda/instances")
def api_list_instances():
    try:
        want = (request.args.get("status") or "active").lower()
        r = cloud_get("/instances")
        if not r.ok:
            return (r.text, r.status_code, {"Content-Type": "application/json"})
        payload = r.json()
        items = payload.get("data") or payload.get("instances") or []
        if want != "all":
            items = [i for i in items if (i.get("status", "").lower() == want)]
        concise = [
            {
                "id": i.get("id"),
                "name": i.get("name"),
                "status": i.get("status"),
                "ip": i.get("ip") or i.get("ipv4"),
                "region_name": i.get("region_name"),
                "instance_type_name": i.get("instance_type_name"),
                "created_at": i.get("created_at"),
            }
            for i in items
        ]
        return jsonify({"status": "ok", "count": len(concise), "instances": concise})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/lambda/terminate-by-id")
def api_terminate_by_id():
    try:
        body = request.get_json(silent=True) or {}
        instance_id = (body.get("instance_id") or "").strip()
        if not instance_id:
            return jsonify({"status": "error", "message": "instance_id required"}), 400
        r = cloud_post(
            "/instance-operations/terminate", {"instance_ids": [instance_id]}
        )
        if not r.ok:
            return (r.text, r.status_code, {"Content-Type": "application/json"})
        saved_id, _ = load_instance_info()
        if saved_id and saved_id == instance_id:
            try:
                if os.path.exists(INSTANCE_INFO_FILE):
                    os.remove(INSTANCE_INFO_FILE)
            except Exception:
                pass
        return jsonify(
            {
                "status": "ok",
                "message": f"Instance {instance_id} terminated",
                "lambda_response": r.json(),
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print(f"LAMBDA_API_KEY status: {'SET' if LAMBDA_API_KEY else 'NOT SET'}")
    app.run(host="0.0.0.0", port=5000, debug=True)
