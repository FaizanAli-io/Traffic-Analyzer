import os
import zipfile
import tempfile
from flask_cors import CORS
from flask import Flask, request, send_file, make_response
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.datastructures import FileStorage
import json

from .config import (
    LAMBDA_API_KEY,
    SSH_KEY_NAMES,
    INSTANCE_NAME,
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

# Configure CORS first, before API initialization
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173", "http://13.60.19.75"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Mount API and Swagger under /api
api = Api(
    app,
    version="1.0",
    title="Lambda Instance Management API",
    description="API for managing cloud instances and video processing",
    prefix="/api",          # <— all routes now under /api/...
    doc="/api/docs"         # <— Swagger UI at /api/docs
)

# Define namespaces for better organization
auth_ns = api.namespace('auth', description='Authentication operations')
lambda_ns = api.namespace('lambda', description='Lambda cloud instance operations')
video_ns = api.namespace('video', description='Video upload and processing')
health_ns = api.namespace('health', description='Health check operations')

# Define models for request/response schemas
login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

launch_model = api.model('LaunchInstance', {
    'region_name': fields.String(required=True, description='AWS region name'),
    'instance_type_name': fields.String(required=True, description='Instance type')
})

video_run_model = api.model('VideoRun', {
    'ip': fields.String(required=True, description='Instance IP address'),
    'video': fields.String(required=True, description='Video filename')
})

replace_file_model = api.model('ReplaceFile', {
    'ip': fields.String(required=True, description='Instance IP address'),
    'local_script': fields.String(required=True, description='Local script path'),
    'remote_script': fields.String(required=True, description='Remote script path')
})

direction_model = api.model('DirectionOrientation', {
    'action': fields.String(required=True, description='Action: rotate_clockwise, rotate_counterclockwise, or get_current')
})

terminate_model = api.model('TerminateInstance', {
    'instance_id': fields.String(required=True, description='Instance ID to terminate')
})

# Response models
success_response = api.model('SuccessResponse', {
    'success': fields.Boolean(description='Operation success status'),
    'message': fields.String(description='Response message')
})

error_response = api.model('ErrorResponse', {
    'status': fields.String(description='Status'),
    'error': fields.String(description='Error message'),
    'message': fields.String(description='Error message')
})

health_response = api.model('HealthResponse', {
    'status': fields.String(description='Service status'),
    'service': fields.String(description='Service name'),
    'api_key_status': fields.String(description='API key status'),
    'api_key_preview': fields.String(description='API key preview')
})

instance_response = api.model('InstanceResponse', {
    'ip': fields.String(description='Instance IP'),
    'status': fields.String(description='Instance status'),
    'message': fields.String(description='Response message'),
    'instance_id': fields.String(description='Instance ID')
})

upload_parser = api.parser()
upload_parser.add_argument('video', location='files', type=FileStorage, required=True, help='Video file')


def safe_json_response(response_obj):
    """Safely convert a requests Response object to Flask JSON response"""
    try:
        # Try to parse as JSON first
        data = response_obj.json()
        return data, response_obj.status_code
    except (json.JSONDecodeError, ValueError):
        # If not valid JSON, return as plain text wrapped in JSON
        return {
            "status": "error" if not response_obj.ok else "ok",
            "message": response_obj.text,
            "status_code": response_obj.status_code
        }, response_obj.status_code


@auth_ns.route("/login")
class AuthLogin(Resource):
    @auth_ns.expect(login_model)
    @auth_ns.response(200, 'Success', success_response)
    @auth_ns.response(400, 'Bad Request', error_response)
    @auth_ns.response(401, 'Unauthorized', error_response)
    def post(self):
        """Authenticate user with username and password"""
        data = request.get_json(silent=True) or {}
        username = data.get("username", "")
        password = data.get("password", "")
        if not username or not password:
            return {"success": False, "error": "username and password required"}, 400
        if username == ENV_USERNAME and password == ENV_PASSWORD:
            return {"success": True, "message": "login ok"}, 200
        return {"success": False, "error": "invalid credentials"}, 401


@health_ns.route("/health")
class Health(Resource):
    @health_ns.response(200, 'Success', health_response)
    def get(self):
        """Check API health status"""
        api_key_status = "SET" if LAMBDA_API_KEY else "NOT SET"
        api_key_preview = f"{LAMBDA_API_KEY[:8]}..." if LAMBDA_API_KEY else "None"
        return {
            "status": "ok",
            "service": "lambda-min-api",
            "api_key_status": api_key_status,
            "api_key_preview": api_key_preview,
        }


@lambda_ns.route("/firewall-rulesets")
class FirewallRulesets(Resource):
    @lambda_ns.response(200, 'Success')
    def get(self):
        """List available firewall rulesets"""
        r = cloud_get("/firewall-rulesets")
        return safe_json_response(r)


@lambda_ns.route("/instance-types")
class InstanceTypes(Resource):
    @lambda_ns.response(200, 'Success')
    def get(self):
        """List available instance types"""
        r = cloud_get("/instance-types")
        return safe_json_response(r)


@lambda_ns.route("/launch-and-setup")
class LaunchAndSetup(Resource):
    @lambda_ns.expect(launch_model)
    @lambda_ns.response(200, 'Success', instance_response)
    @lambda_ns.response(400, 'Bad Request', error_response)
    @lambda_ns.response(500, 'Internal Server Error', error_response)
    @lambda_ns.response(504, 'Gateway Timeout', error_response)
    def post(self):
        """Launch a new cloud instance and set it up"""
        try:
            incoming = request.get_json(silent=True) or {}
        except Exception:
            incoming = {}

        region_name = (incoming.get("region_name") or "").strip()
        instance_type_name = (incoming.get("instance_type_name") or "").strip()

        # Validate required fields
        missing = []
        if not region_name:
            missing.append("region_name")
        if not instance_type_name:
            missing.append("instance_type_name")
        if missing:
            return {
                "status": "error",
                "message": "Missing required field(s): " + ", ".join(missing),
            }, 400

        body = {
            "name": INSTANCE_NAME,
            "region_name": region_name,
            "ssh_key_names": SSH_KEY_NAMES,
            # "firewall_rulesets": FIREWALL_RULESETS,
            "instance_type_name": instance_type_name,
        }
        body = {k: v for k, v in body.items() if v not in ("", [], {}, None)}
        r = cloud_post("/instance-operations/launch", body)
        if not r.ok:
            print(f"[DEBUG] Request body: {body}")
            print(f"[DEBUG] Launch failed: {r.status_code} {r.text}")
            return safe_json_response(r)
        try:
            payload = r.json()
        except Exception:
            return {"status": "error", "message": "Invalid JSON in launch response"}, 500
        instance_id = extract_instance_id(payload)
        if not instance_id:
            return {
                "status": "error", 
                "message": "No instance id in launch response"
            }, 500
        try:
            ip = wait_for_instance_ip(cloud_get, instance_id, timeout=600, interval=5)
            save_instance_info(instance_id, ip)
        except RuntimeError as e:
            return {"status": "error", "message": str(e)}, 504
        if not wait_for_ssh(ip, timeout=600):
            return {"status": "error", "message": "SSH not ready"}, 504
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
            return {"status": "error", "message": str(e)}, 500
        return {
            "ip": ip,
            "status": "ready",
            "message": "GPU setup done",
            "instance_id": instance_id,
        }


@lambda_ns.route("/upload-video-and-run")
class UploadVideoAndRun(Resource):
    @lambda_ns.expect(video_run_model)
    @lambda_ns.response(200, 'Success')
    @lambda_ns.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Upload video to instance and run processing"""
        try:
            body = request.get_json(silent=True) or {}
            ip = body.get("ip") or None
            video = body.get("video") or None
            direction = getattr(app, "direction_orientation", 0)
            result = svc_upload_video_and_run(
                ip=ip, video_filename=video, direction_orientation=direction
            )
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500


@lambda_ns.route("/replace-file")
class ReplaceFile(Resource):
    @lambda_ns.expect(replace_file_model)
    @lambda_ns.response(200, 'Success')
    @lambda_ns.response(404, 'Not Found', error_response)
    @lambda_ns.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Replace a file on the remote instance"""
        try:
            body = request.get_json(silent=True) or {}
            ip = body.get("ip")
            local_script = body.get("local_script")
            remote_script = body.get("remote_script")
            from .processing import replace_remote_file

            res = replace_remote_file(
                ip=ip, local_script=local_script, remote_script=remote_script
            )
            return res
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}, 404
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500


@lambda_ns.route("/download-output")
class DownloadOutput(Resource):
    @lambda_ns.param('ip', 'Instance IP address')
    @lambda_ns.param('type', 'File type to download: video, csv, or both', default='both')
    @lambda_ns.response(200, 'Success - File download')
    @lambda_ns.response(404, 'Not Found', error_response)
    @lambda_ns.response(500, 'Internal Server Error', error_response)
    def get(self):
        """Download processed output files from instance"""
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
                    return {
                        "status": "error",
                        "message": f"Video not found: {video_local}",
                    }, 404

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
                    return {
                        "status": "error",
                        "message": "No CSV file found in remote directory",
                    }, 404
                import os as _os
                csv_name = _os.path.basename(csv_remote)
                csv_local = pull_remote_file(ip=ip, remote_filename=csv_name)
                if not os.path.exists(csv_local):
                    return {
                        "status": "error", 
                        "message": f"CSV not found: {csv_local}"
                    }, 404

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
            return {"status": "error", "message": str(e)}, 404
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500


@video_ns.route("/upload-video")
class UploadVideo(Resource):
    @video_ns.expect(upload_parser)
    @video_ns.response(200, 'Success')
    @video_ns.response(400, 'Bad Request', error_response)
    @video_ns.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Upload a video file for processing"""
        try:
            if "video" not in request.files:
                return {"status": "error", "message": "No video file provided"}, 400
            file = request.files["video"]
            if file.filename == "":
                return {"status": "error", "message": "No file selected"}, 400
            allowed_extensions = {".mp4", ".mov", ".avi"}
            import os as _os
            orig_ext = _os.path.splitext(file.filename)[1].lower()
            if orig_ext not in allowed_extensions:
                return {
                    "status": "error",
                    "message": "Invalid file type. Only MP4, MOV, AVI allowed",
                }, 400
            os.makedirs(INPUT_VIDEO_DIR, exist_ok=True)
            fixed_path = os.path.join(INPUT_VIDEO_DIR, FIXED_VIDEO_BASENAME)
            file.save(fixed_path)
            return {
                "status": "ok",
                "message": "Video uploaded successfully",
                "filename": FIXED_VIDEO_BASENAME,
                "path": fixed_path,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500


@lambda_ns.route("/set-direction-orientation")
class SetDirectionOrientation(Resource):
    @lambda_ns.expect(direction_model)
    @lambda_ns.response(200, 'Success')
    @lambda_ns.response(400, 'Bad Request', error_response)
    @lambda_ns.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Set or rotate the direction orientation for video processing"""
        try:
            body = request.get_json(silent=True) or {}
            action = body.get("action")
            if not action:
                return {"status": "error", "message": "Action required"}, 400
            if not hasattr(app, "direction_orientation"):
                app.direction_orientation = 0
            base_directions = ["North", "East", "South", "West"]
            if action == "rotate_clockwise":
                app.direction_orientation = (app.direction_orientation + 1) % 4
            elif action == "rotate_counterclockwise":
                app.direction_orientation = (app.direction_orientation - 1) % 4
            elif action != "get_current":
                return {"status": "error", "message": "Invalid action"}, 400
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
            return {
                "status": "ok",
                "orientation": app.direction_orientation,
                "mapping": mapping,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500


def terminate_saved_instance():
    from .instance_info import load_instance_info

    saved_id, _ip = load_instance_info()
    if not saved_id:
        raise ValueError("No saved instance found. Check instance_info.txt file.")
    r = cloud_post("/instance-operations/terminate", {"instance_ids": [saved_id]})
    if not r.ok:
        raise RuntimeError(f"Failed to terminate instance {saved_id}: {r.text}")
    return r.json(), saved_id, _ip


@lambda_ns.route("/terminate-instance")
class TerminateInstance(Resource):
    @lambda_ns.response(200, 'Success')
    @lambda_ns.response(404, 'Not Found', error_response)
    @lambda_ns.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Terminate the saved instance"""
        try:
            response_data, instance_id, ip = terminate_saved_instance()
            try:
                if os.path.exists(INSTANCE_INFO_FILE):
                    os.remove(INSTANCE_INFO_FILE)
            except Exception as e:
                print(f"Warning: Could not remove instance info file: {e}")
            return {
                "status": "ok",
                "message": "Instance terminated successfully",
                "instance_id": instance_id,
                "ip": ip,
                "lambda_response": response_data,
            }
        except ValueError as e:
            return {"status": "error", "message": str(e)}, 404
        except RuntimeError as e:
            return {"status": "error", "message": str(e)}, 500
        except Exception as e:
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}, 500


@lambda_ns.route("/instances")
class ListInstances(Resource):
    @lambda_ns.param('status', 'Filter by status (default: active)', default='active')
    @lambda_ns.response(200, 'Success')
    @lambda_ns.response(500, 'Internal Server Error', error_response)
    def get(self):
        """List cloud instances with optional status filter"""
        try:
            want = (request.args.get("status") or "active").lower()
            r = cloud_get("/instances")
            if not r.ok:
                return safe_json_response(r)
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
            return {"status": "ok", "count": len(concise), "instances": concise}
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500


@lambda_ns.route("/terminate-by-id")
class TerminateById(Resource):
    @lambda_ns.expect(terminate_model)
    @lambda_ns.response(200, 'Success')
    @lambda_ns.response(400, 'Bad Request', error_response)
    @lambda_ns.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Terminate a specific instance by ID"""
        try:
            body = request.get_json(silent=True) or {}
            instance_id = (body.get("instance_id") or "").strip()
            if not instance_id:
                return {"status": "error", "message": "instance_id required"}, 400
            r = cloud_post(
                "/instance-operations/terminate", {"instance_ids": [instance_id]}
            )
            if not r.ok:
                return safe_json_response(r)
            saved_id, _ = load_instance_info()
            if saved_id and saved_id == instance_id:
                try:
                    if os.path.exists(INSTANCE_INFO_FILE):
                        os.remove(INSTANCE_INFO_FILE)
                except Exception:
                    pass
            return {
                "status": "ok",
                "message": f"Instance {instance_id} terminated",
                "lambda_response": r.json(),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500


# Add OPTIONS handler for CORS preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response


if __name__ == "__main__":
    print(f"LAMBDA_API_KEY status: {'SET' if LAMBDA_API_KEY else 'NOT SET'}")
    print("Swagger UI available at: http://localhost:5000/docs/")
    app.run(host="0.0.0.0", port=5000, debug=True)