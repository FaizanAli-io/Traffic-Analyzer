import os
import zipfile
import tempfile
from flask_cors import CORS
from flask import Flask, jsonify, request, send_file
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.datastructures import FileStorage

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
CORS(app)

# Initialize Flask-RESTX
api = Api(
    app,
    version='1.0',
    title='Lambda Video Processing API',
    description='A comprehensive API for managing GPU instances and video processing on Lambda Labs',
    doc='/swagger/',  # Swagger UI will be available at /swagger/
    prefix='/api/v1'  # Optional: add API versioning
)

# Define namespaces for better organization
auth_ns = Namespace('auth', description='Authentication operations')
health_ns = Namespace('health', description='Health check operations')
lambda_ns = Namespace('lambda', description='Lambda Labs cloud operations')
video_ns = Namespace('video', description='Video processing operations')

api.add_namespace(auth_ns)
api.add_namespace(health_ns)
api.add_namespace(lambda_ns)
api.add_namespace(video_ns)

# Define models for request/response validation
login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username', example='admin'),
    'password': fields.String(required=True, description='Password', example='password123')
})

login_response = api.model('LoginResponse', {
    'success': fields.Boolean(description='Login success status'),
    'message': fields.String(description='Response message'),
    'error': fields.String(description='Error message if login failed')
})

health_response = api.model('HealthResponse', {
    'status': fields.String(description='Service status'),
    'service': fields.String(description='Service name'),
    'api_key_status': fields.String(description='API key status'),
    'api_key_preview': fields.String(description='API key preview')
})

launch_request = api.model('LaunchRequest', {
    'region_name': fields.String(required=True, description='AWS region name', example='us-east-1'),
    'instance_type_name': fields.String(required=True, description='Instance type', example='gpu_1x_a10')
})

launch_response = api.model('LaunchResponse', {
    'ip': fields.String(description='Instance IP address'),
    'status': fields.String(description='Instance status'),
    'message': fields.String(description='Response message'),
    'instance_id': fields.String(description='Instance ID')
})

video_upload_request = api.model('VideoUploadRequest', {
    'ip': fields.String(required=False, description='Instance IP address'),
    'video': fields.String(required=True, description='Video filename')
})

direction_request = api.model('DirectionRequest', {
    'action': fields.String(required=True, description='Action to perform', 
                          enum=['rotate_clockwise', 'rotate_counterclockwise', 'get_current'],
                          example='rotate_clockwise')
})

direction_response = api.model('DirectionResponse', {
    'status': fields.String(description='Response status'),
    'orientation': fields.Integer(description='Current orientation index'),
    'mapping': fields.Raw(description='Direction mapping object')
})

terminate_request = api.model('TerminateRequest', {
    'instance_id': fields.String(required=True, description='Instance ID to terminate')
})

error_response = api.model('ErrorResponse', {
    'status': fields.String(description='Error status', example='error'),
    'message': fields.String(description='Error message')
})

# File upload parser
upload_parser = api.parser()
upload_parser.add_argument('video', location='files', type=FileStorage, required=True, 
                         help='Video file (MP4, MOV, AVI)')

@auth_ns.route('/login')
class AuthLogin(Resource):
    @api.expect(login_model, validate=True)
    @api.marshal_with(login_response)
    @api.response(200, 'Success')
    @api.response(400, 'Bad Request - Missing username or password')
    @api.response(401, 'Unauthorized - Invalid credentials')
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

@health_ns.route('/health')
class Health(Resource):
    @api.marshal_with(health_response)
    @api.response(200, 'Success')
    def get(self):
        """Check service health and API key status"""
        api_key_status = "SET" if LAMBDA_API_KEY else "NOT SET"
        api_key_preview = f"{LAMBDA_API_KEY[:8]}..." if LAMBDA_API_KEY else "None"
        
        return {
            "status": "ok",
            "service": "lambda-min-api",
            "api_key_status": api_key_status,
            "api_key_preview": api_key_preview,
        }

@lambda_ns.route('/firewall-rulesets')
class FirewallRulesets(Resource):
    @api.response(200, 'Success')
    def get(self):
        """List available firewall rulesets from Lambda Labs"""
        r = cloud_get("/firewall-rulesets")
        return r.json() if r.ok else api.abort(r.status_code, r.text)

@lambda_ns.route('/instance-types')
class InstanceTypes(Resource):
    @api.response(200, 'Success')
    def get(self):
        """List available instance types from Lambda Labs"""
        r = cloud_get("/instance-types")
        return r.json() if r.ok else api.abort(r.status_code, r.text)

@lambda_ns.route('/launch-and-setup')
class LaunchAndSetup(Resource):
    @api.expect(launch_request, validate=True)
    @api.marshal_with(launch_response)
    @api.response(200, 'Success - Instance launched and configured')
    @api.response(400, 'Bad Request - Missing required fields')
    @api.response(500, 'Internal Server Error')
    @api.response(504, 'Gateway Timeout - Instance setup failed')
    def post(self):
        """Launch a new Lambda Labs instance and set it up for video processing"""
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
            api.abort(400, f"Missing required field(s): {', '.join(missing)}")

        body = {
            "name": INSTANCE_NAME,
            "region_name": region_name,
            "ssh_key_names": SSH_KEY_NAMES,
            "instance_type_name": instance_type_name,
        }
        body = {k: v for k, v in body.items() if v not in ("", [], {}, None)}
        
        r = cloud_post("/instance-operations/launch", body)
        if not r.ok:
            print(f"[DEBUG] Request body: {body}")
            print(f"[DEBUG] Launch failed: {r.status_code} {r.text}")
            api.abort(r.status_code, r.text)
        
        try:
            payload = r.json()
        except Exception:
            api.abort(500, "Invalid JSON in launch response")
        
        instance_id = extract_instance_id(payload)
        if not instance_id:
            api.abort(500, "No instance id in launch response")
        
        try:
            ip = wait_for_instance_ip(cloud_get, instance_id, timeout=600, interval=5)
            save_instance_info(instance_id, ip)
        except RuntimeError as e:
            api.abort(504, str(e))
        
        if not wait_for_ssh(ip, timeout=600):
            api.abort(504, "SSH not ready")
        
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
            api.abort(500, str(e))
        
        return {
            "ip": ip,
            "status": "ready",
            "message": "GPU setup done",
            "instance_id": instance_id,
        }

@lambda_ns.route('/upload-video-and-run')
class UploadVideoAndRun(Resource):
    @api.expect(video_upload_request, validate=True)
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error')
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
            api.abort(500, str(e))

@lambda_ns.route('/instances')
class Instances(Resource):
    @api.param('status', 'Filter by instance status', enum=['active', 'terminated', 'all'], default='active')
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error')
    def get(self):
        """List Lambda Labs instances with optional status filter"""
        try:
            want = (request.args.get("status") or "active").lower()
            r = cloud_get("/instances")
            if not r.ok:
                api.abort(r.status_code, r.text)
            
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
            api.abort(500, str(e))

@lambda_ns.route('/terminate-instance')
class TerminateInstance(Resource):
    @api.response(200, 'Success - Instance terminated')
    @api.response(404, 'Not Found - No saved instance')
    @api.response(500, 'Internal Server Error')
    def post(self):
        """Terminate the currently saved instance"""
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
            api.abort(404, str(e))
        except RuntimeError as e:
            api.abort(500, str(e))
        except Exception as e:
            api.abort(500, f"Unexpected error: {str(e)}")

@lambda_ns.route('/terminate-by-id')
class TerminateById(Resource):
    @api.expect(terminate_request, validate=True)
    @api.response(200, 'Success - Instance terminated')
    @api.response(400, 'Bad Request - Missing instance_id')
    @api.response(500, 'Internal Server Error')
    def post(self):
        """Terminate a specific instance by ID"""
        try:
            body = request.get_json(silent=True) or {}
            instance_id = (body.get("instance_id") or "").strip()
            if not instance_id:
                api.abort(400, "instance_id required")
            
            r = cloud_post(
                "/instance-operations/terminate", {"instance_ids": [instance_id]}
            )
            if not r.ok:
                api.abort(r.status_code, r.text)
            
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
            api.abort(500, str(e))

@lambda_ns.route('/download-output')
class DownloadOutput(Resource):
    @api.param('ip', 'Instance IP address')
    @api.param('type', 'File type to download', enum=['video', 'csv', 'both'], default='both')
    @api.response(200, 'Success - File downloaded')
    @api.response(404, 'Not Found - File not found')
    @api.response(500, 'Internal Server Error')
    def get(self):
        """Download processed video and/or CSV output from instance"""
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
                    api.abort(404, f"Video not found: {video_local}")

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
                    api.abort(404, "No CSV file found in remote directory")
                
                import os as _os
                csv_name = _os.path.basename(csv_remote)
                csv_local = pull_remote_file(ip=ip, remote_filename=csv_name)
                if not os.path.exists(csv_local):
                    api.abort(404, f"CSV not found: {csv_local}")

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
            api.abort(404, str(e))
        except Exception as e:
            api.abort(500, str(e))

@lambda_ns.route('/set-direction-orientation')
class SetDirectionOrientation(Resource):
    @api.expect(direction_request, validate=True)
    @api.marshal_with(direction_response)
    @api.response(200, 'Success')
    @api.response(400, 'Bad Request - Invalid action')
    @api.response(500, 'Internal Server Error')
    def post(self):
        """Set or rotate the direction orientation for video processing"""
        try:
            body = request.get_json(silent=True) or {}
            action = body.get("action")
            if not action:
                api.abort(400, "Action required")
            
            if not hasattr(app, "direction_orientation"):
                app.direction_orientation = 0
            
            base_directions = ["North", "East", "South", "West"]
            
            if action == "rotate_clockwise":
                app.direction_orientation = (app.direction_orientation + 1) % 4
            elif action == "rotate_counterclockwise":
                app.direction_orientation = (app.direction_orientation - 1) % 4
            elif action != "get_current":
                api.abort(400, "Invalid action")
            
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
            api.abort(500, str(e))

@video_ns.route('/upload')
class UploadVideo(Resource):
    @api.expect(upload_parser)
    @api.response(200, 'Success - Video uploaded')
    @api.response(400, 'Bad Request - Invalid file')
    @api.response(500, 'Internal Server Error')
    def post(self):
        """Upload a video file for processing"""
        try:
            if "video" not in request.files:
                api.abort(400, "No video file provided")
            
            file = request.files["video"]
            if file.filename == "":
                api.abort(400, "No file selected")
            
            allowed_extensions = {".mp4", ".mov", ".avi"}
            import os as _os
            orig_ext = _os.path.splitext(file.filename)[1].lower()
            if orig_ext not in allowed_extensions:
                api.abort(400, "Invalid file type. Only MP4, MOV, AVI allowed")
            
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
            api.abort(500, str(e))

# Helper function (keep the original function)
def terminate_saved_instance():
    from .instance_info import load_instance_info

    saved_id, _ip = load_instance_info()
    if not saved_id:
        raise ValueError("No saved instance found. Check instance_info.txt file.")
    r = cloud_post("/instance-operations/terminate", {"instance_ids": [saved_id]})
    if not r.ok:
        raise RuntimeError(f"Failed to terminate instance {saved_id}: {r.text}")
    return r.json(), saved_id, _ip

if __name__ == "__main__":
    print(f"LAMBDA_API_KEY status: {'SET' if LAMBDA_API_KEY else 'NOT SET'}")
    print("Swagger UI available at: http://localhost:5000/swagger/")
    app.run(host="0.0.0.0", port=5000, debug=True)