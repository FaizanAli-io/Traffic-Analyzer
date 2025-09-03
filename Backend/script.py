import requests
import json
import time
import os
import paramiko
import scp
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import uuid
import threading
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
LAMBDA_API_KEY = "YOUR_LAMBDA_API_KEY"  # Replace with your actual API key
LAMBDA_BASE_URL = "https://cloud.lambdalabs.com/api/v1"
SSH_KEY_PATH = "path/to/your/private/key.pem"  # Path to your SSH private key

UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './results'
SCRIPTS_FOLDER = './scripts'

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, SCRIPTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB

# Global variables to track processing jobs
processing_jobs = {}

class LambdaCloudManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def launch_instance(self, instance_type="gpu_1x_a10", region="us-west-1", ssh_key_name=None):
        """Launch a new instance"""
        data = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": [ssh_key_name] if ssh_key_name else [],
            "quantity": 1
        }
        
        response = requests.post(f"{LAMBDA_BASE_URL}/instance-operations/launch",
                               headers=self.headers, json=data)
        return response.json()
    
    def list_instances(self):
        """List all instances"""
        response = requests.get(f"{LAMBDA_BASE_URL}/instances",
                              headers={'Authorization': f'Bearer {self.api_key}'})
        return response.json()
    
    def get_running_instance(self):
        """Get first available running instance"""
        instances = self.list_instances()
        for instance in instances.get('data', []):
            if instance['status'] == 'running':
                return instance
        return None
    
    def terminate_instance(self, instance_id):
        """Terminate an instance"""
        data = {"instance_ids": [instance_id]}
        response = requests.post(f"{LAMBDA_BASE_URL}/instance-operations/terminate",
                               headers=self.headers, json=data)
        return response.json()

class SSHManager:
    def __init__(self, hostname, username, key_path):
        self.hostname = hostname
        self.username = username
        self.key_path = key_path
        self.ssh_client = None
        self.scp_client = None
    
    def connect(self):
        """Establish SSH connection"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            private_key = paramiko.RSAKey.from_private_key_file(self.key_path)
            self.ssh_client.connect(
                hostname=self.hostname,
                username=self.username,
                pkey=private_key,
                timeout=30
            )
            
            self.scp_client = scp.SCPClient(self.ssh_client.get_transport())
            return True
        except Exception as e:
            print(f"SSH connection failed: {e}")
            return False
    
    def upload_file(self, local_path, remote_path):
        """Upload file to remote server"""
        try:
            self.scp_client.put(local_path, remote_path)
            return True
        except Exception as e:
            print(f"File upload failed: {e}")
            return False
    
    def download_file(self, remote_path, local_path):
        """Download file from remote server"""
        try:
            self.scp_client.get(remote_path, local_path)
            return True
        except Exception as e:
            print(f"File download failed: {e}")
            return False
    
    def execute_command(self, command):
        """Execute command on remote server"""
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            output = stdout.read().decode()
            error = stderr.read().decode()
            return_code = stdout.channel.recv_exit_status()
            
            return {
                'output': output,
                'error': error,
                'return_code': return_code
            }
        except Exception as e:
            return {
                'output': '',
                'error': str(e),
                'return_code': -1
            }
    
    def disconnect(self):
        """Close SSH connection"""
        if self.scp_client:
            self.scp_client.close()
        if self.ssh_client:
            self.ssh_client.close()

def process_video_on_lambda(job_id, video_path, script_path, instance_ip, processing_options):
    """Background task to process video on Lambda instance"""
    ssh_manager = SSHManager(instance_ip, 'ubuntu', SSH_KEY_PATH)
    
    try:
        processing_jobs[job_id]['status'] = 'connecting'
        
        # Connect to instance
        if not ssh_manager.connect():
            processing_jobs[job_id]['status'] = 'failed'
            processing_jobs[job_id]['error'] = 'Failed to connect to Lambda instance'
            return
        
        processing_jobs[job_id]['status'] = 'uploading'
        
        # Create directories on remote
        ssh_manager.execute_command('mkdir -p ~/video_processing/uploads ~/video_processing/results ~/video_processing/scripts')
        
        # Upload video and script
        remote_video_path = f"~/video_processing/uploads/{os.path.basename(video_path)}"
        remote_script_path = f"~/video_processing/scripts/{os.path.basename(script_path)}"
        
        if not ssh_manager.upload_file(video_path, remote_video_path):
            processing_jobs[job_id]['status'] = 'failed'
            processing_jobs[job_id]['error'] = 'Failed to upload video'
            return
        
        if not ssh_manager.upload_file(script_path, remote_script_path):
            processing_jobs[job_id]['status'] = 'failed'
            processing_jobs[job_id]['error'] = 'Failed to upload script'
            return
        
        processing_jobs[job_id]['status'] = 'processing'
        
        # Make script executable and run it
        ssh_manager.execute_command(f'chmod +x {remote_script_path}')
        
        # Run processing script
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"~/video_processing/results/{video_name}_processed.mp4"
        
        command = f'cd ~/video_processing && python3 {remote_script_path} --input {remote_video_path} --output {output_path}'
        
        if processing_options:
            command += f' --options \'{json.dumps(processing_options)}\''
        
        result = ssh_manager.execute_command(command)
        
        if result['return_code'] != 0:
            processing_jobs[job_id]['status'] = 'failed'
            processing_jobs[job_id]['error'] = f"Processing failed: {result['error']}"
            return
        
        processing_jobs[job_id]['status'] = 'downloading'
        
        # Download processed video
        local_output_path = os.path.join(RESULTS_FOLDER, f"{video_name}_processed.mp4")
        
        if ssh_manager.download_file(output_path, local_output_path):
            processing_jobs[job_id]['status'] = 'completed'
            processing_jobs[job_id]['output_file'] = local_output_path
            processing_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        else:
            processing_jobs[job_id]['status'] = 'failed'
            processing_jobs[job_id]['error'] = 'Failed to download processed video'
    
    except Exception as e:
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['error'] = str(e)
    
    finally:
        ssh_manager.disconnect()

# Initialize Lambda manager
lambda_manager = LambdaCloudManager(LAMBDA_API_KEY)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Lambda Video Processing API"})

@app.route('/lambda/instances', methods=['GET'])
def list_instances():
    """List all Lambda instances"""
    result = lambda_manager.list_instances()
    return jsonify(result)

@app.route('/lambda/launch', methods=['POST'])
def launch_instance():
    """Launch a new Lambda instance"""
    data = request.json or {}
    instance_type = data.get('instance_type', 'gpu_1x_a10')
    region = data.get('region', 'us-west-1')
    ssh_key_name = data.get('ssh_key_name')
    
    result = lambda_manager.launch_instance(instance_type, region, ssh_key_name)
    return jsonify(result)

@app.route('/upload/video', methods=['POST'])
def upload_video():
    """Upload video for processing"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({
        "message": "Video uploaded successfully",
        "filename": filename,
        "file_id": filename.split('.')[0],
        "file_path": filepath
    })

@app.route('/upload/script', methods=['POST'])
def upload_script():
    """Upload processing script"""
    if 'script' not in request.files:
        return jsonify({"error": "No script file provided"}), 400
    
    file = request.files['script']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(SCRIPTS_FOLDER, filename)
    file.save(filepath)
    
    return jsonify({
        "message": "Script uploaded successfully",
        "filename": filename,
        "script_id": filename.split('.')[0],
        "file_path": filepath
    })

@app.route('/process', methods=['POST'])
def start_processing():
    """Start video processing on Lambda Cloud"""
    data = request.json
    
    video_filename = data.get('video_filename')
    script_filename = data.get('script_filename')
    processing_options = data.get('options', {})
    
    if not video_filename or not script_filename:
        return jsonify({"error": "Both video_filename and script_filename required"}), 400
    
    # Check if files exist
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    script_path = os.path.join(SCRIPTS_FOLDER, script_filename)
    
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    if not os.path.exists(script_path):
        return jsonify({"error": "Script file not found"}), 404
    
    # Get running Lambda instance
    instance = lambda_manager.get_running_instance()
    if not instance:
        return jsonify({
            "error": "No running Lambda instance available",
            "suggestion": "Launch an instance first using /lambda/launch"
        }), 400
    
    # Create processing job
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        'status': 'queued',
        'video_filename': video_filename,
        'script_filename': script_filename,
        'instance_id': instance['id'],
        'instance_ip': instance['ip'],
        'created_at': datetime.now().isoformat(),
        'options': processing_options
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_video_on_lambda,
        args=(job_id, video_path, script_path, instance['ip'], processing_options)
    )
    thread.start()
    
    return jsonify({
        "message": "Processing started",
        "job_id": job_id,
        "status": "queued",
        "instance_ip": instance['ip']
    })

@app.route('/status/<job_id>', methods=['GET'])
def get_processing_status(job_id):
    """Get processing status"""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    return jsonify({
        "job_id": job_id,
        "status": job['status'],
        "video_filename": job['video_filename'],
        "script_filename": job['script_filename'],
        "created_at": job['created_at'],
        "error": job.get('error'),
        "completed_at": job.get('completed_at')
    })

@app.route('/download/<job_id>', methods=['GET'])
def download_processed_video(job_id):
    """Download processed video"""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({"error": f"Job status is '{job['status']}', not completed"}), 400
    
    if 'output_file' not in job or not os.path.exists(job['output_file']):
        return jsonify({"error": "Output file not found"}), 404
    
    return send_file(job['output_file'], as_attachment=True)

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all processing jobs"""
    return jsonify({
        "jobs": [
            {
                "job_id": job_id,
                "status": job['status'],
                "video_filename": job['video_filename'],
                "created_at": job['created_at']
            }
            for job_id, job in processing_jobs.items()
        ]
    })

if __name__ == '__main__':
    print("Starting Lambda Cloud Video Processing API...")
    print("Make sure to configure:")
    print("1. LAMBDA_API_KEY")
    print("2. SSH_KEY_PATH")
    print("3. Install required packages: pip install paramiko scp")
    app.run(host='0.0.0.0', port=5000, debug=True)