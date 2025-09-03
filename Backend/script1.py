# file: lambda_min_api.py
import os
import requests
from flask import Flask, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Config ---
LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
LAMBDA_BASE_URL = "https://cloud.lambdalabs.com/api/v1"

app = Flask(__name__)

def cloud_get(path: str):
    if not LAMBDA_API_KEY:
        return type('MockResponse', (), {
            'ok': False, 
            'status_code': 401, 
            'text': '{"error": "No API key provided"}',
            'json': lambda: {"error": "No API key provided"}
        })()
    
    headers = {"Authorization": f"Bearer {LAMBDA_API_KEY}"}
    resp = requests.get(f"{LAMBDA_BASE_URL}{path}", headers=headers, timeout=30)
    return resp

@app.get("/health")
def health():
    api_key_status = "SET" if LAMBDA_API_KEY else "NOT SET"
    api_key_preview = f"{LAMBDA_API_KEY[:8]}..." if LAMBDA_API_KEY else "None"
    
    return jsonify({
        "status": "ok", 
        "service": "lambda-min-api",
        "api_key_status": api_key_status,
        "api_key_preview": api_key_preview
    })

@app.get("/lambda/instances")
def list_instances():
    """Proxy: return Lambda Labs /instances response as-is."""
    r = cloud_get("/instances")
    return (r.text, r.status_code, {"Content-Type": "application/json"})

@app.get("/lambda/running")
def running_instances():
    """Return only instances whose status == 'running'."""
    r = cloud_get("/instances")
    if not r.ok:
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    data = r.json()
    running = [inst for inst in data.get("data", []) if inst.get("status") == "running"]
    return jsonify({"count": len(running), "instances": running})

if __name__ == "__main__":
    print(f"LAMBDA_API_KEY status: {'SET' if LAMBDA_API_KEY else 'NOT SET'}")
    if LAMBDA_API_KEY:
        print(f"API Key preview: {LAMBDA_API_KEY[:8]}...")
    else:
        print("⚠️  API Key is not set! Set it with: $env:LAMBDA_API_KEY = \"sk-your-key\"")
    
    app.run(host="0.0.0.0", port=5000, debug=True)