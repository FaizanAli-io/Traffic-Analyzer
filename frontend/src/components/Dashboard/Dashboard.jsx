import React, { useState } from "react";
import "./Dashboard.css";

const API_BASE = import.meta.env.VITE_BACKEND_SERVER || "http://localhost:5000";

export default function Dashboard() {
  // GPU
  const [isLaunching, setIsLaunching] = useState(false);
  const [launchMsg, setLaunchMsg] = useState("");
  const [launchErr, setLaunchErr] = useState("");
  const [ip, setIp] = useState("");

  // Terminate Instance
  const [isTerminating, setIsTerminating] = useState(false);
  const [terminateMsg, setTerminateMsg] = useState("");
  const [terminateErr, setTerminateErr] = useState("");

  // Upload
  const [file, setFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState(""); // backend fixed name to use later
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploadErr, setUploadErr] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  // Process
  const [isProcessing, setIsProcessing] = useState(false);
  const [processMsg, setProcessMsg] = useState("");
  const [processErr, setProcessErr] = useState("");

  // Download
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadErr, setDownloadErr] = useState("");

  // Direction Control
  const [directionMapping, setDirectionMapping] = useState({
    top: "North",
    right: "East", 
    bottom: "South",
    left: "West"
  });
  const [isRotating, setIsRotating] = useState(false);
  const [directionMsg, setDirectionMsg] = useState("");
  const [directionErr, setDirectionErr] = useState("");

  const fetchCurrentDirectionMapping = async () => {
    try {
      const resp = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "get_current" }),
      });
      const data = await resp.json().catch(() => ({}));
      if (resp.ok && data.status === "ok") {
        setDirectionMapping(data.mapping);
      }
    } catch (e) {
      console.error("Failed to fetch direction mapping:", e);
    }
  };

  // Add this with your other useEffect hooks (if any)
  React.useEffect(() => {
    fetchCurrentDirectionMapping();
  }, []);

  const rotateDirectionClockwise = async () => {
    setIsRotating(true);
    setDirectionErr("");
    setDirectionMsg("Rotating clockwise...");
    try {
      const resp = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "rotate_clockwise" }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.status !== "ok") {
        throw new Error(data.message || `Rotation failed (HTTP ${resp.status})`);
      }
      setDirectionMapping(data.mapping);
      setDirectionMsg("Rotated clockwise ↻");
    } catch (e) {
      setDirectionErr(e.message || "Rotation failed.");
      setDirectionMsg("");
    } finally {
      setIsRotating(false);
    }
  };

  const rotateDirectionCounterclockwise = async () => {
    setIsRotating(true);
    setDirectionErr("");
    setDirectionMsg("Rotating counter-clockwise...");
    try {
      const resp = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "rotate_counterclockwise" }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.status !== "ok") {
        throw new Error(data.message || `Rotation failed (HTTP ${resp.status})`);
      }
      setDirectionMapping(data.mapping);
      setDirectionMsg("Rotated counter-clockwise ↺");
    } catch (e) {
      setDirectionErr(e.message || "Rotation failed.");
      setDirectionMsg("");
    } finally {
      setIsRotating(false);
    }
  };
  const startGpu = async () => {
    setIsLaunching(true);
    setLaunchErr("");
    setLaunchMsg("Starting and setting up GPU… please wait. Don't refresh or close this tab.");
    try {
      const resp = await fetch(`${API_BASE}/lambda/launch-and-setup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.status !== "ready") {
        throw new Error(data.message || `Launch failed (HTTP ${resp.status})`);
      }
      setIp(data.ip || "");
      setLaunchMsg("GPU setup done ✅");
    } catch (e) {
      setLaunchErr(e.message || "GPU launch failed.");
      setLaunchMsg("");
    } finally {
      setIsLaunching(false);
    }
  };

  const terminateInstance = async () => {
    setIsTerminating(true);
    setTerminateErr("");
    setTerminateMsg("Terminating GPU instance… please wait.");
    try {
      const resp = await fetch(`${API_BASE}/lambda/terminate-instance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.status !== "ok") {
        throw new Error(data.message || `Termination failed (HTTP ${resp.status})`);
      }
      setTerminateMsg("GPU instance terminated successfully ✅");
      setIp("");
      setLaunchMsg("");
      setProcessMsg("");
      setUploadMsg("");
    } catch (e) {
      setTerminateErr(e.message || "Instance termination failed.");
      setTerminateMsg("");
    } finally {
      setIsTerminating(false);
    }
  };

  const onChooseFile = (e) => {
    const f = e.target.files?.[0];
    setUploadErr("");
    setUploadMsg("");
    setFile(null);
    setUploadedFilename("");
    if (!f) return;
    const name = f.name.toLowerCase();
    const ok = name.endsWith(".mp4") || name.endsWith(".mov") || name.endsWith(".avi");
    if (!ok) {
      setUploadErr("Invalid file type. Only MP4, MOV, AVI allowed.");
      return;
    }
    setFile(f);
  };

  const uploadVideo = async () => {
    if (!file) return;
    setIsUploading(true);
    setUploadErr("");
    setUploadMsg("Uploading…");
    try {
      const fd = new FormData();
      fd.append("video", file);
      const resp = await fetch(`${API_BASE}/upload-video`, { method: "POST", body: fd });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.status !== "ok") {
        throw new Error(data.message || `Upload failed (HTTP ${resp.status})`);
      }
      setUploadedFilename(data.filename || "input_video_4.mp4");
      setUploadMsg(`Uploaded successfully as: ${data.filename || "input_video_4.mp4"} ✅`);
    } catch (e) {
      setUploadErr(e.message || "Upload failed.");
      setUploadMsg("");
    } finally {
      setIsUploading(false);
    }
  };

  const processVideo = async () => {
    setIsProcessing(true);
    setProcessErr("");
    setProcessMsg("Processing video on GPU… this can take several minutes. Please don't refresh or close this tab.");
    try {
      const body = {
        ip: ip || undefined,
        video: uploadedFilename || undefined,
      };
      const resp = await fetch(`${API_BASE}/lambda/upload-video-and-run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || (data.status && data.status !== "ok")) {
        throw new Error(data.message || `Processing failed (HTTP ${resp.status})`);
      }
      const out = data.output_path || data.output || data.result || "";
      setProcessMsg(`Processing complete ✅ ${out ? `Output: ${out}` : ""}`);
    } catch (e) {
      setProcessErr(e.message || "Processing failed.");
      setProcessMsg("");
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadFiles = async () => {
    setIsDownloading(true);
    setDownloadErr("");
    try {
      const params = new URLSearchParams();
      if (ip) params.set("ip", ip);

      const resp = await fetch(`${API_BASE}/lambda/download-output?${params.toString()}`, {
        method: "GET",
      });
      if (!resp.ok) {
        const maybeJson = await resp.json().catch(() => ({}));
        throw new Error(maybeJson.message || `Download failed (HTTP ${resp.status})`);
      }

      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "outputs_bundle.zip";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (e) {
      setDownloadErr(e.message || "Download failed.");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="dashboard-main-container">
      <div className="dashboard-header">
        <h1 className="dashboard-title">
          <span className="dashboard-icon">⚡</span>
          GPU Processing Center
        </h1>
        <p className="dashboard-subtitle">Manage your GPU instances and process videos with ease</p>
      </div>

      <div className="dashboard-grid">
        {/* GPU Control Section */}
        <div className="dashboard-card">
          <h2 className="dashboard-card-title">
            <span className="dashboard-card-icon">🖥️</span>
            GPU Instance Control
          </h2>
          
          <div className="dashboard-button-group">
            <button
              onClick={startGpu}
              disabled={isLaunching}
              className={`dashboard-btn dashboard-btn-primary ${isLaunching ? 'dashboard-loading' : ''}`}
            >
              {isLaunching && <div className="dashboard-spinner"></div>}
              {isLaunching ? "Launching…" : "Start GPU"}
            </button>
            
            <button
              onClick={terminateInstance}
              disabled={isTerminating}
              className={`dashboard-btn dashboard-btn-danger ${isTerminating ? 'dashboard-loading' : ''}`}
            >
              {isTerminating && <div className="dashboard-spinner"></div>}
              {isTerminating ? "Terminating…" : "Terminate Instance"}
            </button>
          </div>

          <div className="dashboard-status-area">
            {launchMsg && (
              <div className="dashboard-status dashboard-success">
                <span className="dashboard-status-icon">✅</span>
                {launchMsg}
              </div>
            )}
            {ip && (
              <div className="dashboard-ip-display">
                <span className="dashboard-ip-label">Instance IP:</span>
                <code className="dashboard-ip-code">{ip}</code>
              </div>
            )}
            {launchErr && (
              <div className="dashboard-status dashboard-error">
                <span className="dashboard-status-icon">❌</span>
                {launchErr}
              </div>
            )}
            {terminateMsg && (
              <div className="dashboard-status dashboard-success">
                <span className="dashboard-status-icon">✅</span>
                {terminateMsg}
              </div>
            )}
            {terminateErr && (
              <div className="dashboard-status dashboard-error">
                <span className="dashboard-status-icon">❌</span>
                {terminateErr}
              </div>
            )}
          </div>
        </div>

        {/* Upload Section */}
        <div className="dashboard-card">
          <h2 className="dashboard-card-title">
            <span className="dashboard-card-icon">📤</span>
            Video Upload
          </h2>
          
          <div className="dashboard-upload-area">
            <label className="dashboard-file-input-wrapper">
              <input
                type="file"
                accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
                onChange={onChooseFile}
                className="dashboard-file-input"
              />
              <div className="dashboard-file-input-display">
                <span className="dashboard-upload-icon">📁</span>
                <span>Click to select video file</span>
                <small>MP4, MOV, AVI supported</small>
              </div>
            </label>

            {file && (
              <div className="dashboard-file-preview">
                <div className="dashboard-file-info">
                  <span className="dashboard-file-name">{file.name}</span>
                  <span className="dashboard-file-size">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              </div>
            )}

            <button
              onClick={uploadVideo}
              disabled={!file || isUploading}
              className={`dashboard-btn dashboard-btn-secondary ${(!file || isUploading) ? 'dashboard-disabled' : ''} ${isUploading ? 'dashboard-loading' : ''}`}
            >
              {isUploading && <div className="dashboard-spinner"></div>}
              {isUploading ? "Uploading…" : "Upload Video"}
            </button>
          </div>

          <div className="dashboard-status-area">
            {uploadMsg && (
              <div className="dashboard-status dashboard-success">
                <span className="dashboard-status-icon">✅</span>
                {uploadMsg}
              </div>
            )}
            {uploadErr && (
              <div className="dashboard-status dashboard-error">
                <span className="dashboard-status-icon">❌</span>
                {uploadErr}
              </div>
            )}
          </div>
        </div>

        {/* Process Section */}
        <div className="dashboard-card">
          <h2 className="dashboard-card-title">
            <span className="dashboard-card-icon">⚙️</span>
            Video Processing
          </h2>
          
          <div className="dashboard-process-area">
            <p className="dashboard-process-description">
              Process your uploaded video using GPU acceleration. This operation may take several minutes.
            </p>
            
            <button
              onClick={processVideo}
              disabled={isProcessing || !uploadedFilename}
              title={!uploadedFilename ? "Upload a video first" : ""}
              className={`dashboard-btn dashboard-btn-accent ${(isProcessing || !uploadedFilename) ? 'dashboard-disabled' : ''} ${isProcessing ? 'dashboard-loading' : ''}`}
            >
              {isProcessing && <div className="dashboard-spinner"></div>}
              {isProcessing ? "Processing…" : "Process Video"}
            </button>
          </div>

          <div className="dashboard-status-area">
            {processMsg && (
              <div className="dashboard-status dashboard-success">
                <span className="dashboard-status-icon">✅</span>
                {processMsg}
              </div>
            )}
            {processErr && (
              <div className="dashboard-status dashboard-error">
                <span className="dashboard-status-icon">❌</span>
                {processErr}
              </div>
            )}
          </div>
        </div>

        {/* Direction Control Section */}
        <div className="dashboard-card">
          <h2 className="dashboard-card-title">
            <span className="dashboard-card-icon">🧭</span>
            Direction Mapping
          </h2>
          
          <div className="direction-control-area">
            <div className="direction-compass">
              <div className="compass-center">
                <div className="compass-direction compass-north">{directionMapping.top}</div>
                <div className="compass-direction compass-east">{directionMapping.right}</div>
                <div className="compass-direction compass-south">{directionMapping.bottom}</div>
                <div className="compass-direction compass-west">{directionMapping.left}</div>
              </div>
            </div>
            
            <div className="direction-button-group">
              <button
                onClick={rotateDirectionClockwise}
                disabled={isRotating}
                className={`dashboard-btn dashboard-btn-secondary ${isRotating ? 'dashboard-loading' : ''}`}
              >
                {isRotating && <div className="dashboard-spinner"></div>}
                Rotate Clockwise ↻
              </button>
              
              <button
                onClick={rotateDirectionCounterclockwise}
                disabled={isRotating}
                className={`dashboard-btn dashboard-btn-secondary ${isRotating ? 'dashboard-loading' : ''}`}
              >
                {isRotating && <div className="dashboard-spinner"></div>}
                Rotate Counter ↺
              </button>
            </div>
            
            <div className="direction-info">
              <small>Adjust the compass to match your camera's orientation</small>
            </div>
          </div>

          <div className="dashboard-status-area">
            {directionMsg && (
              <div className="dashboard-status dashboard-success">
                <span className="dashboard-status-icon">✅</span>
                {directionMsg}
              </div>
            )}
            {directionErr && (
              <div className="dashboard-status dashboard-error">
                <span className="dashboard-status-icon">❌</span>
                {directionErr}
              </div>
            )}
          </div>
        </div>

        {/* Download Section */}
        <div className="dashboard-card">
          <h2 className="dashboard-card-title">
            <span className="dashboard-card-icon">📥</span>
            Download Results
          </h2>
          
          <div className="dashboard-download-area">
            <p className="dashboard-download-description">
              Download a ZIP file containing both the processed video and CSV data with analysis results.
            </p>
            
            <button
              onClick={downloadFiles}
              disabled={isDownloading}
              className={`dashboard-btn dashboard-btn-success ${isDownloading ? 'dashboard-loading' : ''}`}
            >
              {isDownloading && <div className="dashboard-spinner"></div>}
              {isDownloading ? "Downloading…" : "Download ZIP (Video + CSV)"}
            </button>
          </div>

          <div className="dashboard-status-area">
            {downloadErr && (
              <div className="dashboard-status dashboard-error">
                <span className="dashboard-status-icon">❌</span>
                {downloadErr}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}