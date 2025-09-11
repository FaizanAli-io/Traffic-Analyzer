import "./Dashboard.css";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const API_BASE = import.meta.env.VITE_BACKEND_SERVER || "http://localhost:5000";

export default function Dashboard({ onLogout }) {
  // Accept onLogout prop
  const navigate = useNavigate();

  // Get user info from localStorage
  const getUserInfo = () => {
    try {
      const userSession = localStorage.getItem("userSession");
      if (userSession) {
        return JSON.parse(userSession);
      }
    } catch (error) {
      console.error("Error parsing user session:", error);
    }
    return null;
  };

  const userInfo = getUserInfo();

  // Updated logout function
  const handleLogout = () => {
    localStorage.removeItem("userSession");
    // Call the parent's logout handler to update App state
    if (onLogout) {
      onLogout();
    }
    // Navigate to login - this will now work properly since App state is updated
    navigate("/login");
  };

  // ---------- Existing state variables ----------
  const [instances, setInstances] = useState([]);
  const [isLoadingInstances, setIsLoadingInstances] = useState(false);
  const [instancesErr, setInstancesErr] = useState("");
  const [terminatingIds, setTerminatingIds] = useState(new Set());

  // GPU
  const [isLaunching, setIsLaunching] = useState(false);
  const [launchMsg, setLaunchMsg] = useState("");
  const [launchErr, setLaunchErr] = useState("");
  const [ip, setIp] = useState("");

  // Terminate (saved instance)
  const [isTerminating, setIsTerminating] = useState(false);
  const [terminateMsg, setTerminateMsg] = useState("");
  const [terminateErr, setTerminateErr] = useState("");

  // Upload
  const [file, setFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState("");
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

  // ---------- Load active instances ----------
  const loadInstances = async () => {
    setIsLoadingInstances(true);
    setInstancesErr("");
    try {
      const resp = await fetch(`${API_BASE}/lambda/instances?status=active`);
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.status !== "ok") {
        throw new Error(data.message || `Failed (HTTP ${resp.status})`);
      }
      setInstances(data.instances || []);
    } catch (e) {
      setInstancesErr(e.message || "Failed to load instances.");
      setInstances([]);
    } finally {
      setIsLoadingInstances(false);
    }
  };

  // ---------- Terminate a specific instance by ID ----------
  const terminateById = async (instanceId) => {
    setTerminatingIds((prev) => new Set([...prev, instanceId]));
    try {
      const resp = await fetch(`${API_BASE}/lambda/terminate-by-id`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance_id: instanceId })
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.status !== "ok") {
        throw new Error(data.message || `Termination failed (HTTP ${resp.status})`);
      }
      await loadInstances();
    } catch (e) {
      setInstancesErr(e.message || "Termination failed.");
    } finally {
      setTerminatingIds((prev) => {
        const n = new Set(prev);
        n.delete(instanceId);
        return n;
      });
    }
  };

  // existing: fetch direction mapping
  const fetchCurrentDirectionMapping = async () => {
    try {
      const resp = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "get_current" })
      });
      const data = await resp.json().catch(() => ({}));
      if (resp.ok && data.status === "ok") setDirectionMapping(data.mapping);
    } catch (e) {
      console.error("Failed to fetch direction mapping:", e);
    }
  };

  // mount hooks
  useEffect(() => {
    fetchCurrentDirectionMapping();
    loadInstances();
  }, []);

  const rotateDirectionClockwise = async () => {
    setIsRotating(true);
    setDirectionErr("");
    setDirectionMsg("Rotating clockwise...");
    try {
      const resp = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "rotate_clockwise" })
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
        body: JSON.stringify({ action: "rotate_counterclockwise" })
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
        headers: { "Content-Type": "application/json" }
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.status !== "ready") {
        throw new Error(data.message || `Launch failed (HTTP ${resp.status})`);
      }
      setIp(data.ip || "");
      setLaunchMsg("GPU setup done ✅");
      await loadInstances();
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
        headers: { "Content-Type": "application/json" }
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
      await loadInstances();
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
    setProcessMsg(
      "Processing video on GPU… this can take several minutes. Please don't refresh or close this tab."
    );
    try {
      const body = { ip: ip || undefined, video: uploadedFilename || undefined };
      const resp = await fetch(`${API_BASE}/lambda/upload-video-and-run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
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

  const downloadFiles = async (type = "both") => {
    setIsDownloading(true);
    setDownloadErr("");
    try {
      const params = new URLSearchParams();
      if (ip) params.set("ip", ip);
      params.set("type", type);
      const resp = await fetch(`${API_BASE}/lambda/download-output?${params.toString()}`);
      if (!resp.ok) {
        const maybeJson = await resp.json().catch(() => ({}));
        throw new Error(maybeJson.message || `Download failed (HTTP ${resp.status})`);
      }
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      // Set the filename based on the type
      if (type === "video") {
        a.download = "output_detr_motion_filtered.mp4";
      } else if (type === "csv") {
        a.download = "traffic_analysis.csv";
      } else {
        a.download = "outputs_bundle.zip";
      }
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
      {/* ---------- User Header with Logout ---------- */}
      <div
        className="dashboard-user-header"
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "16px",
          backgroundColor: "#f8f9fa",
          borderRadius: "8px",
          marginBottom: "16px"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ fontSize: "20px" }}>👤</span>
          <div>
            <div style={{ fontWeight: "bold", fontSize: "16px" }}>
              Welcome, {userInfo?.username || "User"}
            </div>
            <div style={{ fontSize: "12px", color: "#666" }}>
              Logged in:{" "}
              {userInfo?.loginTime ? new Date(userInfo.loginTime).toLocaleString() : "Unknown"}
            </div>
          </div>
        </div>
        <button
          onClick={handleLogout}
          className="dashboard-btn dashboard-btn-danger"
          style={{ fontSize: "14px", padding: "8px 16px" }}
        >
          Logout
        </button>
      </div>

      {/* ---------- Running Instances panel ---------- */}
      <div className="dashboard-card" style={{ marginBottom: 16 }}>
        <div
          className="dashboard-card-title"
          style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}
        >
          <div>
            <span className="dashboard-card-icon">📡</span>
            Running Instances
          </div>
          <button
            onClick={loadInstances}
            disabled={isLoadingInstances}
            className={`dashboard-btn dashboard-btn-secondary ${
              isLoadingInstances ? "dashboard-loading" : ""
            }`}
          >
            {isLoadingInstances && <div className="dashboard-spinner"></div>}
            Refresh
          </button>
        </div>

        {instancesErr && (
          <div className="dashboard-status dashboard-error" style={{ marginTop: 8 }}>
            <span className="dashboard-status-icon">❌</span>
            {instancesErr}
          </div>
        )}

        <div className="instances-table-wrapper">
          {isLoadingInstances ? (
            <div className="dashboard-status">Loading instances…</div>
          ) : instances.length === 0 ? (
            <div className="dashboard-status">No active instances.</div>
          ) : (
            <table className="dashboard-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Name</th>
                  <th>Status</th>
                  <th>IP</th>
                  <th>Type</th>
                  <th>Region</th>
                  <th>Created</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {instances.map((i) => {
                  const busy = terminatingIds.has(i.id);
                  return (
                    <tr key={i.id}>
                      <td>
                        <code>{i.id}</code>
                      </td>
                      <td>{i.name || "-"}</td>
                      <td>{i.status}</td>
                      <td>{i.ip || "-"}</td>
                      <td>{i.instance_type_name || "-"}</td>
                      <td>{i.region_name || "-"}</td>
                      <td>{i.created_at ? new Date(i.created_at).toLocaleString() : "-"}</td>
                      <td>
                        <button
                          onClick={() => terminateById(i.id)}
                          disabled={busy}
                          className={`dashboard-btn dashboard-btn-danger ${
                            busy ? "dashboard-loading" : ""
                          }`}
                          title="Terminate this instance"
                        >
                          {busy && <div className="dashboard-spinner"></div>}
                          Terminate
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* ------- Existing header ------- */}
      <div className="dashboard-header">
        <h1 className="dashboard-title">
          <span className="dashboard-icon">⚡</span>
          GPU Processing Center
        </h1>
        <p className="dashboard-subtitle">Manage your GPU instances and process videos with ease</p>
      </div>

      {/* ------- rest of your existing UI stays exactly the same ------- */}
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
              className={`dashboard-btn dashboard-btn-primary ${
                isLaunching ? "dashboard-loading" : ""
              }`}
            >
              {isLaunching && <div className="dashboard-spinner"></div>}
              {isLaunching ? "Launching…" : "Start GPU"}
            </button>
            <button
              onClick={terminateInstance}
              disabled={isTerminating}
              className={`dashboard-btn dashboard-btn-danger ${
                isTerminating ? "dashboard-loading" : ""
              }`}
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
              className={`dashboard-btn dashboard-btn-secondary ${
                !file || isUploading ? "dashboard-disabled" : ""
              } ${isUploading ? "dashboard-loading" : ""}`}
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
                className={`dashboard-btn dashboard-btn-secondary ${
                  isRotating ? "dashboard-loading" : ""
                }`}
              >
                {isRotating && <div className="dashboard-spinner"></div>}
                Rotate Clockwise ↻
              </button>
              <button
                onClick={rotateDirectionCounterclockwise}
                disabled={isRotating}
                className={`dashboard-btn dashboard-btn-secondary ${
                  isRotating ? "dashboard-loading" : ""
                }`}
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

        <div className="dashboard-card">
          <h2 className="dashboard-card-title">
            <span className="dashboard-card-icon">⚙️</span>
            Video Processing
          </h2>
          <div className="dashboard-process-area">
            <p className="dashboard-process-description">
              Process your uploaded video using GPU acceleration. This operation may take several
              minutes.
            </p>
            <button
              onClick={processVideo}
              disabled={isProcessing || !uploadedFilename}
              title={!uploadedFilename ? "Upload a video first" : ""}
              className={`dashboard-btn dashboard-btn-accent ${
                isProcessing || !uploadedFilename ? "dashboard-disabled" : ""
              } ${isProcessing ? "dashboard-loading" : ""}`}
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

        <div className="dashboard-card">
          <h2 className="dashboard-card-title">
            <span className="dashboard-card-icon">📥</span>
            Download Results
          </h2>
          <div className="dashboard-download-area">
            <p className="dashboard-download-description">
              Download your processed results. You can download the video and CSV separately or get
              both in a ZIP file.
            </p>
            <div className="dashboard-button-group">
              <button
                onClick={() => downloadFiles("video")}
                disabled={isDownloading}
                className={`dashboard-btn dashboard-btn-success ${
                  isDownloading ? "dashboard-loading" : ""
                }`}
              >
                {isDownloading && <div className="dashboard-spinner"></div>}
                {isDownloading ? "Downloading…" : "Download Video"}
              </button>
              <button
                onClick={() => downloadFiles("csv")}
                disabled={isDownloading}
                className={`dashboard-btn dashboard-btn-success ${
                  isDownloading ? "dashboard-loading" : ""
                }`}
              >
                {isDownloading && <div className="dashboard-spinner"></div>}
                {isDownloading ? "Downloading…" : "Download CSV"}
              </button>
              <button
                onClick={() => downloadFiles("both")}
                disabled={isDownloading}
                className={`dashboard-btn dashboard-btn-secondary ${
                  isDownloading ? "dashboard-loading" : ""
                }`}
              >
                {isDownloading && <div className="dashboard-spinner"></div>}
                {isDownloading ? "Downloading…" : "Download All (ZIP)"}
              </button>
            </div>
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
