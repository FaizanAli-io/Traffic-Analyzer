// Dashboard.jsx
import React, { useState } from "react";
import "./Dashboard.css";

const API_BASE = import.meta.env.VITE_BACKEND_SERVER || "http://localhost:5000";

export default function Dashboard() {
  // GPU
  const [isLaunching, setIsLaunching] = useState(false);
  const [launchMsg, setLaunchMsg] = useState("");
  const [launchErr, setLaunchErr] = useState("");
  const [ip, setIp] = useState("");

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

  const onChooseFile = (e) => {
    const f = e.target.files?.[0];
    setUploadErr("");
    setUploadMsg("");
    setFile(null);
    setUploadedFilename(""); // reset when choosing a new file
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
      // backend always saves as a fixed name; keep it for processing
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
        // pass what you have; backend will use saved defaults if omitted
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
      a.download = "outputs_bundle.zip"; // ZIP containing video + CSV
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
    <div style={{ maxWidth: 640, margin: "40px auto", textAlign: "center" }}>
      <h1>GPU Control</h1>

      {/* Start GPU */}
      <button
        onClick={startGpu}
        disabled={isLaunching}
        style={{ padding: "12px 20px", fontSize: 16, borderRadius: 10 }}
      >
        {isLaunching ? "Launching…" : "Start GPU"}
      </button>
      {launchMsg && <p style={{ marginTop: 12 }}>{launchMsg}</p>}
      {ip && <p style={{ marginTop: 6, fontFamily: "monospace" }}>IP: {ip}</p>}
      {launchErr && <p style={{ marginTop: 12, color: "#c00" }}>{launchErr}</p>}

      {/* Upload */}
      <hr style={{ margin: "28px 0" }} />
      <h2>Upload Video</h2>
      <input
        type="file"
        accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
        onChange={onChooseFile}
      />
      {file && (
        <p style={{ marginTop: 8 }}>
          Selected: <strong>{file.name}</strong> ({(file.size / 1024 / 1024).toFixed(2)} MB)
        </p>
      )}
      <div style={{ marginTop: 10 }}>
        <button
          onClick={uploadVideo}
          disabled={!file || isUploading}
          style={{ padding: "10px 16px", borderRadius: 8 }}
        >
          {isUploading ? "Uploading…" : "Upload Video"}
        </button>
      </div>
      {uploadMsg && <p style={{ marginTop: 10 }}>{uploadMsg}</p>}
      {uploadErr && <p style={{ marginTop: 10, color: "#c00" }}>{uploadErr}</p>}

      {/* Process */}
      <hr style={{ margin: "28px 0" }} />
      <h2>Process Video</h2>
      <button
        onClick={processVideo}
        disabled={isProcessing || !uploadedFilename}
        title={!uploadedFilename ? "Upload a video first" : ""}
        style={{ padding: "10px 16px", borderRadius: 8 }}
      >
        {isProcessing ? "Processing…" : "Process Video"}
      </button>
      {processMsg && <p style={{ marginTop: 10 }}>{processMsg}</p>}
      {processErr && <p style={{ marginTop: 10, color: "#c00" }}>{processErr}</p>}

      {/* Download */}
      <hr style={{ margin: "28px 0" }} />
      <h2>Download Results</h2>
      <p style={{ fontSize: 14, color: "#666", marginBottom: 16 }}>
        Downloads a ZIP file containing both the processed video and CSV data
      </p>
      <button
        onClick={downloadFiles}
        disabled={isDownloading}
        style={{ padding: "10px 16px", borderRadius: 8 }}
      >
        {isDownloading ? "Downloading…" : "Download ZIP (Video + CSV)"}
      </button>
      {downloadErr && <p style={{ marginTop: 10, color: "#c00" }}>{downloadErr}</p>}
    </div>
  );
}