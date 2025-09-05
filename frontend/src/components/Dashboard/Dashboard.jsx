import React, { useState, useEffect } from 'react';
import './Dashboard.css';

const API_BASE = import.meta.env.VITE_BACKEND_SERVER || 'http://localhost:5000';

function Dashboard({ onLogout }) {
  const [gpuStatus, setGpuStatus] = useState({ running: false, ip: null });
  const [gpuLoading, setGpuLoading] = useState(false);
  const [statusError, setStatusError] = useState('');
  const [launchLogs, setLaunchLogs] = useState([]);

  // Video processing state
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(''); // ✅ add this
  const [uploadStatus, setUploadStatus] = useState('');
  const [processStatus, setProcessStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoError, setVideoError] = useState('');

  // ---- Health check ----
  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) throw new Error('Backend not responding');
      setStatusError('');
    } catch {
      setStatusError('Backend connection failed');
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // ---- GPU controls ----
  const startGpu = async () => {
    setGpuLoading(true);
    setStatusError('');
    setLaunchLogs(['Starting GPU instance...']);
    try {
      const response = await fetch(`${API_BASE}/lambda/launch-and-setup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json();
      if (response.ok && data.status === 'launched+provisioned') {
        setGpuStatus({ running: true, ip: data.ip });
        setLaunchLogs([
          'Instance launched successfully!',
          `IP: ${data.ip}`,
          'System bootstrapped',
          'Dependencies installed',
          'Ready for video processing',
        ]);
      } else {
        throw new Error(data.message || `Launch failed: ${response.status}`);
      }
    } catch (err) {
      setStatusError(`GPU launch failed: ${err.message}`);
      setLaunchLogs([`Error: ${err.message}`]);
    } finally {
      setGpuLoading(false);
    }
  };

  const stopGpu = async () => {
    // (Optional) call your terminate API here
    setGpuStatus({ running: false, ip: null });
    setLaunchLogs(['GPU instance stopped']);
  };

  // ---- File selection ----
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
    const isValidType =
      validTypes.includes(file.type) ||
      file.name.toLowerCase().endsWith('.mp4') ||
      file.name.toLowerCase().endsWith('.mov') ||
      file.name.toLowerCase().endsWith('.avi');

    if (!isValidType) {
      setVideoError('Please select a valid video file (MP4, MOV, AVI).');
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setVideoError('');
    setProcessStatus('');
    setUploadStatus('');
    setUploadedFileName(''); // reset previously uploaded filename
  };

  // ---- API helpers ----
  const uploadVideoFile = async (file) => {
    const formData = new FormData();
    formData.append('video', file);

    const response = await fetch(`${API_BASE}/upload-video`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Upload failed: ${response.status}`);
    }
    return await response.json();
  };

  const requestProcessVideo = async (videoFilename) => {
    const response = await fetch(`${API_BASE}/lambda/upload-video-and-run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ip: gpuStatus.ip,
        video: videoFilename,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Processing failed: ${response.status}`);
    }
    return await response.json();
  };

  // ---- Button handlers (fixed names used in JSX) ----
  const handleUploadClick = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setVideoError('');
    setUploadStatus('');
    try {
      setUploadStatus('Uploading video file...');
      const uploadResult = await uploadVideoFile(selectedFile);
      const name = uploadResult.filename || selectedFile.name;
      setUploadedFileName(name);               // ✅ save for processing step
      setUploadStatus(`Upload complete: ${name}`);
    } catch (err) {
      setVideoError(`Error: ${err.message}`);
      setUploadStatus('');
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcessClick = async () => {
    if (!uploadedFileName) return;
    if (!gpuStatus.running) {
      setVideoError('Please start the GPU instance first');
      return;
    }
    setIsProcessing(true);
    setVideoError('');
    setProcessStatus('Starting video processing on GPU...');
    try {
      const result = await requestProcessVideo(uploadedFileName);
      if (result.status === 'ok') {
        setProcessStatus('Video processing completed successfully!');
      } else {
        throw new Error(result.message || 'Processing failed');
      }
    } catch (err) {
      setVideoError(`Error: ${err.message}`);
      setProcessStatus('');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleUploadAndProcess = async () => {
    // Optional one-click path
    await handleUploadClick();
    if (uploadedFileName) await handleProcessClick();
  };

  // ---- Download ----
  const downloadResult = async () => {
    try {
      const response = await fetch(`${API_BASE}/lambda/download-output?ip=${gpuStatus.ip}`, {
        method: 'GET',
      });
      if (!response.ok) throw new Error('Download failed');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'processed_video.mp4';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setVideoError(`Download error: ${err.message}`);
    }
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1 className="dashboard-title">GPU Dashboard</h1>
        <button onClick={onLogout} className="dashboard-logout-btn">Logout</button>
      </div>

      <div className="dashboard-content">
        {/* GPU Controls */}
        <div className="dashboard-card">
          <h2 className="dashboard-card-title">GPU Controls</h2>

          <div className="dashboard-gpu-status">
            <span className="dashboard-status-label">Status:</span>
            <span
              className={`dashboard-status-value ${
                gpuStatus.running ? 'dashboard-status-running' : 'dashboard-status-stopped'
              }`}
            >
              {gpuStatus.running ? `Running (${gpuStatus.ip})` : 'Stopped'}
            </span>

            <button
              onClick={gpuStatus.running ? stopGpu : startGpu}
              disabled={gpuLoading}
              className={`dashboard-gpu-button ${
                gpuStatus.running ? 'dashboard-gpu-button-stop' : 'dashboard-gpu-button-start'
              }`}
            >
              {gpuLoading ? 'Launching...' : gpuStatus.running ? 'Stop GPU' : 'Start GPU'}
            </button>
          </div>

          {statusError && <p className="dashboard-error">{statusError}</p>}

          {launchLogs.length > 0 && (
            <div className="dashboard-logs">
              <h4>Launch Logs:</h4>
              <ul>{launchLogs.map((log, i) => <li key={i}>{log}</li>)}</ul>
            </div>
          )}
        </div>

        {/* Video Processing */}
        <div className="dashboard-card">
          <h2 className="dashboard-card-title">Video Upload & Process</h2>

          <div className="dashboard-video-section">
            <div className="dashboard-file-input-wrapper">
              <label className="dashboard-file-label">Select Video File</label>
              <input
                type="file"
                accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
                onChange={handleFileSelect}
                className="dashboard-file-input"
              />
              {selectedFile && (
                <p className="dashboard-file-info">
                  Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}

              {uploadedFileName && (
                <p className="dashboard-file-info dashboard-uploaded-info">
                  Uploaded: {uploadedFileName} ✓
                </p>
              )}
            </div>

            <div className="dashboard-button-group">
              <button
                onClick={handleUploadClick}
                disabled={!selectedFile || isUploading}
                className="dashboard-upload-button"
              >
                {isUploading ? 'Uploading...' : 'Upload Video'}
              </button>

              <button
                onClick={handleProcessClick}
                disabled={!uploadedFileName || isProcessing || !gpuStatus.running}
                className="dashboard-process-button"
              >
                {isProcessing
                  ? 'Processing...'
                  : !gpuStatus.running
                  ? 'Start GPU First'
                  : !uploadedFileName
                  ? 'Upload Video First'
                  : 'Process Video'}
              </button>

              {/* Optional one-click flow */}
              <button
                onClick={handleUploadAndProcess}
                disabled={!selectedFile || isUploading || isProcessing}
                className="dashboard-process-button"
                title="Uploads then immediately starts processing"
              >
                {isUploading || isProcessing ? 'Working...' : 'Upload & Process'}
              </button>
            </div>

            {uploadStatus && <p className="dashboard-status-message">{uploadStatus}</p>}
            {processStatus && <p className="dashboard-status-message">{processStatus}</p>}
            {videoError && <p className="dashboard-error">{videoError}</p>}

            {processStatus.includes('completed successfully') && (
              <div className="dashboard-result">
                <h3 className="dashboard-result-title">Processing Complete</h3>
                <button onClick={downloadResult} className="dashboard-download-button">
                  Download Processed Video
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
