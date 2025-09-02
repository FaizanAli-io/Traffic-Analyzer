import React, { useState, useEffect } from 'react';
import './Dashboard.css';

function Dashboard({ onLogout }) {
  const [gpuStatus, setGpuStatus] = useState({ running: false });
  const [gpuLoading, setGpuLoading] = useState(false);
  const [statusError, setStatusError] = useState('');

  // Video processing state
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [jobId, setJobId] = useState('');
  const [videoId, setVideoId] = useState('');
  const [processStatus, setProcessStatus] = useState('');
  const [resultUrl, setResultUrl] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoError, setVideoError] = useState('');

  // Mock GPU status fetching - remove API calls for MVP
  const fetchGpuStatus = async () => {
    try {
      // Simulate successful status fetch
      setStatusError('');
    } catch (error) {
      setStatusError('Network error');
    }
  };

  useEffect(() => {
    fetchGpuStatus();
    const interval = setInterval(fetchGpuStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const toggleGpu = async () => {
    setGpuLoading(true);
    setStatusError('');
    
    // Simulate API call delay
    setTimeout(() => {
      setGpuStatus(prev => ({ running: !prev.running }));
      setGpuLoading(false);
    }, 1500);
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
      if (validTypes.includes(file.type) || file.name.toLowerCase().endsWith('.mov')) {
        setSelectedFile(file);
        setVideoError('');
        setJobId('');
        setVideoId('');
        setProcessStatus('');
        setResultUrl('');
        setUploadStatus('');
      } else {
        setVideoError('Please select a valid video file (MP4, MOV)');
        setSelectedFile(null);
      }
    }
  };

  const uploadAndProcess = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setVideoError('');
    setUploadStatus('Uploading...');

    // Simulate upload process
    setTimeout(() => {
      const mockJobId = 'job_' + Math.random().toString(36).substr(2, 9);
      const mockVideoId = 'video_' + Math.random().toString(36).substr(2, 9);
      
      setJobId(mockJobId);
      setVideoId(mockVideoId);
      setUploadStatus('Upload complete! Processing...');
      setIsUploading(false);
      setIsProcessing(true);
      
      pollJobStatus(mockJobId, mockVideoId);
    }, 2000);
  };

  const pollJobStatus = async (jobId, videoId) => {
    let pollCount = 0;
    const maxPolls = 10; // Simulate 30 seconds of processing
    
    const pollInterval = setInterval(() => {
      pollCount++;
      
      if (pollCount < maxPolls) {
        setProcessStatus(`Status: Processing... (${pollCount}/${maxPolls})`);
      } else {
        clearInterval(pollInterval);
        setIsProcessing(false);
        setProcessStatus('Status: Completed');
        fetchResult(videoId);
      }
    }, 3000);
  };

  const fetchResult = async (videoId) => {
    try {
      // Simulate successful result - use a demo video URL
      setResultUrl('https://www.w3schools.com/html/mov_bbb.mp4');
      setProcessStatus('Processing complete!');
    } catch (error) {
      setVideoError('Network error while fetching result');
    }
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1 className="dashboard-title">GPU Dashboard</h1>
        <button onClick={onLogout} className="dashboard-logout-btn">
          Logout
        </button>
      </div>

      <div className="dashboard-content">
        {/* GPU Controls */}
        <div className="dashboard-card">
          <h2 className="dashboard-card-title">GPU Controls</h2>
          
          <div className="dashboard-gpu-status">
            <span className="dashboard-status-label">Status:</span>
            <span className={`dashboard-status-value ${gpuStatus.running ? 'dashboard-status-running' : 'dashboard-status-stopped'}`}>
              {gpuStatus.running ? 'Running' : 'Stopped'}
            </span>
            
            <button
              onClick={toggleGpu}
              disabled={gpuLoading}
              className={`dashboard-gpu-button ${
                gpuStatus.running 
                  ? 'dashboard-gpu-button-stop' 
                  : 'dashboard-gpu-button-start'
              }`}
            >
              {gpuLoading 
                ? 'Processing...' 
                : (gpuStatus.running ? 'Stop GPU' : 'Start GPU')
              }
            </button>
          </div>
          
          {statusError && (
            <p className="dashboard-error">{statusError}</p>
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
                accept=".mp4,.mov,video/mp4,video/quicktime"
                onChange={handleFileSelect}
                className="dashboard-file-input"
              />
              {selectedFile && (
                <p className="dashboard-file-info">
                  Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <button
              onClick={uploadAndProcess}
              disabled={!selectedFile || isUploading || isProcessing}
              className="dashboard-upload-button"
            >
              {isUploading ? 'Uploading...' : isProcessing ? 'Processing...' : 'Upload & Process'}
            </button>

            {uploadStatus && (
              <p className="dashboard-status-message">{uploadStatus}</p>
            )}
            
            {processStatus && (
              <p className="dashboard-status-message">{processStatus}</p>
            )}
            
            {videoError && (
              <p className="dashboard-error">{videoError}</p>
            )}

            {resultUrl && (
              <div className="dashboard-result">
                <h3 className="dashboard-result-title">Processed Video</h3>
                <video
                  controls
                  src={resultUrl}
                  className="dashboard-video-player"
                >
                  Your browser does not support the video tag.
                </video>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;