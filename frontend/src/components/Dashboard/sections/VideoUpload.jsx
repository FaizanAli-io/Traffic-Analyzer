import { Card, Button, StatusMessage } from "../../shared";
import { useVideoUpload } from "../../../hooks/useVideoUpload";

const VideoUpload = ({ ip }) => {
  const {
    file,
    isUploading,
    uploadMsg,
    uploadErr,
    uploadedFilename,
    isProcessing,
    processMsg,
    processErr,
    onChooseFile,
    uploadVideo,
    processVideo
  } = useVideoUpload();

  return (
    <Card>
      <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2 mb-6">
        <span className="text-xl">📹</span>
        Video Upload
      </h2>

      <div className="space-y-6">
        <div className="text-sm text-gray-600 leading-relaxed">
          <p className="mb-2">
            Upload your traffic video for analysis. Supported formats: MP4, MOV, AVI.
          </p>
          <p className="text-xs text-gray-500">
            Maximum file size: 100MB. Processing time depends on video length and complexity.
          </p>
        </div>

        <div className="space-y-4">
          <div className="relative">
            <input
              type="file"
              accept=".mp4,.mov,.avi"
              onChange={onChooseFile}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-3 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-gradient-to-r file:from-emerald-50 file:to-emerald-100 file:text-emerald-700 hover:file:from-emerald-100 hover:file:to-emerald-200 file:transition-all file:duration-200 border border-gray-200 rounded-lg bg-gray-50 focus-within:ring-2 focus-within:ring-emerald-500 focus-within:border-emerald-500 transition-all duration-200"
            />
          </div>

          {file && (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-emerald-800">Selected File:</p>
                  <p className="text-sm text-emerald-600 font-mono">{file.name}</p>
                  <p className="text-xs text-emerald-500 mt-1">
                    Size: {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
                <span className="text-emerald-500 text-xl">✓</span>
              </div>
            </div>
          )}

          {uploadedFilename && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm font-medium text-blue-800">Uploaded as:</p>
              <p className="text-sm text-blue-600 font-mono">{uploadedFilename}</p>
            </div>
          )}

          <Button
            onClick={uploadVideo}
            disabled={!file}
            loading={isUploading}
            variant="primary"
            className="w-full py-3"
          >
            {isUploading ? "Uploading…" : "Upload Video"}
          </Button>
        </div>

        <div className="space-y-3">
          {uploadErr && <StatusMessage type="error">{uploadErr}</StatusMessage>}
          {uploadMsg && <StatusMessage type="success">{uploadMsg}</StatusMessage>}
        </div>

        {/* Video Processing Section */}
        {uploadedFilename && (
          <>
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2 mb-4">
                <span className="text-lg">⚙️</span>
                Video Processing
              </h3>

              <div className="space-y-4">
                <p className="text-sm text-gray-600">
                  Process your uploaded video using GPU acceleration. This operation may take
                  several minutes.
                </p>

                <Button
                  onClick={() => processVideo(ip)}
                  disabled={isProcessing || !uploadedFilename}
                  loading={isProcessing}
                  variant="accent"
                  className="w-full py-3"
                >
                  {isProcessing ? "Processing…" : "Process Video"}
                </Button>
              </div>

              <div className="space-y-3 mt-4">
                {processErr && <StatusMessage type="error">{processErr}</StatusMessage>}
                {processMsg && <StatusMessage type="success">{processMsg}</StatusMessage>}
              </div>
            </div>
          </>
        )}
      </div>
    </Card>
  );
};

export default VideoUpload;
