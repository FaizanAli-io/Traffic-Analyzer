import { Card, Button, StatusMessage } from "../../shared";
import { useDownload } from "../../../hooks/useDownload";

const DownloadResults = ({ ip }) => {
  const { isDownloading, downloadErr, downloadFiles } = useDownload(ip);

  return (
    <Card>
      <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2 mb-6">
        <span className="text-xl">📥</span>
        Download Results
      </h2>

      <div className="space-y-4">
        <p className="text-gray-600 text-sm leading-relaxed">
          Download your processed results. You can download the video and CSV separately or get both
          in a ZIP file.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Button
            onClick={() => downloadFiles("video")}
            loading={isDownloading}
            variant="success"
            className="text-sm"
          >
            Download Video
          </Button>
          <Button
            onClick={() => downloadFiles("csv")}
            loading={isDownloading}
            variant="success"
            className="text-sm"
          >
            Download CSV
          </Button>
          <Button
            onClick={() => downloadFiles("both")}
            loading={isDownloading}
            variant="secondary"
            className="text-sm"
          >
            Download All (ZIP)
          </Button>
        </div>
      </div>

      <div className="mt-6 space-y-3">
        {downloadErr && <StatusMessage type="error">{downloadErr}</StatusMessage>}
      </div>
    </Card>
  );
};

export default DownloadResults;
