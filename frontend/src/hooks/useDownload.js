import { useState } from "react";
import { downloadAPI } from "../services/api";

export const useDownload = (ip) => {
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadErr, setDownloadErr] = useState("");

  const downloadFiles = async (fileType) => {
    setIsDownloading(true);
    setDownloadErr("");

    try {
      const response = await downloadAPI.downloadFiles(fileType, ip);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Download failed (HTTP ${response.status})`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;

      // Set filename based on content-disposition header or file type
      const contentDisposition = response.headers.get("content-disposition");
      let filename = "download";

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="(.+)"/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      } else {
        // Fallback based on file type
        switch (fileType) {
          case "video":
            filename = "processed_video.mp4";
            break;
          case "csv":
            filename = "traffic_data.csv";
            break;
          case "both":
            filename = "traffic_analysis_results.zip";
            break;
          default:
            filename = "download";
        }
      }

      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (e) {
      setDownloadErr(e.message || "Download failed.");
    } finally {
      setIsDownloading(false);
    }
  };

  return {
    // State
    isDownloading,
    downloadErr,

    // Actions
    downloadFiles
  };
};
