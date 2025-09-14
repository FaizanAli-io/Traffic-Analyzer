import { useState } from "react";
import { videoAPI } from "../services/api";

export const useVideoUpload = () => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploadErr, setUploadErr] = useState("");
  const [uploadedFilename, setUploadedFilename] = useState("");

  // Video processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processMsg, setProcessMsg] = useState("");
  const [processErr, setProcessErr] = useState("");

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
      const { response, data } = await videoAPI.uploadVideo(file);

      if (!response.ok || data.status !== "ok") {
        throw new Error(data.message || `Upload failed (HTTP ${response.status})`);
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

  const processVideo = async (ip) => {
    setIsProcessing(true);
    setProcessErr("");
    setProcessMsg(
      "Processing video on GPU… this can take several minutes. Please don't refresh or close this tab."
    );

    try {
      const { response, data } = await videoAPI.processVideo(ip, uploadedFilename);

      if (!response.ok || (data.status && data.status !== "ok")) {
        throw new Error(data.message || `Processing failed (HTTP ${response.status})`);
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

  return {
    // State
    file,
    isUploading,
    uploadMsg,
    uploadErr,
    uploadedFilename,
    isProcessing,
    processMsg,
    processErr,

    // Actions
    onChooseFile,
    uploadVideo,
    processVideo
  };
};
