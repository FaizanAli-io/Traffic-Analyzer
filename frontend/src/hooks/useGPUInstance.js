import { useState } from "react";
import { gpuAPI } from "../services/api";

export const useGPUInstance = (onIpChange) => {
  const [isLaunching, setIsLaunching] = useState(false);
  const [launchMsg, setLaunchMsg] = useState("");
  const [launchErr, setLaunchErr] = useState("");
  const [ip, setIp] = useState("");

  const [isTerminating, setIsTerminating] = useState(false);
  const [terminateMsg, setTerminateMsg] = useState("");
  const [terminateErr, setTerminateErr] = useState("");

  const [isRunning, setIsRunning] = useState(false);
  const [runMsg, setRunMsg] = useState("");
  const [runErr, setRunErr] = useState("");

  const updateIp = (newIp) => {
    setIp(newIp);
    if (onIpChange) onIpChange(newIp);
  };

  const startGpu = async () => {
    setIsLaunching(true);
    setLaunchMsg("");
    setLaunchErr("");

    try {
      const { response, data } = await gpuAPI.launchInstance();

      if (!response.ok || data.status !== "ready") {
        throw new Error(data.message || `Launch failed (HTTP ${response.status})`);
      }

      updateIp(data.ip || "");
      setLaunchMsg(`GPU instance launched successfully! IP: ${data.ip || "N/A"} ✅`);
    } catch (e) {
      setLaunchErr(e.message || "Failed to launch GPU instance.");
      setLaunchMsg("");
    } finally {
      setIsLaunching(false);
    }
  };

  const terminateInstance = async () => {
    setIsTerminating(true);
    setTerminateMsg("");
    setTerminateErr("");

    try {
      const { response, data } = await gpuAPI.terminateInstance();

      if (!response.ok || data.status !== "ready") {
        throw new Error(data.message || `Termination failed (HTTP ${response.status})`);
      }

      updateIp("");
      setLaunchMsg("");
      setTerminateMsg("GPU instance terminated successfully! ✅");
    } catch (e) {
      setTerminateErr(e.message || "Failed to terminate GPU instance.");
      setTerminateMsg("");
    } finally {
      setIsTerminating(false);
    }
  };

  const runScript = async () => {
    setIsRunning(true);
    setRunMsg("");
    setRunErr("");

    try {
      const { response, data } = await gpuAPI.runScript();

      if (!response.ok || data.status !== "ready") {
        throw new Error(data.message || `Script execution failed (HTTP ${response.status})`);
      }

      setRunMsg("Script executed successfully! Processing complete. ✅");
    } catch (e) {
      setRunErr(e.message || "Failed to execute script.");
      setRunMsg("");
    } finally {
      setIsRunning(false);
    }
  };

  return {
    // State
    isLaunching,
    launchMsg,
    launchErr,
    ip,
    isTerminating,
    terminateMsg,
    terminateErr,
    isRunning,
    runMsg,
    runErr,

    // Actions
    startGpu,
    terminateInstance,
    runScript
  };
};
