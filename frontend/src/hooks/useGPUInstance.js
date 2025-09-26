import { useEffect, useState } from "react";
import { gpuAPI } from "../services/api";

export const useGPUInstance = (onIpChange) => {
  const [ip, setIp] = useState("");
  const [launchMsg, setLaunchMsg] = useState("");
  const [launchErr, setLaunchErr] = useState("");
  const [isLaunching, setIsLaunching] = useState(false);

  const [options, setOptions] = useState([]);
  const [regionName, setRegionName] = useState("");
  const [instanceTypeName, setInstanceTypeName] = useState("");
  const [isLoadingOptions, setIsLoadingOptions] = useState(false);

  const [terminateMsg, setTerminateMsg] = useState("");
  const [terminateErr, setTerminateErr] = useState("");
  const [isTerminating, setIsTerminating] = useState(false);

  const [runMsg, setRunMsg] = useState("");
  const [runErr, setRunErr] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  const updateIp = (newIp) => {
    setIp(newIp);
    if (onIpChange) onIpChange(newIp);
  };

  const startGpu = async () => {
    setIsLaunching(true);
    setLaunchMsg("");
    setLaunchErr("");

    try {
      if (!instanceTypeName || !regionName) {
        throw new Error("Select an instance type and region before launching.");
      }
      const { response, data } = await gpuAPI.launchInstance({
        region_name: regionName,
        instance_type_name: instanceTypeName
      });

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

  const loadOptions = async () => {
    setIsLoadingOptions(true);
    try {
      const { response, data } = await gpuAPI.listInstanceTypes();
      if (!response.ok) throw new Error(`Failed to load instance types (HTTP ${response.status})`);
      const raw = data?.data || data?.instance_types || data || [];

      const normalized = [];
      if (Array.isArray(raw)) {
        for (const it of raw) {
          const info = it?.instance_type || it || {};
          const name = info?.name || it?.name;
          if (!name) continue;
          const regs = (it?.regions || it?.available_regions || [])
            .map((r) => r?.name || r?.region_name)
            .filter(Boolean);
          const uniqRegs = Array.from(new Set(regs)).sort();
          normalized.push({
            name,
            regions: uniqRegs,
            info: {
              description: info.description,
              gpu_description: info.gpu_description,
              price_cents_per_hour: info.price_cents_per_hour,
              specs: info.specs
            }
          });
        }
      } else if (raw && typeof raw === "object") {
        for (const [key, val] of Object.entries(raw)) {
          const info = val?.instance_type || {};
          const name = info?.name || key;
          if (!name) continue;
          const regs = (val?.regions_with_capacity_available || [])
            .map((r) => r?.name || r?.region_name)
            .filter(Boolean);
          const uniqRegs = Array.from(new Set(regs)).sort();
          normalized.push({
            name,
            regions: uniqRegs,
            info: {
              description: info.description,
              gpu_description: info.gpu_description,
              price_cents_per_hour: info.price_cents_per_hour,
              specs: info.specs
            }
          });
        }
      }

      const filtered = normalized
        .filter((t) => t.regions && t.regions.length > 0)
        .sort((a, b) => {
          const ap = a.info?.price_cents_per_hour;
          const bp = b.info?.price_cents_per_hour;
          if (typeof ap === "number" && typeof bp === "number" && ap !== bp) return ap - bp;

          return a.name.localeCompare(b.name);
        });
      setOptions(filtered);

      const currentTypeValid =
        instanceTypeName && filtered.some((t) => t.name === instanceTypeName);
      const chosenType = currentTypeValid ? instanceTypeName : filtered[0]?.name || "";
      if (!currentTypeValid && chosenType) setInstanceTypeName(chosenType);

      const chosenTypeObj = filtered.find((t) => t.name === chosenType);
      const regsForType = chosenTypeObj?.regions || [];
      const currentRegionValid = regionName && regsForType.includes(regionName);
      if (!currentRegionValid) setRegionName(regsForType[0] || "");
    } catch (e) {
      console.warn("Failed to load instance options:", e);
    } finally {
      setIsLoadingOptions(false);
    }
  };

  useEffect(() => {
    if (!instanceTypeName) return;
    const selected = options.find((t) => t.name === instanceTypeName);
    const regs = selected?.regions || [];
    if (!regionName || !regs.includes(regionName)) setRegionName(regs[0] || "");
  }, [instanceTypeName, options]);

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
    isLaunching,
    launchMsg,
    launchErr,
    ip,
    regionName,
    instanceTypeName,
    availableRegions: options.find((t) => t.name === instanceTypeName)?.regions || [],
    availableInstanceTypes: options.map((t) => t.name),
    isLoadingOptions,
    selectedTypeInfo: options.find((t) => t.name === instanceTypeName)?.info,
    typeInfoByName: Object.fromEntries(options.map((t) => [t.name, t.info])),
    isTerminating,
    terminateMsg,
    terminateErr,
    isRunning,
    runMsg,
    runErr,

    startGpu,
    setRegionName,
    setInstanceTypeName,
    loadOptions,
    terminateInstance,
    runScript
  };
};
