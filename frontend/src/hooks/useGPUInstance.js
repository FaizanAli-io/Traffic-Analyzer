import { useEffect, useState } from "react";
import { gpuAPI } from "../services/api";

export const useGPUInstance = (onIpChange) => {
  const [isLaunching, setIsLaunching] = useState(false);
  const [launchMsg, setLaunchMsg] = useState("");
  const [launchErr, setLaunchErr] = useState("");
  const [ip, setIp] = useState("");

  // Selection state
  const [regionName, setRegionName] = useState("");
  const [instanceTypeName, setInstanceTypeName] = useState("");
  const [isLoadingOptions, setIsLoadingOptions] = useState(false);
  // Single source of truth for options; derive the rest
  // Shape: [{ name, regions: [string], info: { description, gpu_description, price_cents_per_hour, specs } }]
  const [options, setOptions] = useState([]);

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

  // Load options from backend (normalize to a simple array)
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

      // Filter to only types with capacity and sort by name
      const filtered = normalized
        .filter((t) => t.regions && t.regions.length > 0)
        .sort((a, b) => a.name.localeCompare(b.name));
      setOptions(filtered);

      // Defaults: prefer existing selections if still valid, else pick the first available
      const currentTypeValid =
        instanceTypeName && filtered.some((t) => t.name === instanceTypeName);
      const chosenType = currentTypeValid ? instanceTypeName : filtered[0]?.name || "";
      if (!currentTypeValid && chosenType) setInstanceTypeName(chosenType);

      const chosenTypeObj = filtered.find((t) => t.name === chosenType);
      const regsForType = chosenTypeObj?.regions || [];
      const currentRegionValid = regionName && regsForType.includes(regionName);
      if (!currentRegionValid) setRegionName(regsForType[0] || "");
    } catch (e) {
      // Non-fatal for UI; keep prior values
      console.warn("Failed to load instance options:", e);
    } finally {
      setIsLoadingOptions(false);
    }
  };

  // When instance type changes, recompute available regions and default selection
  useEffect(() => {
    if (!instanceTypeName) return;
    const selected = options.find((t) => t.name === instanceTypeName);
    const regs = selected?.regions || [];
    if (!regionName || !regs.includes(regionName)) setRegionName(regs[0] || "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    // State
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
    // Expose a simple map so UI can show prices in the dropdown
    typeInfoByName: Object.fromEntries(options.map((t) => [t.name, t.info])),
    isTerminating,
    terminateMsg,
    terminateErr,
    isRunning,
    runMsg,
    runErr,

    // Actions
    startGpu,
    setRegionName,
    setInstanceTypeName,
    loadOptions,
    terminateInstance,
    runScript
  };
};
