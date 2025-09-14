import { useState, useEffect } from "react";
import { gpuAPI } from "../services/api";

export const useInstances = () => {
  const [instances, setInstances] = useState([]);
  const [isLoadingInstances, setIsLoadingInstances] = useState(false);
  const [instancesErr, setInstancesErr] = useState("");
  const [terminatingIds, setTerminatingIds] = useState(new Set());

  const loadInstances = async () => {
    setIsLoadingInstances(true);
    setInstancesErr("");

    try {
      const { response, data } = await gpuAPI.getInstances();

      if (!response.ok || data.status !== "ok") {
        throw new Error(data.message || `Failed to load instances (HTTP ${response.status})`);
      }

      setInstances(data.instances || []);
    } catch (e) {
      setInstancesErr(e.message || "Failed to load instances.");
    } finally {
      setIsLoadingInstances(false);
    }
  };

  const terminateSpecificInstance = async (instanceId) => {
    setTerminatingIds((prev) => new Set(prev).add(instanceId));

    try {
      const { response, data } = await gpuAPI.terminateSpecificInstance(instanceId);

      if (!response.ok || data.status !== "ok") {
        throw new Error(data.message || `Failed to terminate instance (HTTP ${response.status})`);
      }

      // Reload instances after successful termination
      await loadInstances();
    } catch (e) {
      setInstancesErr(e.message || "Failed to terminate instance.");
    } finally {
      setTerminatingIds((prev) => {
        const newSet = new Set(prev);
        newSet.delete(instanceId);
        return newSet;
      });
    }
  };

  // Load instances on mount
  useEffect(() => {
    loadInstances();
  }, []);

  return {
    // State
    instances,
    isLoadingInstances,
    instancesErr,
    terminatingIds,

    // Actions
    loadInstances,
    terminateSpecificInstance
  };
};
