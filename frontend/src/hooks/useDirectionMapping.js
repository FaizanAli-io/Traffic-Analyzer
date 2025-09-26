import { useState, useEffect } from "react";
import { directionAPI } from "../services/api";

export const useDirectionMapping = () => {
  const [directionMapping, setDirectionMapping] = useState({
    top: "North",
    right: "East",
    bottom: "South",
    left: "West"
  });
  const [isRotating, setIsRotating] = useState(false);
  const [directionMsg, setDirectionMsg] = useState("");
  const [directionErr, setDirectionErr] = useState("");

  const fetchCurrentDirectionMapping = async () => {
    try {
      const { response, data } = await directionAPI.getCurrentMapping();

      if (response.ok && data.status === "ok") {
        setDirectionMapping(data.mapping);
      }
    } catch (e) {
      console.error("Failed to fetch direction mapping:", e);
    }
  };

  const updateDirectionMapping = async (newMapping) => {
    setDirectionErr("");
    setDirectionMsg("");

    try {
      const { response, data } = await directionAPI.updateMapping(newMapping);

      if (!response.ok || data.status !== "ok") {
        throw new Error(
          data.message || `Failed to update direction mapping (HTTP ${response.status})`
        );
      }

      setDirectionMapping(newMapping);
      setDirectionMsg("Direction mapping updated successfully! ✅");
    } catch (e) {
      setDirectionErr(e.message || "Failed to update direction mapping.");
    }
  };

  const rotateDirections = async () => {
    setIsRotating(true);
    setDirectionErr("");
    setDirectionMsg("Rotating clockwise...");

    try {
      const { response, data } = await directionAPI.rotateMappingCounterClockwise();

      if (!response.ok || data.status !== "ok") {
        throw new Error(data.message || `Rotation failed (HTTP ${response.status})`);
      }

      setDirectionMapping(data.mapping);
      setDirectionMsg("Rotated clockwise ↻");
    } catch (e) {
      setDirectionErr(e.message || "Rotation failed.");
      setDirectionMsg("");
    } finally {
      setIsRotating(false);
    }
  };

  const rotateDirectionsCounterClockwise = async () => {
    setIsRotating(true);
    setDirectionErr("");
    setDirectionMsg("Rotating counter-clockwise...");

    try {
      const { response, data } = await directionAPI.rotateMapping();

      if (!response.ok || data.status !== "ok") {
        throw new Error(data.message || `Rotation failed (HTTP ${response.status})`);
      }

      setDirectionMapping(data.mapping);
      setDirectionMsg("Rotated counter-clockwise ↺");
    } catch (e) {
      setDirectionErr(e.message || "Rotation failed.");
      setDirectionMsg("");
    } finally {
      setIsRotating(false);
    }
  };

  // Load current mapping on mount
  useEffect(() => {
    fetchCurrentDirectionMapping();
  }, []);

  return {
    // State
    directionMapping,
    isRotating,
    directionMsg,
    directionErr,

    // Actions
    updateDirectionMapping,
    rotateDirections,
    rotateDirectionsCounterClockwise,
    fetchCurrentDirectionMapping
  };
};
