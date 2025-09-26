const API_BASE = import.meta.env.VITE_BACKEND_SERVER || "http://localhost:5000";

// Authentication API
export const authAPI = {
  login: async (credentials) => {
    const response = await fetch(`${API_BASE}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(credentials)
    });
    return response.json();
  }
};

// GPU Instance Management API
export const gpuAPI = {
  // Launch a new GPU instance
  launchInstance: async (opts = {}) => {
    const payload = {};
    if (opts.region_name) payload.region_name = opts.region_name;
    if (opts.instance_type_name) payload.instance_type_name = opts.instance_type_name;
    const response = await fetch(`${API_BASE}/lambda/launch-and-setup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  // Fetch types
  listInstanceTypes: async () => {
    const response = await fetch(`${API_BASE}/lambda/instance-types`);
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  // Terminate GPU instance
  terminateInstance: async () => {
    const response = await fetch(`${API_BASE}/lambda/terminate-instance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  // Get all instances
  getInstances: async () => {
    const response = await fetch(`${API_BASE}/lambda/instances?status=active`);
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  // Terminate specific instance
  terminateSpecificInstance: async (instanceId) => {
    const response = await fetch(`${API_BASE}/lambda/terminate-by-id`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instance_id: instanceId })
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  // Run script on instance
  runScript: async () => {
    const response = await fetch(`${API_BASE}/lambda/run-script`, { method: "POST" });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  }
};

// Video Upload API
export const videoAPI = {
  uploadVideo: async (file) => {
    const formData = new FormData();
    formData.append("video", file);
    const response = await fetch(`${API_BASE}/upload-video`, {
      method: "POST",
      body: formData
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  processVideo: async (ip, uploadedFilename) => {
    const body = { ip: ip || undefined, video: uploadedFilename || undefined };
    const response = await fetch(`${API_BASE}/lambda/upload-video-and-run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  }
};

// Direction Mapping API
export const directionAPI = {
  getCurrentMapping: async () => {
    const response = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "get_current" })
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  updateMapping: async (mapping) => {
    const response = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "update", mapping })
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  rotateMapping: async () => {
    const response = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "rotate_clockwise" })
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  },

  rotateMappingCounterClockwise: async () => {
    const response = await fetch(`${API_BASE}/lambda/set-direction-orientation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "rotate_counterclockwise" })
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  }
};

// Download API
export const downloadAPI = {
  downloadFiles: async (fileType, ip) => {
    const params = new URLSearchParams();
    if (ip) params.set("ip", ip);
    params.set("type", fileType);

    const response = await fetch(`${API_BASE}/lambda/download-output?${params.toString()}`, {
      method: "GET"
    });
    return response;
  }
};
