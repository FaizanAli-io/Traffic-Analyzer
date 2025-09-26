import { Card, Button, StatusMessage } from "../../shared";
import { useGPUInstance } from "../../../hooks/useGPUInstance";

const GPUControl = ({ setIp }) => {
  const {
    isLaunching,
    launchMsg,
    launchErr,
    ip,
    regionName,
    instanceTypeName,
    availableRegions,
    availableInstanceTypes,
    isLoadingOptions,
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
    runScript,
    selectedTypeInfo,
    typeInfoByName
  } = useGPUInstance(setIp);

  // Options will be fetched on-demand via the Fetch button

  return (
    <Card>
      <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2 mb-6">
        <span className="text-xl">🖥️</span>
        GPU Instance Control
      </h2>

      <div className="space-y-6">
        <div className="text-sm text-gray-600 leading-relaxed">
          <p className="mb-2">
            Launch a Lambda Cloud GPU instance for video processing. The instance will be
            automatically configured with the required dependencies.
          </p>
          {ip && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mt-3">
              <span className="font-medium text-blue-800">Active Instance IP: </span>
              <span className="font-mono text-blue-600">{ip}</span>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-gray-700 mb-1">Instance Type</label>
              <select
                className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
                value={instanceTypeName}
                onChange={(e) => setInstanceTypeName(e.target.value)}
                disabled={isLoadingOptions || isLaunching}
              >
                {isLoadingOptions && availableInstanceTypes.length === 0 ? (
                  <option value="">Loading types...</option>
                ) : availableInstanceTypes.length === 0 ? (
                  <option value="">No types available</option>
                ) : (
                  availableInstanceTypes.map((t) => {
                    const info = typeInfoByName?.[t];
                    const cents = info?.price_cents_per_hour;
                    const price =
                      typeof cents === "number" ? ` ($${(cents / 100).toFixed(2)}/hr)` : "";
                    return (
                      <option key={t} value={t}>
                        {t}
                        {price}
                      </option>
                    );
                  })
                )}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-700 mb-1">Region</label>
              <select
                className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
                value={regionName}
                onChange={(e) => setRegionName(e.target.value)}
                disabled={isLoadingOptions || isLaunching || availableRegions.length === 0}
              >
                {isLoadingOptions && availableRegions.length === 0 ? (
                  <option value="">Loading regions...</option>
                ) : availableRegions.length === 0 ? (
                  <option value="">No regions for selected type</option>
                ) : (
                  availableRegions.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))
                )}
              </select>
            </div>
            <div className="flex items-end">
              <Button
                onClick={loadOptions}
                loading={isLoadingOptions}
                variant="accent"
                className="h-10 w-full"
              >
                {isLoadingOptions ? "Fetching..." : "Fetch regions"}
              </Button>
            </div>
          </div>

          {/* Selected instance info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-900">
            {selectedTypeInfo ? (
              <div>
                <div className="text-base sm:text-lg font-semibold mb-6 text-center">
                  GPU Cost & Specifications
                </div>
                {(() => {
                  const s = selectedTypeInfo.specs || {};
                  const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : n ?? "?");
                  return (
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-blue-800">
                      <div className="font-medium">
                        <span className="text-blue-700/80">Cost:</span>{" "}
                        {selectedTypeInfo.price_cents_per_hour != null
                          ? `$${(selectedTypeInfo.price_cents_per_hour / 100).toFixed(2)}/hr`
                          : "N/A"}
                      </div>
                      <div className="font-medium">
                        <span className="text-blue-700/80">GPU:</span>{" "}
                        {selectedTypeInfo.gpu_description || "N/A"}
                      </div>
                      <div className="font-medium">
                        <span className="text-blue-700/80">GPUs:</span> {fmt(s.gpus)}
                      </div>
                      <div className="font-medium">
                        <span className="text-blue-700/80">vCPUs:</span> {fmt(s.vcpus)}
                      </div>
                      <div className="font-medium">
                        <span className="text-blue-700/80">Memory:</span> {fmt(s.memory_gib)} GiB
                      </div>
                      <div className="font-medium">
                        <span className="text-blue-700/80">Disk:</span> {fmt(s.storage_gib)} GiB
                      </div>
                    </div>
                  );
                })()}
              </div>
            ) : (
              <div className="text-blue-800">Select a type to see cost and specs.</div>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Button
              onClick={startGpu}
              loading={isLaunching}
              variant="primary"
              className="py-3 px-6 w-full"
              disabled={
                isLoadingOptions ||
                !regionName ||
                !instanceTypeName ||
                availableRegions.length === 0 ||
                availableInstanceTypes.length === 0
              }
            >
              {isLaunching ? "Launching..." : "Launch GPU Instance"}
            </Button>
            <Button
              onClick={terminateInstance}
              loading={isTerminating}
              variant="danger"
              className="py-3 px-6 w-full"
            >
              {isTerminating ? "Terminating..." : "Terminate Instance"}
            </Button>
          </div>
        </div>

        <div className="space-y-3 mt-1">
          {launchErr && <StatusMessage type="error">{launchErr}</StatusMessage>}
          {launchMsg && <StatusMessage type="success">{launchMsg}</StatusMessage>}
          {terminateErr && <StatusMessage type="error">{terminateErr}</StatusMessage>}
          {terminateMsg && <StatusMessage type="success">{terminateMsg}</StatusMessage>}
        </div>
      </div>

      {/* Script Execution Section */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <h3 className="text-lg font-medium text-gray-800 mb-4 flex items-center gap-2">
          <span className="text-lg">⚡</span>
          Execute Processing Script
        </h3>

        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Execute the DETR traffic analysis script on the uploaded video. Make sure you have
            uploaded a video file first.
          </p>

          <Button onClick={runScript} loading={isRunning} variant="accent" className="py-3 px-6">
            {isRunning ? "Processing..." : "Run Analysis Script"}
          </Button>

          <div className="space-y-3">
            {runErr && <StatusMessage type="error">{runErr}</StatusMessage>}
            {runMsg && <StatusMessage type="success">{runMsg}</StatusMessage>}
          </div>
        </div>
      </div>
    </Card>
  );
};

export default GPUControl;
