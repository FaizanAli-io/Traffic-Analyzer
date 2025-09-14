import { Card, Button, StatusMessage } from "../../shared";
import { useGPUInstance } from "../../../hooks/useGPUInstance";

const GPUControl = ({ setIp }) => {
  const {
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
    startGpu,
    terminateInstance,
    runScript
  } = useGPUInstance(setIp);

  return (
    <Card>
      <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2 mb-6">
        <span className="text-xl">🖥️</span>
        GPU Instance Control
      </h2>

      <div className="space-y-6">
        <div className="text-sm text-gray-600 leading-relaxed">
          <p className="mb-2">
            Launch a Lambda Labs GPU instance for video processing. The instance will be
            automatically configured with the required dependencies.
          </p>
          {ip && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mt-3">
              <span className="font-medium text-blue-800">Active Instance IP: </span>
              <span className="font-mono text-blue-600">{ip}</span>
            </div>
          )}
        </div>

        <div className="flex flex-col sm:flex-row gap-3">
          <Button onClick={startGpu} loading={isLaunching} variant="primary" className="py-3 px-6">
            {isLaunching ? "Launching..." : "Launch GPU Instance"}
          </Button>
          <Button
            onClick={terminateInstance}
            loading={isTerminating}
            variant="danger"
            className="py-3 px-6"
          >
            {isTerminating ? "Terminating..." : "Terminate Instance"}
          </Button>
        </div>

        <div className="space-y-3">
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
