import { Card, Button, StatusMessage } from "../../shared";
import { useInstances } from "../../../hooks/useInstances";

const InstancesTable = () => {
  const {
    instances,
    isLoadingInstances,
    instancesErr,
    terminatingIds,
    loadInstances,
    terminateSpecificInstance
  } = useInstances();

  const handleTerminate = async (instanceId) => {
    if (
      window.confirm(
        "Are you sure you want to terminate this instance? This action cannot be undone."
      )
    ) {
      await terminateSpecificInstance(instanceId);
    }
  };

  return (
    <Card>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
          <span className="text-xl">📊</span>
          Current Instances
        </h2>
        <Button
          onClick={loadInstances}
          loading={isLoadingInstances}
          variant="secondary"
          className="text-sm py-2 px-4"
        >
          {isLoadingInstances ? "Refreshing..." : "Refresh"}
        </Button>
      </div>

      {instancesErr && <StatusMessage type="error">{instancesErr}</StatusMessage>}

      {isLoadingInstances ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-8 h-8 border-4 border-emerald-200 border-t-emerald-600 rounded-full animate-spin"></div>
          <span className="ml-3 text-gray-600">Loading instances...</span>
        </div>
      ) : instances.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🚫</div>
          <p className="text-gray-500 text-lg">No active instances found</p>
          <p className="text-gray-400 text-sm mt-2">Launch a GPU instance to get started</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-2 font-medium text-gray-700">Instance ID</th>
                <th className="text-left py-3 px-2 font-medium text-gray-700">Name</th>
                <th className="text-left py-3 px-2 font-medium text-gray-700">Status</th>
                <th className="text-left py-3 px-2 font-medium text-gray-700">Type</th>
                <th className="text-left py-3 px-2 font-medium text-gray-700">Region</th>
                <th className="text-left py-3 px-2 font-medium text-gray-700">IP</th>
                <th className="text-left py-3 px-2 font-medium text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody>
              {instances.map((instance) => (
                <tr key={instance.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-2 font-mono text-xs text-gray-600">{instance.id}</td>
                  <td className="py-3 px-2">{instance.name || "N/A"}</td>
                  <td className="py-3 px-2">
                    <span
                      className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                        instance.status === "active"
                          ? "bg-green-100 text-green-800"
                          : instance.status === "booting"
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {instance.status}
                    </span>
                  </td>
                  <td className="py-3 px-2 font-mono text-xs">{instance.instance_type || "N/A"}</td>
                  <td className="py-3 px-2">{instance.region || "N/A"}</td>
                  <td className="py-3 px-2 font-mono text-xs">{instance.ip || "N/A"}</td>
                  <td className="py-3 px-2">
                    <Button
                      onClick={() => handleTerminate(instance.id)}
                      loading={terminatingIds.has(instance.id)}
                      variant="danger"
                      className="text-xs py-1 px-3"
                    >
                      {terminatingIds.has(instance.id) ? "..." : "Terminate"}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <p className="text-xs text-gray-600">
          <span className="font-medium">Note:</span> Terminating instances will stop all running
          processes. Make sure to download your results before terminating.
        </p>
      </div>
    </Card>
  );
};

export default InstancesTable;
