import { useState } from "react";
import { Button } from "../shared";
import { useNavigate } from "react-router-dom";
import GPUControl from "./sections/GPUControl";
import VideoUpload from "./sections/VideoUpload";
import InstancesTable from "./sections/InstancesTable";
import DownloadResults from "./sections/DownloadResults";
import DirectionMapping from "./sections/DirectionMapping";

const Dashboard = () => {
  const navigate = useNavigate();
  const [ip, setIp] = useState("");

  const handleLogout = () => {
    localStorage.removeItem("userSession");
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white/90 backdrop-blur-sm shadow-lg border-b border-white/20 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-blue-600 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">🚦</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                  Traffic Analyzer
                </h1>
                <p className="text-sm text-gray-500">Cloud GPU Processing Dashboard</p>
              </div>
            </div>

            <Button onClick={handleLogout} variant="secondary" className="text-sm">
              Logout
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-8">
            <GPUControl ip={ip} setIp={setIp} />
            <VideoUpload ip={ip} />
            <DownloadResults ip={ip} />
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            <InstancesTable />
            <DirectionMapping />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
