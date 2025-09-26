const StatusMessage = ({ type, children }) => {
  const variants = {
    success: "bg-emerald-50 border-emerald-200 text-emerald-700",
    error: "bg-red-50 border-red-200 text-red-600",
    info: "bg-indigo-50 border-indigo-200 text-indigo-700"
  };

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️"
  };

  return (
    <div
      className={`px-4 py-3 rounded-lg border text-sm font-medium ${variants[type]} flex items-start gap-2`}
    >
      <span className="text-sm">{icons[type]}</span>
      <span>{children}</span>
    </div>
  );
};

export default StatusMessage;
