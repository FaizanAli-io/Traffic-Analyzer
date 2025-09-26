import { Card, Button, StatusMessage } from "../../shared";
import { useDirectionMapping } from "../../../hooks/useDirectionMapping";

const DirectionMapping = () => {
  const {
    directionMapping,
    isRotating,
    directionMsg,
    directionErr,
    rotateDirections,
    rotateDirectionsCounterClockwise
  } = useDirectionMapping();

  return (
    <Card>
      <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2 mb-6">
        <span className="text-xl">🧭</span>
        Direction Mapping Control
      </h2>

      <div className="space-y-6">
        {/* Original Compass Display */}
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
          <div className="flex flex-col items-center gap-6">
            {/* Compass Display */}
            <div className="relative w-40 h-40 border-4 border-gray-300 rounded-full bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center shadow-inner">
              <div className="absolute top-2 left-1/2 transform -translate-x-1/2 bg-white border border-gray-300 px-2 py-1 rounded text-xs font-semibold text-red-600 shadow-sm">
                {directionMapping.top}
              </div>
              <div className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-white border border-gray-300 px-2 py-1 rounded text-xs font-semibold text-orange-600 shadow-sm">
                {directionMapping.right}
              </div>
              <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 bg-white border border-gray-300 px-2 py-1 rounded text-xs font-semibold text-green-600 shadow-sm">
                {directionMapping.bottom}
              </div>
              <div className="absolute left-2 top-1/2 transform -translate-y-1/2 bg-white border border-gray-300 px-2 py-1 rounded text-xs font-semibold text-blue-600 shadow-sm">
                {directionMapping.left}
              </div>
            </div>

            {/* Direction Control Buttons */}
            <div className="flex flex-wrap gap-3 justify-center">
              <Button
                onClick={rotateDirections}
                loading={isRotating}
                variant="secondary"
                className="text-sm"
              >
                Rotate CW ↻
              </Button>
              <Button
                onClick={rotateDirectionsCounterClockwise}
                loading={isRotating}
                variant="accent"
                className="text-sm"
              >
                Rotate CCW ↺
              </Button>
            </div>
            <div className="text-center text-sm text-gray-600">
              <p>Adjust directional mapping for accurate traffic analysis</p>
              <small className="text-xs text-gray-500">
                Changes affect vehicle direction detection
              </small>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {directionErr && <StatusMessage type="error">{directionErr}</StatusMessage>}
          {directionMsg && <StatusMessage type="success">{directionMsg}</StatusMessage>}
        </div>
      </div>
    </Card>
  );
};

export default DirectionMapping;
