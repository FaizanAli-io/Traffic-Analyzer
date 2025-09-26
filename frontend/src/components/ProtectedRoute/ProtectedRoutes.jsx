import { Navigate } from "react-router-dom";
import { useState, useEffect } from "react";

const ProtectedRoute = ({ children, isAuthenticated }) => {
  const [authState, setAuthState] = useState(null); // null = checking, true = authenticated, false = not authenticated

  useEffect(() => {
    const checkAuth = () => {
      // If isAuthenticated prop is provided, use it
      if (typeof isAuthenticated === "boolean") {
        setAuthState(isAuthenticated);
        return;
      }

      // Fallback to localStorage check
      try {
        const userSession = localStorage.getItem("userSession");
        if (!userSession) {
          setAuthState(false);
          return;
        }

        const sessionData = JSON.parse(userSession);
        const isValid = sessionData?.isLoggedIn === true;

        // Optional: Check if session is expired (if you add timestamp validation later)
        setAuthState(isValid);
      } catch (error) {
        console.error("Error checking authentication:", error);
        // Clear corrupted session data
        localStorage.removeItem("userSession");
        setAuthState(false);
      }
    };

    checkAuth();
  }, [isAuthenticated]);

  // Show loading spinner while checking authentication
  if (authState === null) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="flex flex-col items-center space-y-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="text-gray-600 text-sm">Checking authentication...</p>
        </div>
      </div>
    );
  }

  // If not authenticated, redirect to login
  if (!authState) {
    return <Navigate to="/login" replace />;
  }

  // If authenticated, render the protected component
  return children;
};

export default ProtectedRoute;
