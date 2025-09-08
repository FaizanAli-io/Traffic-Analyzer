import React from 'react';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children, isAuthenticated }) => {
  // Use the prop first, then fallback to localStorage check
  const checkAuth = () => {
    // If isAuthenticated prop is provided, use it
    if (typeof isAuthenticated === 'boolean') {
      return isAuthenticated;
    }
    
    // Fallback to localStorage check
    try {
      const userSession = localStorage.getItem('userSession');
      if (!userSession) return false;
      
      const sessionData = JSON.parse(userSession);
      return sessionData?.isLoggedIn === true;
    } catch (error) {
      console.error('Error checking auth:', error);
      return false;
    }
  };

  const authenticated = checkAuth();

  // If not authenticated, redirect to login
  if (!authenticated) {
    return <Navigate to="/login" replace />;
  }

  // If authenticated, render the protected component
  return children;
};

export default ProtectedRoute;