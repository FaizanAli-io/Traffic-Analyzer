import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

const API_BASE = import.meta.env.VITE_BACKEND_SERVER;

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);

  const navigate = useNavigate(); // ✅ hook for navigation

  const validateForm = () => {
    const newErrors = {};
    if (!username) newErrors.username = 'Username is required';
    if (!password) newErrors.password = 'Password is required';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;
    setIsLoading(true);
    setErrors({});

    try {
      const resp = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      const data = await resp.json().catch(() => ({}));

      if (resp.ok && data?.success) {
        onLogin?.(); // optional callback
        navigate('/dashboard'); // ✅ redirect after successful login
      } else {
        setErrors({
          general: data?.error || `Login failed (status ${resp.status})`
        });
      }
    } catch (err) {
      setErrors({ general: `Network error: ${err.message || err}` });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSubmit(e);
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h1 className="login-title">GPU Video Processing</h1>

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="login-field">
            <label className="login-label">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={handleKeyDown}
              className="login-input"
              placeholder="Enter your username"
              autoComplete="username"
            />
            {errors.username && <p className="login-error">{errors.username}</p>}
          </div>

          <div className="login-field">
            <label className="login-label">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={handleKeyDown}
              className="login-input"
              placeholder="Enter your password"
              autoComplete="current-password"
            />
            {errors.password && <p className="login-error">{errors.password}</p>}
          </div>

          {errors.general && (
            <p className="login-error login-general-error">{errors.general}</p>
          )}

          <button type="submit" disabled={isLoading} className="login-button">
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default Login;
