import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const initialDiagnostics = {
  model_loaded: false,
  dataset_loaded: false,
  device: "checking",
  latent_dim: 2,
  input_dim: 140,
  threshold: 0.037435,
};

function formatNumber(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toFixed(digits);
}

async function apiRequest(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }
  return payload;
}

function parseSignalInput(value) {
  const values = value
    .replace(/,/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map(Number);

  if (values.some((item) => Number.isNaN(item))) {
    throw new Error("Only numeric ECG values are accepted.");
  }
  if (values.length !== 140) {
    throw new Error(`Expected exactly 140 values, received ${values.length}.`);
  }
  return values;
}

function WaveformChart({ signal = [], reconstruction = [], stepErrors = [] }) {
  const width = 960;
  const height = 340;
  const padding = 34;

  const allValues = [...signal, ...reconstruction];
  const minValue = allValues.length ? Math.min(...allValues) : 0;
  const maxValue = allValues.length ? Math.max(...allValues) : 1;
  const range = Math.max(maxValue - minValue, 0.0001);

  const makePath = (values) =>
    values
      .map((value, index) => {
        const x = padding + (index / Math.max(values.length - 1, 1)) * (width - padding * 2);
        const y = height - padding - ((value - minValue) / range) * (height - padding * 2);
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(" ");

  const maxStepError = Math.max(...stepErrors, 0.0001);
  const bars = stepErrors.map((value, index) => {
    const barWidth = (width - padding * 2) / stepErrors.length;
    const x = padding + index * barWidth;
    const barHeight = (value / maxStepError) * 58;
    return { x, y: height - padding - barHeight, width: Math.max(barWidth - 1, 1), height: barHeight };
  });

  if (!signal.length) {
    return (
      <div className="empty-chart">
        <div className="empty-chart__line" />
        <span>Load a sample or upload a CSV file to inspect the waveform.</span>
      </div>
    );
  }

  return (
    <svg className="wave-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="ECG waveform reconstruction chart">
      <rect x="0" y="0" width={width} height={height} rx="8" className="chart-bg" />
      {[0, 1, 2, 3].map((item) => (
        <line
          key={item}
          x1={padding}
          x2={width - padding}
          y1={padding + item * 70}
          y2={padding + item * 70}
          className="grid-line"
        />
      ))}
      {bars.map((bar, index) => (
        <rect key={index} {...bar} className="error-bar" />
      ))}
      <path d={makePath(signal)} className="signal-line" />
      <path d={makePath(reconstruction)} className="reconstruction-line" />
    </svg>
  );
}

function App() {
  const [health, setHealth] = useState(initialDiagnostics);
  const [signalText, setSignalText] = useState("");
  const [loadedLabel, setLoadedLabel] = useState("No signal loaded");
  const [signal, setSignal] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    } catch {
      return "light";
    }
  });
  const fileRef = useRef(null);

  useEffect(() => {
    apiRequest("/api/health")
      .then(setHealth)
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // ignore storage failures in private mode
    }
  }, [theme]);

  const toggleTheme = () => setTheme((current) => (current === "dark" ? "light" : "dark"));

  const statusTone = result?.status || "pending";
  const pointCount = useMemo(() => signal.length || parseSignalCount(signalText), [signal.length, signalText]);

  async function loadSample(type) {
    setBusy(true);
    setError("");
    try {
      const payload = await apiRequest(`/api/sample/${type}`);
      setSignal(payload.signal);
      setSignalText(payload.preview);
      setLoadedLabel(`${payload.sample_type} sample #${payload.index}`);
      setResult(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadCsv(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const payload = await apiRequest("/api/upload", {
        method: "POST",
        body: form,
      });
      setSignal(payload.signal);
      setSignalText(payload.preview);
      setLoadedLabel(file.name);
      setResult(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function runAnalysis() {
    setBusy(true);
    setError("");
    try {
      const values = signal.length ? signal : parseSignalInput(signalText);
      const payload = await apiRequest("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ signal: values }),
      });
      setSignal(payload.signal);
      setSignalText(payload.signal.map((value) => value.toFixed(4)).join(", "));
      setResult(payload);
      setLoadedLabel("Analyzed ECG vector");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function handleSignalTextChange(event) {
    setSignalText(event.target.value);
    setSignal([]);
    setResult(null);
    setLoadedLabel("Manual ECG vector");
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Medical AI Screening Dashboard</p>
          <h1>ECG Anomaly Detection</h1>
          <p className="lede">
            VAE reconstruction analysis with a compact React interface served by FastAPI.
          </p>
        </div>
        <div className="topbar-actions">
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M12 4.5a7.5 7.5 0 1 0 7.5 7.5 7.51 7.51 0 0 0-7.5-7.5Zm0 13a5.5 5.5 0 1 1 5.5-5.5 5.507 5.507 0 0 1-5.5 5.5Z"
                  fill="currentColor"
                />
                <path
                  d="M12 2.5v2.5M12 19v2.5M4.5 12H2m20 0h-2.5M5.636 5.636l-1.768-1.768M20.132 20.132l-1.768-1.768M5.636 18.364l-1.768 1.768M20.132 3.868l-1.768 1.768"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M13 2.5c0 4.69-3.81 8.5-8.5 8.5 0 4.69 3.81 8.5 8.5 8.5 4.69 0 8.5-3.81 8.5-8.5S17.69 2.5 13 2.5Z"
                  fill="currentColor"
                />
              </svg>
            )}
          </button>
          <div className="status-strip" aria-label="system status">
            <span className={health.model_loaded ? "dot dot--ok" : "dot dot--bad"} />
            <div>
              <strong>{health.model_loaded ? "Model online" : "Model missing"}</strong>
              <small>{health.device} / latent dim {health.latent_dim}</small>
            </div>
          </div>
        </div>
      </section>

      <section className="workspace">
        <aside className="control-panel">
          <div className="panel-heading">
            <span>Data Controls</span>
            <strong>{pointCount}/140</strong>
          </div>

          <div className="button-grid">
            <button type="button" onClick={() => loadSample("normal")} disabled={busy}>
              Load Normal
            </button>
            <button type="button" className="danger-light" onClick={() => loadSample("anomaly")} disabled={busy}>
              Load Anomaly
            </button>
          </div>

          <label className="file-drop">
            <input ref={fileRef} type="file" accept=".csv" onChange={uploadCsv} disabled={busy} />
            <span>Upload CSV</span>
            <small>Exactly 140 numeric values</small>
          </label>

          <label className="input-label" htmlFor="signal-input">
            ECG vector
          </label>
          <textarea
            id="signal-input"
            value={signalText}
            onChange={handleSignalTextChange}
            placeholder="-0.1120, 0.2340, 0.8910, ..."
            spellCheck="false"
          />

          {error && <p className="error-message">{error}</p>}

          <button type="button" className="primary-action" onClick={runAnalysis} disabled={busy}>
            {busy ? "Processing..." : "Run Analysis"}
          </button>
        </aside>

        <section className="analysis-panel">
          <div className={`diagnosis diagnosis--${statusTone}`}>
            <div>
              <span>{statusTone === "pending" ? "Ready" : result.status === "normal" ? "Low Risk" : "Attention Required"}</span>
              <h2>{result?.title || "Awaiting ECG Signal"}</h2>
              <p>{result?.subtitle || "Load a sample or upload a CSV file to begin reconstruction analysis."}</p>
            </div>
            <div className="pulse-mark" aria-hidden="true" />
          </div>

          <div className="metrics-row">
            <Metric label="Loaded source" value={loadedLabel} />
            <Metric label="Reconstruction error" value={formatNumber(result?.reconstruction_error)} />
            <Metric label="Threshold" value={formatNumber(health.threshold)} />
            <Metric label="Severity index" value={result ? `${result.severity_index.toFixed(1)}%` : "N/A"} />
          </div>

          <div className="chart-card">
            <div className="chart-header">
              <div>
                <span>Waveform Review</span>
                <strong>Original vs Reconstruction</strong>
              </div>
              <div className="legend">
                <i className="legend-signal" /> Original
                <i className="legend-recon" /> Reconstruction
                <i className="legend-error" /> Error
              </div>
            </div>
            <WaveformChart
              signal={result?.signal || signal}
              reconstruction={result?.reconstruction || []}
              stepErrors={result?.step_errors || []}
            />
          </div>

          <div className="detail-row">
            <Metric label="Latent coordinates" value={result ? `[${result.latent_mu.join(", ")}]` : "N/A"} />
            <Metric label="Dataset" value={health.dataset_loaded ? "Processed ECG5000 loaded" : "Dataset unavailable"} />
          </div>
        </section>
      </section>
    </main>
  );
}

function parseSignalCount(value) {
  if (!value.trim()) return 0;
  return value.replace(/,/g, " ").split(/\s+/).filter(Boolean).length;
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
