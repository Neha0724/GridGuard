import { useEffect, useState } from "react";
import axios from "axios";
import { Activity, Bolt, CheckCircle2, CircleHelp, Gauge, ShieldAlert, ShieldCheck, Sparkles, Zap, Users, FileText } from "lucide-react";

const API = "http://127.0.0.1:8000/api";
const initial = {
  consumer_name: "",
  consumer_id: "",
  readings: "",
  consumption_mean: "1.8",
  consumption_std: "1.2",
  consumption_min: "0",
  consumption_max: "8.4",
  consumption_median: "1.1",
  consumption_iqr: "1.9",
  zero_day_ratio: "0.12",
  missing_day_ratio: "0.01",
  consumption_trend: "0.002"
};

const fields = [
  { key: "consumption_mean", label: "Average daily consumption", hint: "Mean daily reading", step: "any", min: "0", suffix: "units" },
  { key: "consumption_std", label: "Consumption variation", hint: "Standard deviation", step: "any", min: "0", suffix: "units" },
  { key: "consumption_min", label: "Minimum daily consumption", hint: "Lowest daily reading", step: "any", min: "0", suffix: "units" },
  { key: "consumption_max", label: "Maximum daily consumption", hint: "Highest daily reading", step: "any", min: "0", suffix: "units" },
  { key: "consumption_median", label: "Median daily consumption", hint: "Middle daily reading", step: "any", min: "0", suffix: "units" },
  { key: "consumption_iqr", label: "Consumption spread", hint: "Interquartile range", step: "any", min: "0", suffix: "units" },
  { key: "zero_day_ratio", label: "Zero-consumption ratio", hint: "Enter a value from 0 to 1", step: "0.01", min: "0", max: "1" },
  { key: "missing_day_ratio", label: "Missing-reading ratio", hint: "Enter a value from 0 to 1", step: "0.01", min: "0", max: "1" },
  { key: "consumption_trend", label: "Consumption trend", hint: "Change across the reading period", step: "any" }
];

export default function App() {
  const [values, setValues] = useState(initial);
  const [status, setStatus] = useState({ ready: false, message: "Checking model…" });
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [csvFile, setCsvFile] = useState(null);
  const [csvBusy, setCsvBusy] = useState(false);
  const [csvSummary, setCsvSummary] = useState(null);
  const [csvError, setCsvError] = useState("");

  useEffect(() => {
    axios.get(`${API}/model-status`)
      .then(({ data }) => setStatus(data))
      .catch(() => setStatus({ ready: false, message: "Start the Python backend to connect GridGuard." }));
  }, []);

  function update(key, value) {
    setValues((old) => ({ ...old, [key]: value }));
  }

  async function predict(e) {
    e.preventDefault();
    setBusy(true); setError(""); setResult(null);
    try {
      const readings = values.readings
        .split(/[;,\\s]+/)
        .map((value) => value.trim())
        .filter(Boolean)
        .map(Number);
      if (readings.length < 2 || readings.some((value) => !Number.isFinite(value))) {
        throw new Error("Enter at least two valid numbers separated by commas or new lines.");
      }
      const { data } = await axios.post(`${API}/predict-readings`, {
        consumer_name: values.consumer_name,
        consumer_id: values.consumer_id,
        readings
      });
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || "Prediction failed. Check the backend and input values.");
    } finally {
      setBusy(false);
    }
  }

  function resetForm() {
    setValues(initial);
    setResult(null);
    setError("");
  }

  async function predictCsv(e) {
    e.preventDefault();
    if (!csvFile) { setCsvError("Choose a CSV file first."); return; }
    setCsvBusy(true); setCsvError(""); setCsvSummary(null);
    try {
      const form = new FormData();
      form.append("file", csvFile);
      const { data } = await axios.post(`${API}/predict-csv`, form, { headers: { "Content-Type": "multipart/form-data" } });
      setCsvSummary(data);
    } catch (e) {
      setCsvError(e.response?.data?.detail || "CSV prediction failed. Check the file format and backend.");
    } finally { setCsvBusy(false); }
  }

  function downloadCsv() {
    if (!csvSummary?.rows?.length) return;
    const rows = csvSummary.rows;
    const headers = [...new Set(rows.flatMap((row) => Object.keys(row)))];
    const escape = (value) => {
      if (value === null || value === undefined) return "";
      const text = String(value);
      return /[\",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    const csv = [headers.map(escape).join(","), ...rows.map((row) => headers.map((key) => escape(row[key])).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" }));
    const link = document.createElement("a"); link.href = url; link.download = `gridguard_predictions_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><Bolt size={23} fill="currentColor" /></div>
        <div><div className="brand-name">gridguard<span>.</span></div><div className="brand-sub">ENERGY INTELLIGENCE</div></div>
      </div>
      <div className="nav-caption">WORKSPACE</div>
      <div className="nav-item active"><Activity size={19} /> Overview</div>
      <div className="nav-item"><Users size={19} /> Consumer screening</div>
      <div className="sidebar-bottom">
        <div className="sidebar-callout">
          <div className="callout-icon"><Sparkles size={17} /></div>
          <div className="callout-title">Insight, not assumption.</div>
          <p>The model reviews consumption patterns. Every flagged result still needs human verification.</p>
        </div>
        <div className="profile-row"><div className="avatar">GG</div><div><strong>GridGuard Lab</strong><span>Research workspace</span></div></div>
      </div>
    </aside>

    <main className="main-content">
      <header className="topbar">
        <div className="breadcrumb">Workspace <span>/</span> <strong>Consumer overview</strong></div>
        <div className="topbar-right"><span className={`status-dot ${status.ready ? "" : "offline"}`} /> <span>{status.ready ? "Model ready" : "Model awaiting data"}</span><div className="top-avatar">N</div></div>
      </header>

      <div className="page-content">
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-dot" /> ELECTRICITY ANALYTICS</div>
            <h1>See the signal.<br /><em>Protect the grid.</em></h1>
            <p>Enter one consumer’s details and consumption profile to get a quick, ML-powered screening result.</p>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => document.getElementById("consumer-form")?.scrollIntoView({ behavior: "smooth" })}><Zap size={17} /> Screen a consumer</button>
              <div className="hero-caption"><ShieldCheck size={16} /> Evidence-led risk screening</div>
            </div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="art-sun" />
            <div className="art-grid">{Array.from({ length: 36 }, (_, i) => <span key={i} style={{ opacity: 0.25 + ((i * 7) % 8) / 10 }} />)}</div>
            <div className="art-bolt"><Bolt size={46} strokeWidth={1.5} fill="currentColor" /></div>
            <div className="art-chip chip-top"><span className="mini-pulse" /> Background model <strong>{status.ready ? "Ready" : "Waiting"}</strong></div>
            <div className="art-chip chip-bottom"><Activity size={15} /><div><strong>Consumption</strong><span>Behaviour analytics</span></div></div>
          </div>
        </section>

        <section className="section-heading">
          <div><div className="eyebrow muted">CONSUMER WORKSPACE</div><h2>Prediction workspace</h2><p className="section-subtitle">Choose manual screening for one consumer or batch screening for a full CSV.</p></div>
          <div className="date-pill"><span className={`status-dot ${status.ready ? "" : "offline"}`} /> {status.ready ? "Prediction available" : "Training data required"}</div>
        </section>

        <div className="metrics-grid">
          <div className="metric-card"><div className="metric-icon teal"><Users size={18} /></div><div className="metric-label">Screening mode</div><div className="metric-value">Single record</div><div className="metric-note">One consumer per prediction</div></div>
          <div className="metric-card"><div className="metric-icon coral"><ShieldAlert size={18} /></div><div className="metric-label">Output class</div><div className="metric-value">0 or 1</div><div className="metric-note">Based on the trained model</div></div>
          <div className="metric-card"><div className="metric-icon gold"><Gauge size={18} /></div><div className="metric-label">Model status</div><div className="metric-value">{status.ready ? "Ready" : "Pending"}</div><div className="metric-note">{status.ready ? "Ready to classify a profile" : "Add SGCC training data"}</div></div>
          <div className="metric-card"><div className="metric-icon blue"><FileText size={18} /></div><div className="metric-label">Input features</div><div className="metric-value">9 signals</div><div className="metric-note">Derived from daily readings</div></div>
        </div>

        {error && <div className="error-banner"><ShieldAlert size={18} /><span>{error}</span><button onClick={() => setError("")}>×</button></div>}

        <div className="content-grid consumer-content">
          <section className="panel upload-panel" id="consumer-form">
            <div className="panel-heading"><div><div className="eyebrow muted">CONSUMER DETAILS</div><h3>Consumption profile</h3></div><span className="panel-icon"><FileText size={19} /></span></div>
            <p className="panel-description">Add the consumer’s ID and daily consumption readings. GridGuard calculates the model features automatically, so you do not need to enter statistics manually.</p>
            <form onSubmit={predict}>
              <div className="form-grid identity-grid">
                <label className="field"><span>Consumer name</span><input value={values.consumer_name} onChange={(e) => update("consumer_name", e.target.value)} placeholder="e.g. Rahul Sharma" /></label>
                <label className="field"><span>Consumer ID</span><input value={values.consumer_id} onChange={(e) => update("consumer_id", e.target.value)} placeholder="e.g. SGCC-1024" /></label>
              </div>
              <div className="form-section-title"><span>01</span><div><strong>Daily consumption readings</strong><small>Paste readings from one consumer. Separate values with commas or new lines.</small></div></div>
              <label className="field readings-field">
                <span>Consumption values (daily units)</span>
                <textarea
                  required
                  rows={5}
                  value={values.readings}
                  onChange={(e) => update("readings", e.target.value)}
                  placeholder={"Example: 1.2, 0.8, 1.5, 0, 1.9, 2.1"}
                />
                <small className="field-hint">The system calculates average, variation, zero-use ratio and trend automatically. Use real readings for a meaningful result.</small>
              </label>
              <div className="readings-meta"><Activity size={16} /><span>{values.readings.split(/[;,\\s]+/).filter(Boolean).length} readings entered</span><span>Minimum 2 readings</span></div>
              <div className="form-actions">
                <button className="secondary-button" type="button" onClick={resetForm}>Reset form</button>
                <button className="primary-button submit-button" disabled={!status.ready || busy} type="submit">{busy ? <span className="spinner" /> : <Zap size={17} />}{busy ? "Checking…" : "Predict class 0 / 1"}</button>
              </div>
              {!status.ready && <div className="model-notice"><CircleHelp size={17} /><span>The model is still preparing or could not load the training data. Wait a moment; if it stays offline, check the backend terminal for the error.</span></div>}
            </form>
          </section>

          <section className="panel chart-panel result-panel">
            <div className="panel-heading"><div><div className="eyebrow muted">PREDICTION OUTPUT</div><h3>Screening result</h3></div><span className="panel-icon coral-icon"><ShieldAlert size={19} /></span></div>
            <p className="panel-description">The model returns a class based on the consumer’s consumption profile.</p>
            {!result && !error && <div className="result-empty">
              <div className="result-empty-icon"><ShieldCheck size={31} /></div>
              <h4>Awaiting input</h4>
              <p>Fill in the form and select Predict. Your result will appear here without exposing the training process.</p>
              <div className="result-preview"><span>0</span><i>Likely safe</i><span className="or">or</span><span className="class-one">1</span><i>Potentially suspicious</i></div>
            </div>}
            {error && <div className="error-box"><ShieldAlert size={22} /><strong>Unable to predict</strong><p>{error}</p></div>}
            {result && <div className={`prediction-result ${result.prediction === 1 ? "risk" : "safe"}`}>
              <div className="result-status-icon">{result.prediction === 1 ? <ShieldAlert size={29} /> : <CheckCircle2 size={29} />}</div>
              <span className="result-kicker">PREDICTED CLASS</span>
              <div className="prediction-class">{result.prediction}</div>
              <h4>{result.label}</h4>
              <p>{result.prediction === 1 ? "This profile resembles patterns associated with flagged examples. Review the case before taking any action." : "This profile resembles patterns associated with the safe class. This is not a guarantee."}</p>
              <div className="risk-meter-head"><span>Model score for class 1</span><strong>{result.risk_score}%</strong></div>
              <div className="risk-meter"><span style={{ width: `${result.risk_score}%` }} /></div>
              {(values.consumer_name || values.consumer_id) && <div className="consumer-summary">{values.consumer_name && <span>{values.consumer_name}</span>}{values.consumer_id && <small>ID: {values.consumer_id}</small>}</div>}
              <div className="result-footnote">{result.confidence_note}</div>
            </div>}
            <div className="class-legend">
              <div><span className="class-chip safe-chip">0</span><div><strong>Likely safe</strong><small>Predicted normal class</small></div></div>
              <div><span className="class-chip risk-chip">1</span><div><strong>Potentially suspicious</strong><small>Predicted flagged class</small></div></div>
            </div>
            <div className="chart-footnote"><CircleHelp size={15} /> The result is a screening signal, not proof of theft.</div>
          </section>
        </div>

        <section className="panel csv-panel" id="csv-upload">
          <div className="panel-heading"><div><div className="eyebrow muted">BATCH SCREENING</div><h3>Upload a CSV and export predictions</h3></div><span className="panel-icon"><FileText size={19} /></span></div>
          <p className="panel-description">Upload a consumer-level SGCC CSV with daily reading columns (for example, date columns such as 1/1/2014). The model calculates the features in the background and appends <code>predicted_class</code>, <code>prediction_label</code>, and <code>risk_score_percent</code> to each row.</p>
          <form className="csv-upload-row" onSubmit={predictCsv}>
            <label className="csv-dropzone"><FileText size={22} /><span><strong>{csvFile ? csvFile.name : "Choose consumer CSV"}</strong><small>{csvFile ? `${(csvFile.size / 1024).toFixed(1)} KB selected` : "CSV format · one consumer per row"}</small></span><input type="file" accept=".csv,text/csv" onChange={(e) => { setCsvFile(e.target.files?.[0] || null); setCsvSummary(null); setCsvError(""); }} /></label>
            <button className="primary-button" disabled={!status.ready || !csvFile || csvBusy} type="submit">{csvBusy ? <span className="spinner" /> : <Zap size={17} />}{csvBusy ? "Predicting rows…" : "Predict CSV"}</button>
          </form>
          {csvError && <div className="error-banner"><ShieldAlert size={18} /><span>{csvError}</span></div>}
          {csvSummary && <div className="csv-summary"><div><strong>{csvSummary.count.toLocaleString()}</strong><small>Rows processed</small></div><div><strong>{csvSummary.class_0.toLocaleString()}</strong><small>Class 0 · likely safe</small></div><div><strong>{csvSummary.class_1.toLocaleString()}</strong><small>Class 1 · potentially suspicious</small></div><button className="secondary-button download-button" type="button" onClick={downloadCsv}><FileText size={16} /> Download results CSV</button></div>}
          {!status.ready && <div className="model-notice"><CircleHelp size={17} /><span>Batch screening becomes available when the model is ready. If this message remains, check the backend terminal.</span></div>}
        </section>

        <div className="disclaimer"><ShieldAlert size={18} /><p><strong>Decision-support only.</strong> A class-1 prediction is not proof of electricity theft. Verify the dataset label meaning and review evidence before making operational decisions.</p></div>
        <footer className="footer"><div className="footer-brand"><Bolt size={16} fill="currentColor" /> gridguard<span>.</span></div><span>Electricity theft detection · ML research prototype</span><span className="footer-status"><span className={`status-dot ${status.ready ? "" : "offline"}`} /> {status.ready ? "Model online" : "Awaiting training data"}</span></footer>
      </div>
    </main>
  </div>;
}
