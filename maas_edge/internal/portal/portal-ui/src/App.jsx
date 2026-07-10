import { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [pairCode, setPairCode] = useState("------");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    // In production this is served by Go daemon which hosts an API
    fetch('/api/pair-code')
      .then(res => res.json())
      .then(data => setPairCode(data.pair_code))
      .catch(err => console.error("Failed to fetch pair code:", err));
  }, []);

  const copyCode = () => {
    navigator.clipboard.writeText(pairCode).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="portal-container">
      <div className="orb orb-1"></div>
      <div class="orb orb-2"></div>

      <div className="glass-card">
        <div className="status-badge">
          <div className="pulse"></div>
          Awaiting Command Center Claim
        </div>
        
        <h1>Edge Node Ready</h1>
        <p>This physical edge device is securely broadcasting its beacon. Enter the 6-digit code below into your MAAS Command Center to establish a Zero-Trust connection.</p>
        
        <div className={`code-container ${copied ? 'copied' : ''}`} onClick={copyCode}>
          <div className="pair-code">{pairCode}</div>
          <div className="copy-hint">Copied to clipboard!</div>
        </div>

        <div className="footer">
          <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
          Zero-Trust Ephemeral Handshake Protocol
        </div>
      </div>
    </div>
  );
}

export default App;
