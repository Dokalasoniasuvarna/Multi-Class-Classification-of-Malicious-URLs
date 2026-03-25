"""
URL Malicious Classifier — Streamlit App (Dark Cyber Theme)
Run: streamlit run app.py
Required: best_lgbm_model.pkl, label_encoder.pkl, feature_columns.pkl
"""

import re
import math
import urllib.parse
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import tldextract
except ImportError:
    st.error("Run: pip install tldextract")
    st.stop()

st.set_page_config(page_title="URL Threat Detector", layout="centered")

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES (DARK CYBER THEME)
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:          #0B1120;      /* Deep Dark Blue */
    --surface:     #151D2E;      /* Card Background */
    --surface-2:   #1E293B;      /* Input Background */
    --border:      #334155;      /* Standard Border */
    --text:        #F1F5F9;      /* Primary Text */
    --muted:       #94A3B8;      /* Secondary Text */
    --accent:      #38BDF8;      /* Cyan Accent */
    
    --safe:        #22C55E;      /* Green */
    --safe-bg:     #052E16;
    --safe-brd:    #22C55E;

    --warn:        #F59E0B;      /* Amber */
    --warn-bg:     #422006;
    --warn-brd:    #F59E0B;

    --danger:      #EF4444;      /* Red */
    --danger-bg:   #450A0A;
    --danger-brd:  #EF4444;

    --purple:      #A855F7;      /* Purple */
    --purple-bg:   #2E1065;
    --purple-brd:  #A855F7;

    --font-mono:   'JetBrains Mono', monospace;
    --font-sans:   'Inter', sans-serif;
    --radius:      8px;
    --shadow:      0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: var(--font-sans);
    color: var(--text);
}

[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 800px;
    padding: 0 2rem 5rem;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Top Navigation Bar ── */
.top-bar {
    background: rgba(21, 29, 46, 0.8);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    margin: 0 -2rem 2rem;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.top-bar-left {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 12px var(--accent);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

.top-bar-title {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
}
.top-bar-status {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--safe);
    padding: 4px 10px;
    background: var(--safe-bg);
    border-radius: 12px;
    border: 1px solid var(--safe-brd);
}

/* ── Hero Section ── */
.hero-section {
    margin-bottom: 2.5rem;
}
.hero-tag {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--accent);
    background: rgba(56, 189, 248, 0.1);
    padding: 6px 12px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 1rem;
    border: 1px solid rgba(56, 189, 248, 0.2);
}
.hero-title {
    font-family: var(--font-sans);
    font-size: 3.5rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.1;
    margin: 0 0 1rem 0;
    letter-spacing: -0.03em;
}
.hero-title span {
    background: linear-gradient(135deg, var(--accent), #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc {
    font-size: 1rem;
    color: var(--muted);
    line-height: 1.6;
    max-width: 600px;
}

/* ── Input Section ── */
.input-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
}
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.section-icon {
    color: var(--accent);
    font-size: 1.2rem;
}
.section-title {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
}

/* ── Text Input Customization ── */
[data-testid="stTextInput"] input {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    transition: all 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: var(--muted) !important;
    opacity: 0.6;
}
[data-testid="stTextInput"] label { display: none !important; }

/* ── Button ── */
[data-testid="stButton"] button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent), #0EA5E9) !important;
    border: none !important;
    color: #0B1120 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 1rem 0 !important;
    border-radius: 6px !important;
    text-transform: uppercase;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25);
}
[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4) !important;
}
[data-testid="stButton"] button:active {
    transform: translateY(0);
}

/* ── Result Card ── */
.result-wrapper { margin-top: 2rem; }

.result-card {
    border-radius: 8px;
    padding: 2.5rem 2rem 2rem;
    border-left: 4px solid;
    background: var(--surface);
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

.card-safe        { border-left-color: var(--safe-brd); background: linear-gradient(90deg, var(--safe-bg), var(--surface)); }
.badge-safe       { color: var(--safe); background: rgba(34, 197, 94, 0.1); }
.label-safe       { color: var(--safe); }

.card-def         { border-left-color: var(--warn-brd); background: linear-gradient(90deg, var(--warn-bg), var(--surface)); }
.badge-def        { color: var(--warn); background: rgba(245, 158, 11, 0.1); }
.label-def        { color: var(--warn); }

.card-phi         { border-left-color: var(--danger-brd); background: linear-gradient(90deg, var(--danger-bg), var(--surface)); }
.badge-phi        { color: var(--danger); background: rgba(239, 68, 68, 0.1); }
.label-phi        { color: var(--danger); }

.card-mal         { border-left-color: var(--danger-brd); background: linear-gradient(90deg, var(--danger-bg), var(--surface)); }
.badge-mal        { color: var(--danger); background: rgba(239, 68, 68, 0.1); }
.label-mal        { color: var(--danger); }

.card-spam        { border-left-color: var(--purple-brd); background: linear-gradient(90deg, var(--purple-bg), var(--surface)); }
.badge-spam       { color: var(--purple); background: rgba(168, 85, 247, 0.1); }
.label-spam       { color: var(--purple); }

.result-badge {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 1rem;
    font-weight: 500;
}

.result-label {
    font-family: var(--font-sans);
    font-size: 2.8rem;
    font-weight: 600;
    line-height: 1.2;
    margin-bottom: 0.8rem;
}

.result-url {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--muted);
    background: var(--bg);
    padding: 0.8rem;
    border-radius: 4px;
    word-break: break-all;
    line-height: 1.5;
    border: 1px solid var(--border);
}

.result-note {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.6;
}

/* ── Footer ── */
.footer-bar {
    margin-top: 4rem;
    border-top: 1px solid var(--border);
    padding-top: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR (LOGIC UNCHANGED)
# ─────────────────────────────────────────────────────────────

def get_entropy(text):
    if not text or len(text) <= 1:
        return 0.0
    p   = Counter(text)
    lns = float(len(text))
    h   = -sum(count / lns * math.log2(count / lns) for count in p.values())
    return h / math.log2(len(text))


def extract_url_features(url):
    parsed    = urllib.parse.urlparse(url)
    ext       = tldextract.extract(url)
    domain    = ext.domain + '.' + ext.suffix
    path      = parsed.path
    query     = parsed.query
    has_query = len(query) > 0

    filename   = path.split('/')[-1] if '/' in path else ''
    directory  = path.rsplit('/', 1)[0] if '/' in path else ''
    extension  = filename.split('.')[-1] if '.' in filename else ''
    after_path = url.split(path)[-1] if path else ''

    domain_tokens = re.split(r'\.|\\-', ext.domain)
    path_tokens   = [t for t in re.split(r'\/|\-|\_|\.', path) if t]
    ldl_pattern   = r'[a-zA-Z][0-9][a-zA-Z]'

    f = {}
    f['Querylength']        = len(query)
    f['domain_token_count'] = len(domain_tokens)
    f['path_token_count']   = len(path_tokens)
    f['avgdomaintokenlen']  = sum(len(t) for t in domain_tokens) / len(domain_tokens) if domain_tokens else 0
    f['longdomaintokenlen'] = max([len(t) for t in domain_tokens]) if domain_tokens else 0
    f['avgpathtokenlen']    = sum(len(t) for t in path_tokens) / len(path_tokens) if path_tokens else 0
    f['tld']                = len(domain_tokens)
    f['charcompvowels']     = sum(1 for c in url if c.lower() in 'aeiou')
    f['charcompace']        = sum(1 for c in url if c.lower() in 'ace')
    f['ldl_url']            = len(re.findall(ldl_pattern, url))
    f['ldl_domain']         = len(re.findall(ldl_pattern, domain))
    f['ldl_path']           = len(re.findall(ldl_pattern, path))
    f['ldl_filename']       = len(re.findall(ldl_pattern, filename))
    f['ldl_getArg']         = len(re.findall(ldl_pattern, query))
    f['dld_url']            = len(re.findall(ldl_pattern, url))
    f['dld_domain']         = len(re.findall(ldl_pattern, domain))
    f['dld_path']           = len(re.findall(ldl_pattern, path))
    f['dld_filename']       = len(re.findall(ldl_pattern, filename))
    f['dld_getArg']         = len(re.findall(ldl_pattern, query))
    f['urlLen']             = len(url)
    f['domainlength']       = len(domain)
    f['pathLength']         = len(path)
    f['subDirLen']          = len(directory)
    f['fileNameLen']        = len(filename)
    f['this.fileExtLen']    = len(extension)

    if has_query:
        f['ArgLen']        = len(query)
        f['ArgUrlRatio']   = len(query) / len(url) if len(url) > 0 else 0
        f['argDomanRatio'] = len(query) / len(domain) if len(domain) > 0 else 0
        f['argPathRatio']  = len(query) / len(path)   if len(path)   > 0 else 0
    else:
        f['ArgLen']        = 2
        f['ArgUrlRatio']   = 2 / len(url) if len(url) > 0 else 0
        f['argDomanRatio'] = 0
        f['argPathRatio']  = 0

    f['pathurlRatio']    = len(path)   / len(url)    if len(url)    > 0 else 0
    f['domainUrlRatio']  = len(domain) / len(url)    if len(url)    > 0 else 0
    f['pathDomainRatio'] = len(path)   / len(domain) if len(domain) > 0 else 0

    f['executable']    = 1 if any(url.endswith(e) for e in ['.exe','.dll','.sh','.bat','.cmd']) else 0
    f['isPortEighty']  = 1 if parsed.port == 80 else (-1 if parsed.port is None else 0)
    f['NumberofDotsinURL']       = url.count('.')
    f['ISIpAddressInDomainName'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ext.domain) else -1
    f['CharacterContinuityRate'] = 0
    f['LongestVariableValue']    = -1 if not has_query else (
        max([len(v) for vals in urllib.parse.parse_qs(query).values() for v in vals], default=0)
    )

    f['URL_DigitCount']       = sum(c.isdigit() for c in url)
    f['host_DigitCount']      = sum(c.isdigit() for c in domain)
    f['Directory_DigitCount'] = sum(c.isdigit() for c in directory)
    f['File_name_DigitCount'] = sum(c.isdigit() for c in filename)
    f['Extension_DigitCount'] = sum(c.isdigit() for c in extension)
    f['Query_DigitCount']     = sum(c.isdigit() for c in query) if has_query else -1

    f['URL_Letter_Count']      = sum(c.isalpha() for c in url)
    f['host_letter_count']     = sum(c.isalpha() for c in domain)
    f['Directory_LetterCount'] = sum(c.isalpha() for c in directory)
    f['Filename_LetterCount']  = sum(c.isalpha() for c in filename)
    f['Extension_LetterCount'] = sum(c.isalpha() for c in extension)
    f['Query_LetterCount']     = sum(c.isalpha() for c in query) if has_query else -1

    f['LongestPathTokenLength']          = max([len(t) for t in path_tokens], default=0)
    f['Domain_LongestWordLength']        = max([len(t) for t in domain_tokens], default=0)
    f['Path_LongestWordLength']          = max([len(t) for t in path_tokens], default=0)
    f['sub-Directory_LongestWordLength'] = 0
    f['Arguments_LongestWordLength']     = -1 if not has_query else 0

    f['URL_sensitiveWord'] = 1 if any(w in url.lower() for w in [
        'login','signin','verify','secure','banking','update','confirm',
        'password','paypal','ebay','amazon','apple','microsoft','admin',
        'support','security','wallet','payment','account','credential'
    ]) else 0
    f['URLQueries_variable'] = len(urllib.parse.parse_qs(query)) if has_query else 0

    f['spcharUrl']        = sum(1 for c in url    if not c.isalnum())
    f['delimeter_Domain'] = sum(1 for c in domain if not c.isalnum())
    f['delimeter_path']   = sum(1 for c in path   if not c.isalnum())
    f['delimeter_Count']  = -1 if not has_query else sum(1 for c in query if not c.isalnum())

    f['NumberRate_URL']           = f['URL_DigitCount'] / len(url)        if len(url)        > 0 else 0
    f['NumberRate_Domain']        = f['host_DigitCount'] / len(domain)    if len(domain)     > 0 else 0
    f['NumberRate_DirectoryName'] = sum(c.isdigit() for c in directory) / len(directory) if len(directory) > 0 else 0
    f['NumberRate_FileName']      = sum(c.isdigit() for c in filename)  / len(filename)  if len(filename)  > 0 else 0
    f['NumberRate_Extension']     = sum(c.isdigit() for c in extension) / len(extension) if len(extension) > 0 else 0
    f['NumberRate_AfterPath']     = sum(c.isdigit() for c in after_path) / len(after_path) if len(after_path) > 0 else -1

    f['SymbolCount_URL']           = sum(1 for c in url       if not c.isalnum())
    f['SymbolCount_Domain']        = sum(1 for c in domain    if not c.isalnum())
    f['SymbolCount_Directoryname'] = sum(1 for c in directory if not c.isalnum())
    f['SymbolCount_FileName']      = sum(1 for c in filename  if not c.isalnum())
    f['SymbolCount_Extension']     = sum(1 for c in extension if not c.isalnum())
    f['SymbolCount_Afterpath']     = sum(1 for c in after_path if not c.isalnum()) if after_path else -1

    f['Entropy_URL']           = get_entropy(url)
    f['Entropy_Domain']        = get_entropy(domain)
    f['Entropy_DirectoryName'] = get_entropy(directory)
    f['Entropy_Filename']      = get_entropy(filename)
    f['Entropy_Extension']     = get_entropy(extension)
    f['Entropy_Afterpath']     = get_entropy(after_path) if has_query else -1.0

    return f


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

def load_model():
    model         = joblib.load('best_lgbm_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    feature_order = joblib.load('feature_columns.pkl')
    return model, label_encoder, feature_order


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

def predict(url, model, le, feature_order):
    raw = extract_url_features(url)
    df  = pd.DataFrame([raw])
    df  = df.reindex(columns=feature_order, fill_value=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    pred_enc   = model.predict(df)[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    return pred_label


# ─────────────────────────────────────────────────────────────
# LABEL CONFIG
# ─────────────────────────────────────────────────────────────

LABEL_CFG = {
    'benign': {
        'card': 'card-safe', 'badge': 'badge-safe',
        'label': 'label-safe', 
        'badge_txt': 'SECURE',
        'display': 'Safe URL',
        'note_txt': 'No threats detected. This link has been analyzed and appears legitimate.'
    },
    'Defacement': {
        'card': 'card-def', 'badge': 'badge-def',
        'label': 'label-def',
        'badge_txt': 'WARNING',
        'display': 'Defacement',
        'note_txt': 'Alert: This URL may link to a defaced or compromised website.'
    },
    'phishing': {
        'card': 'card-phi', 'badge': 'badge-phi',
        'label': 'label-phi',
        'badge_txt': 'DANGER',
        'display': 'Phishing',
        'note_txt': 'Critical Risk: This URL exhibits phishing characteristics. Do not enter credentials.'
    },
    'malware': {
        'card': 'card-mal', 'badge': 'badge-mal',
        'label': 'label-mal',
        'badge_txt': 'CRITICAL',
        'display': 'Malware',
        'note_txt': 'High Risk: Associated with malware distribution. Do not visit this URL.'
    },
    'spam': {
        'card': 'card-spam', 'badge': 'badge-spam',
        'label': 'label-spam',
        'badge_txt': 'SPAM',
        'display': 'Spam',
        'note_txt': 'Caution: Flagged as spam. May contain unwanted or deceptive content.'
    },
}

FALLBACK_CFG = {
    'card': 'card-def', 'badge': 'badge-def',
    'label': 'label-def',
    'badge_txt': 'UNKNOWN', 'display': 'Unknown',
    'note_txt': 'Unable to classify definitively.'
}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    # Top Navigation Bar
    st.markdown("""
    <div class="top-bar">
        <div class="top-bar-left">
            <div class="status-dot"></div>
            <div class="top-bar-title">CYBER SHIELD</div>
        </div>
        <div class="top-bar-status">SYSTEM ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-tag">Threat Intelligence Engine</div>
        <div class="hero-title">Malicious URL<br><span>Detector</span></div>
        <div class="hero-desc">
            Analyze any URL for malicious intent. Our system extracts lexical and structural features to identify phishing, malware, and spam in real-time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    try:
        model, le, feature_order = load_model()
    except FileNotFoundError:
        st.error("Model files not found — place best_lgbm_model.pkl, label_encoder.pkl, feature_columns.pkl in this directory.")
        st.stop()

    # Input Section
    st.markdown("""
    <div class="input-section">
        <div class="section-header">
            <div class="section-title">Target URL</div>
        </div>
    """, unsafe_allow_html=True)

    url_input = st.text_input(
        "url", label_visibility="collapsed",
        placeholder="https://example-site.com/suspicious-path"
    )

    analyse = st.button("Scan URL", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if analyse:
        url = url_input.strip()
        if not url:
            st.warning("Please enter a URL to scan.")
            return
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        with st.spinner("Processing Analysis..."):
            try:
                pred_label = predict(url, model, le, feature_order)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        cfg = LABEL_CFG.get(pred_label, FALLBACK_CFG)

        st.markdown(f"""
        <div class="result-wrapper">
            <div class="result-card {cfg['card']}">
                <div class="result-badge {cfg['badge']}">{cfg['badge_txt']}</div>
                <div class="result-label {cfg['label']}">{cfg['display']}</div>
                <div class="result-url">{url}</div>
                <div class="result-note">{cfg['note_txt']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-bar">
        <span>Security Research Tool</span>
        <span>Version 2.0 — Cyber Theme</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()