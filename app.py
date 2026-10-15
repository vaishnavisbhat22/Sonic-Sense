
import streamlit as st
import io
import wave
import math
import numpy as np
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SonicSense | Speech Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080B12;
    --panel: #101622;
    --panel2: #141D2C;
    --border: #253148;
    --text: #F1F5FF;
    --muted: #8E9BB2;
    --accent: #8B7CFF;
    --cyan: #5CE1E6;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, #24204A 0%, transparent 35%),
        radial-gradient(ellipse at 95% 15%, #10313A 0%, transparent 28%),
        var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

p, label, li {
    color: #B4BED0;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}

.brand span {
    color: var(--cyan);
}

.eyebrow {
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-size: 0.72rem;
    font-weight: 700;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4.2rem);
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -2.5px;
    margin: 12px 0;
    color: #FFFFFF;
}

.gradient-text {
    background: linear-gradient(100deg, #A99DFF, #5CE1E6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-copy {
    max-width: 660px;
    line-height: 1.8;
    font-size: 1.02rem;
    color: #A8B4C8;
}

.panel {
    background: rgba(16, 22, 34, 0.94);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 16px;
}

.panel-title {
    color: #F4F6FF;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.15rem;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 0.88rem;
    margin-top: 5px;
    line-height: 1.6;
}

.metric-card {
    background: linear-gradient(145deg, #141D2C, #0D121D);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: var(--muted);
    font-size: 0.8rem;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 10px;
}

.status-pill {
    display: inline-block;
    border: 1px solid #31585C;
    background: #102B30;
    color: #80E6DE;
    padding: 6px 11px;
    border-radius: 100px;
    font-size: 0.74rem;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(100deg, #8172F6, #5B8DF7);
    color: #FFFFFF;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    min-height: 46px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #B8B0FF;
    filter: brightness(1.12);
    color: #FFFFFF;
}

.stButton > button:disabled {
    background: #272D3C;
    color: #7C879A;
}

[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #485775;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stAudioInput"] {
    background: #111827;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: #A8B4C8;
    font-weight: 600;
    padding: 12px 16px;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: var(--border);
}

.small-muted {
    color: #8290A8;
    font-size: 0.8rem;
    line-height: 1.6;
}

.footer {
    text-align: center;
    color: #66738A;
    font-size: 0.8rem;
    padding: 25px 0 5px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
left, right = st.columns([3, 1])

with left:
    st.markdown(
        '<div class="brand">SONIC<span>SENSE</span> '
        '<span style="color:#8491A8;font-size:0.7rem;'
        'letter-spacing:1px;">/ SPEECH INTELLIGENCE</span></div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div style="text-align:right;padding-top:5px;">'
        '<span class="status-pill">● LOCAL WORKSPACE</span></div>',
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(
    '<div class="eyebrow">AUDIO AUTHENTICITY LAB</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">Hear beyond the <br>'
    '<span class="gradient-text">surface.</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-copy">'
    'Analyze speech recordings to distinguish genuine human speech '
    'from AI-generated or synthetic voices. Submit a recording, '
    'prepare it for analysis, and inspect the detection results.'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")
st.write("")

# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">INPUT SOURCES</div>
        <div class="metric-value">Record + Upload</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">DETECTION TASK</div>
        <div class="metric-value">Real vs Synthetic</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ANALYSIS ENGINE</div>
        <div class="metric-value">Integration Pending</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# --------------------------------------------------
# AUDIO WORKSPACE
# --------------------------------------------------
col_input, col_results = st.columns([1.08, 0.92], gap="large")

with col_input:
    st.markdown("""
    <div class="panel">
        <div class="eyebrow">01 / AUDIO INPUT</div>
        <div class="panel-title" style="margin-top:10px;">
            Submit your audio
        </div>
        <div class="panel-subtitle">
            Record a voice sample or upload an existing audio file.
        </div>
    </div>
    """, unsafe_allow_html=True)

    record_tab, upload_tab = st.tabs(["🎙️  Record audio", "↑  Upload audio"])

    with record_tab:
        st.markdown("#### Record a voice sample")
        st.caption("Allow microphone access when your browser asks.")

        recorded_audio = st.audio_input(
            "Record your speech",
            key="sonicsense_recording",
        )

        if recorded_audio is not None:
            st.audio(recorded_audio)

    with upload_tab:
        st.markdown("#### Choose an audio file")
        uploaded_audio = st.file_uploader(
            "Drop your recording here",
            type=["wav", "flac", "mp3", "ogg", "m4a"],
            help="Select a supported audio recording from your device.",
            key="sonicsense_upload",
        )

        if uploaded_audio is not None:
            st.audio(uploaded_audio)

    # Choose whichever source is currently populated.
    # Recording takes priority if both are populated.
    audio_file = recorded_audio if recorded_audio is not None else uploaded_audio

    if audio_file is not None:
        st.success(f"Audio selected: {audio_file.name}")

        audio_bytes = audio_file.getvalue()

        # Estimate duration for uncompressed PCM WAV files.
        # Do not infer duration from file size for compressed formats.
        duration = None

        if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    if rate > 0:
                        duration = frames / rate
            except (wave.Error, EOFError):
                duration = None

        if duration is not None:
            st.caption(f"Duration: {duration:.1f} seconds")

        st.caption(f"File size: {len(audio_bytes) / (1024 * 1024):.2f} MB")

    st.write("")

    analyze_clicked = st.button(
        "Prepare audio for detection →",
        use_container_width=True,
        disabled=audio_file is None,
    )

    if analyze_clicked and audio_file is not None:
        # Save the selected audio in session state.
        # No classification is performed here.
        st.session_state["selected_audio_name"] = audio_file.name
        st.session_state["selected_audio_bytes"] = audio_file.getvalue()
        st.session_state["selected_audio_type"] = audio_file.type

        st.session_state["analysis_requested"] = True

    st.markdown(
        '<div class="small-muted">'
        'Supported formats: WAV, FLAC, MP3, OGG and M4A. '
        'Actual format compatibility depends on the installed audio '
        'processing libraries and the detection pipeline.'
        '</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# RESULTS PANEL
# --------------------------------------------------
with col_results:
    st.markdown("""
    <div class="panel">
        <div class="eyebrow">02 / DETECTION REPORT</div>
        <div class="panel-title" style="margin-top:10px;">
            Analysis results
        </div>
        <div class="panel-subtitle">
            Classification and confidence will appear here when
            the trained detection model is connected.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("analysis_requested", False):
        st.markdown("""
        <div class="panel" style="text-align:center;padding:34px 20px;">
            <div style="font-size:2.5rem;">◈</div>
            <div class="panel-title" style="margin-top:14px;">
                Audio received successfully
            </div>
            <div class="panel-subtitle" style="margin-top:10px;">
                Your recording is stored in this session and is ready
                for preprocessing. The trained classifier has not
                been connected yet, so no prediction is available.
            </div>
            <div style="margin-top:20px;">
                <span class="status-pill">AWAITING MODEL INTEGRATION</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Selected recording")
        st.write(st.session_state["selected_audio_name"])

        st.markdown("#### Pipeline status")

        stages = [
            ("01", "Audio submission", "Complete"),
            ("02", "Format validation", "Basic checks only"),
            ("03", "Feature extraction", "Pending"),
            ("04", "ML classification", "Pending"),
        ]

        for number, name, status in stages:
            a, b = st.columns([3, 2])
            a.markdown(f"**{number} · {name}**")
            b.caption(status)

        if st.button("Reset current analysis", use_container_width=True):
            for key in [
                "selected_audio_name",
                "selected_audio_bytes",
                "selected_audio_type",
                "analysis_requested",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

    else:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:45px 20px;">
            <div style="font-size:3rem;color:#8B7CFF;">〰</div>
            <div class="panel-title" style="margin-top:15px;">
                Your report starts here
            </div>
            <div class="panel-subtitle" style="margin-top:10px;">
                Submit an audio recording to prepare your analysis.
                Classification results will be available after the
                SonicSense ML engine is integrated.
            </div>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)

        with r1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">PREDICTION</div>
                <div class="metric-value">—</div>
                <div class="small-muted">Not analyzed</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">CONFIDENCE</div>
                <div class="metric-value">—</div>
                <div class="small-muted">Not available</div>
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.write("")
st.markdown("---")
st.markdown("""
<div class="footer">
    SONICSENSE · SPEECH AUTHENTICITY RESEARCH<br>
    <span style="font-size:0.72rem;">
        Research prototype · Results require a validated detection model
    </span>
</div>
""", unsafe_allow_html=True)
