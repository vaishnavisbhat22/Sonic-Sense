import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  Activity, AudioWaveform, CheckCircle2, ChevronRight, Clock3, FileAudio,
  FileCheck2, Gauge, History, LayoutDashboard, LoaderCircle, Menu,
  Microscope, ShieldCheck, Sparkles, UploadCloud, X, AlertTriangle
} from "lucide-react";
import { analyzeAudio } from "./api";
import { encodePcmToWav } from "./audioRecording";
import type { Prediction } from "./types";

type Tab = "overview" | "analyze" | "history" | "model";
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

const navItems: { id: Tab; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "analyze", label: "Analyze audio", icon: AudioWaveform },
  { id: "history", label: "Analysis history", icon: History },
  { id: "model", label: "Model details", icon: Microscope }
];

function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accepted = useMemo(() => ".wav,.flac,.mp3,.ogg,.m4a", []);

  function chooseFile(next?: File) {
    setError("");
    setResult(null);
    if (!next) return;
    const allowed = ["audio/wav", "audio/x-wav", "audio/flac", "audio/mpeg",
      "audio/ogg", "audio/mp4", "audio/x-m4a", ""];
    const extensionOk = /\.(wav|flac|mp3|ogg|m4a)$/i.test(next.name);
    if (!extensionOk || (next.type && !allowed.includes(next.type))) {
      setFile(null);
      setError("Unsupported file type. Upload WAV, FLAC, MP3, OGG or M4A audio.");
      return;
    }
    if (next.size > MAX_UPLOAD_BYTES) {
      setFile(null);
      setError("File exceeds the 25 MiB limit. Choose a smaller audio file.");
      return;
    }
    setFile(next);
  }

  async function runAnalysis() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const prediction = await analyzeAudio(file);
      setResult(prediction);
      setHistory(current => [prediction, ...current]);
    } catch (error: unknown) {
      const detail: unknown = axios.isAxiosError(error)
        ? error.response?.data?.detail
        : undefined;
      if (typeof detail === "string") {
        setError(detail);
      } else if (Array.isArray(detail)) {
        const messages = detail
          .map(item => (
            typeof item === "object" && item !== null && "msg" in item &&
            typeof item.msg === "string"
              ? item.msg
              : ""
          ))
          .filter(Boolean);
        setError(messages.join("; ") || "The backend rejected the uploaded recording.");
      } else if (error instanceof Error) {
        setError(error.message);
      } else {
        setError("Could not analyze this recording. Check that the backend is running.");
      }
    } finally {
      setLoading(false);
    }
  }

  const activeTitle = navItems.find(item => item.id === tab)?.label || "Overview";

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><AudioWaveform size={23} /></div>
          <div><div className="brand-name">sonic<span>sense</span></div><div className="brand-caption">AUDIO INTELLIGENCE</div></div>
          <button className="icon-button close-nav" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={18}/></button>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav>
          {navItems.map(item => {
            const Icon = item.icon;
            return <button key={item.id} className={`nav-item ${tab === item.id ? "active" : ""}`}
              onClick={() => { setTab(item.id); setMobileNav(false); }}>
              <Icon size={18}/><span>{item.label}</span>{tab === item.id && <ChevronRight size={15} className="nav-chevron"/>}
            </button>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="status-card">
            <div className="status-line"><span className="status-dot"/><span>Detection workspace</span></div>
            <p>Connect the API to run live audio analysis.</p>
          </div>
          <div className="sidebar-footer"><ShieldCheck size={15}/> SonicSense Research Project</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={20}/></button>
          <div className="breadcrumb">Workspace <span>/</span> <strong>{activeTitle}</strong></div>
          <div className="topbar-right"><span className="environment"><span className="status-dot"/> LOCAL ENVIRONMENT</span><div className="avatar">SS</div></div>
        </header>

        <div className="page-content">
          {tab === "overview" && <Overview onAnalyze={() => setTab("analyze")} history={history}/>}
          {tab === "analyze" && <Analyze file={file} setFile={chooseFile} setError={setError} inputRef={inputRef} accepted={accepted}
            result={result} loading={loading} error={error} onRun={runAnalysis} onClear={() => {setFile(null);setResult(null);setError("");}}/>}
          {tab === "history" && <HistoryPage history={history} onAnalyze={() => setTab("analyze")}/>}
          {tab === "model" && <ModelPage/>}
        </div>
        <footer className="page-footer"><span>SONICSENSE <b>•</b> SYNTHETIC SPEECH DETECTION</span><span>Research prototype · Results depend on input quality and dataset coverage</span></footer>
      </main>
    </div>
  );
}

function PageHeading({eyebrow, title, description, action}: {eyebrow:string;title:string;description:string;action?:React.ReactNode}) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

function Overview({onAnalyze, history}: {onAnalyze:()=>void;history:Prediction[]}) {
  return <>
    <PageHeading eyebrow="OVERVIEW" title="Audio intelligence, at a glance." description="Detect whether a recording contains natural human speech or AI-generated synthetic speech."
      action={<button className="primary-button" onClick={onAnalyze}><UploadCloud size={17}/> Analyze audio</button>}/>
    <section className="hero-panel">
      <div className="hero-copy"><div className="hero-pill"><Sparkles size={13}/> SYNTHETIC SPEECH DETECTION</div>
        <h2>Hear the difference.<br/><span>Verify the voice.</span></h2>
        <p>Analyze an audio recording with the SonicSense machine learning pipeline and review its classification result.</p>
        <button className="hero-button" onClick={onAnalyze}>Start an analysis <ChevronRight size={17}/></button>
      </div>
      <div className="wave-art" aria-hidden="true"><div className="wave-orb"><AudioWaveform size={86} strokeWidth={1}/></div><div className="wave-ring ring-one"/><div className="wave-ring ring-two"/><div className="wave-caption">SIGNAL ANALYSIS <span>●</span> READY</div></div>
    </section>
    <div className="section-title"><div><h3>Model snapshot</h3><p>Held-out test-set evaluation from the current training run.</p></div><span className="tag">XGBOOST MODEL</span></div>
    <div className="metric-grid">
      <Metric icon={Gauge} label="Test accuracy" value="95.96%" note="7,110 test samples"/>
      <Metric icon={Activity} label="Macro F1-score" value="94.44%" note="Balanced class-level F1"/>
      <Metric icon={CheckCircle2} label="Human recall" value="96.57%" note="Test-set human samples"/>
      <Metric icon={Microscope} label="ROC-AUC" value="99.30%" note="Held-out test set"/>
    </div>
    <div className="bottom-grid">
      <div className="content-card"><div className="card-heading"><div><h3>Detection workflow</h3><p>From recording to prediction</p></div><Activity size={18}/></div>
        <div className="workflow">
          {["Audio upload","Preprocessing","Feature extraction","XGBoost prediction"].map((step,i)=><div className="workflow-step" key={step}><div className="step-number">{String(i+1).padStart(2,"0")}</div><div><strong>{step}</strong><small>{["Select a supported recording","Decode, convert to mono and resample to 16 kHz","Generate the model's 134 input features","Return human or synthetic label"][i]}</small></div>{i<3&&<div className="step-connector"/>}</div>)}
        </div>
      </div>
      <div className="content-card"><div className="card-heading"><div><h3>Recent analyses</h3><p>Current browser session</p></div><Clock3 size={18}/></div>
        {history.length ? history.slice(0,4).map((item,i)=><div className="history-row" key={`${item.filename}-${i}`}><FileAudio size={17}/><div className="history-file"><strong>{item.filename}</strong><small>{new Date(item.analyzed_at).toLocaleString()}</small></div><ResultBadge label={item.label}/></div>) : <div className="empty-state compact"><History size={25}/><strong>No analyses yet</strong><span>Your completed audio analyses will appear here.</span></div>}
      </div>
    </div>
  </>;
}

function Metric({icon:Icon,label,value,note}:{icon:typeof Gauge;label:string;value:string;note:string}) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><Icon size={18}/></div><div className="metric-value">{value}</div><div className="metric-note">{note}</div></div>;
}

function Analyze({file,setFile,setError,inputRef,accepted,result,loading,error,onRun,onClear}:{file:File|null;setFile:(f?:File)=>void;setError:(message:string)=>void;inputRef:React.RefObject<HTMLInputElement>;accepted:string;result:Prediction|null;loading:boolean;error:string;onRun:()=>void;onClear:()=>void}) {
  const [dragging,setDragging] = useState(false);
  const [recording, setRecording] = useState(false);
  const [requestingMicrophone, setRequestingMicrophone] = useState(false);
  const [finalizingRecording, setFinalizingRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingPreviewUrl, setRecordingPreviewUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mutedOutputRef = useRef<GainNode | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const pcmFrameCountRef = useRef(0);
  const recordingTooLargeRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      stopAudioCapture();
      mediaStreamRef.current?.getTracks().forEach(track => track.stop());
      if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  function clearRecordingPreview() {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setRecordingPreviewUrl(null);
  }

  function stopMediaStream() {
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;
  }

  function stopAudioCapture() {
    if (audioProcessorRef.current) {
      audioProcessorRef.current.onaudioprocess = null;
      audioProcessorRef.current.disconnect();
      audioProcessorRef.current = null;
    }
    audioSourceRef.current?.disconnect();
    audioSourceRef.current = null;
    mutedOutputRef.current?.disconnect();
    mutedOutputRef.current = null;
    const audioContext = audioContextRef.current;
    audioContextRef.current = null;
    if (audioContext && audioContext.state !== "closed") {
      void audioContext.close().catch(captureError => {
        console.error("Could not close the microphone audio context.", captureError);
      });
    }
  }

  function chooseInputFile(next?: File) {
    clearRecordingPreview();
    setFile(next);
  }

  async function startRecording() {
    clearRecordingPreview();
    setRecordingSeconds(0);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("This browser does not support microphone recording.");
      return;
    }

    let stream: MediaStream | null = null;
    setRequestingMicrophone(true);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!mountedRef.current) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      const mimeType = [
        "audio/webm;codecs=opus",
        "audio/ogg;codecs=opus",
        "audio/mp4",
      ].find(type => MediaRecorder.isTypeSupported(type));
      const mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      const audioContext = new AudioContext();
      await audioContext.resume();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      const mutedOutput = audioContext.createGain();
      mutedOutput.gain.value = 0;
      audioContextRef.current = audioContext;
      audioSourceRef.current = source;
      audioProcessorRef.current = processor;
      mutedOutputRef.current = mutedOutput;
      pcmChunksRef.current = [];
      pcmFrameCountRef.current = 0;
      recordingTooLargeRef.current = false;
      processor.onaudioprocess = event => {
        const samples = event.inputBuffer.getChannelData(0);
        const maxFrames = Math.floor((MAX_UPLOAD_BYTES - 44) / 2);
        const remainingFrames = maxFrames - pcmFrameCountRef.current;
        if (remainingFrames <= 0) {
          recordingTooLargeRef.current = true;
          if (mediaRecorder.state === "recording") mediaRecorder.stop();
          return;
        }
        const captured = samples.length > remainingFrames
          ? samples.slice(0, remainingFrames)
          : new Float32Array(samples);
        pcmChunksRef.current.push(captured);
        pcmFrameCountRef.current += captured.length;
        if (captured.length < samples.length) {
          recordingTooLargeRef.current = true;
          if (mediaRecorder.state === "recording") mediaRecorder.stop();
        }
      };
      source.connect(processor);
      processor.connect(mutedOutput);
      mutedOutput.connect(audioContext.destination);
      onClear();
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.onerror = () => {
        if (mountedRef.current) setError("Microphone recording failed.");
      };
      mediaRecorder.onstop = async () => {
        if (timerRef.current !== null) {
          window.clearInterval(timerRef.current);
          timerRef.current = null;
        }
        stopAudioCapture();
        stopMediaStream();
        if (!mountedRef.current) return;

        setRecording(false);
        setFinalizingRecording(true);
        try {
          if (recordingTooLargeRef.current) {
            throw new Error("Recording exceeds the 25 MiB limit. Record a shorter clip.");
          }
          const wavBlob = encodePcmToWav(
            pcmChunksRef.current,
            audioContext.sampleRate,
            MAX_UPLOAD_BYTES,
          );
          if (!mountedRef.current) return;
          const wavFile = new File(
            [wavBlob],
            `microphone-${Date.now()}.wav`,
            { type: "audio/wav" },
          );
          clearRecordingPreview();
          const previewUrl = URL.createObjectURL(wavFile);
          previewUrlRef.current = previewUrl;
          setRecordingPreviewUrl(previewUrl);
          setFile(wavFile);
          setError("");
        } catch (recordingError) {
          if (!mountedRef.current) return;
          setFile();
          setError(
            recordingError instanceof Error
              ? recordingError.message
              : "Could not prepare the microphone recording.",
          );
        } finally {
          if (mountedRef.current) setFinalizingRecording(false);
          mediaRecorderRef.current = null;
          pcmChunksRef.current = [];
          pcmFrameCountRef.current = 0;
          recordingTooLargeRef.current = false;
        }
      };

      mediaRecorder.start(250);
      setRecording(true);
      timerRef.current = window.setInterval(
        () => setRecordingSeconds(seconds => seconds + 1),
        1000,
      );
    } catch (recordingError) {
      stopAudioCapture();
      stream?.getTracks().forEach(track => track.stop());
      setError(
        recordingError instanceof Error
          ? `Could not access the microphone: ${recordingError.message}`
          : "Could not access the microphone. Check browser permissions.",
      );
    } finally {
      if (mountedRef.current) setRequestingMicrophone(false);
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
  }

  function discardRecording() {
    clearRecordingPreview();
    setRecordingSeconds(0);
    onClear();
  }

  function formatRecordingTime(seconds: number) {
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  return <>
    <PageHeading eyebrow="AUDIO ANALYSIS" title="Analyze a recording." description="Upload an audio file or record speech with your microphone to classify it as human or synthetic."/>
    <div className="analysis-layout">
      <section className="content-card upload-card">
        <div className="card-heading"><div><h3>Audio input</h3><p>Select a recording for analysis</p></div><FileAudio size={19}/></div>
        <div className="recording-panel">
          <div className="recording-copy">
            <strong>Record speech with your microphone</strong>
            <span>{recording ? "Recording in progress" : finalizingRecording ? "Preparing WAV audio…" : "Your recording will be converted to WAV before analysis."}</span>
          </div>
          <div className="recording-controls">
            {recording
              ? <><span className="recording-timer"><span className="recording-dot"/> {formatRecordingTime(recordingSeconds)}</span><button className="secondary-button" onClick={stopRecording}><span className="stop-square"/> Stop recording</button></>
              : <button className="secondary-button" disabled={requestingMicrophone || finalizingRecording || loading} onClick={startRecording}><span className="recording-dot"/> {requestingMicrophone ? "Requesting access…" : finalizingRecording ? "Preparing…" : "Start recording"}</button>}
          </div>
          {recordingPreviewUrl && <div className="recording-preview">
            <audio controls src={recordingPreviewUrl}>Audio playback is not supported by this browser.</audio>
            <button className="text-button" onClick={discardRecording}>Discard and re-record</button>
          </div>}
        </div>
        <input ref={inputRef} type="file" accept={accepted} hidden onChange={e=>{const selected=e.target.files?.[0];e.currentTarget.value="";chooseInputFile(selected);}}/>
        {!file ? <div className={`dropzone ${dragging?"dragging":""} ${recording || requestingMicrophone || finalizingRecording ? "disabled-dropzone" : ""}`} onDragOver={e=>{e.preventDefault();if(!recording&&!requestingMicrophone&&!finalizingRecording)setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);if(!recording&&!requestingMicrophone&&!finalizingRecording)chooseInputFile(e.dataTransfer.files?.[0]);}} onClick={()=>{if(!recording&&!requestingMicrophone&&!finalizingRecording)inputRef.current?.click();}}>
          <div className="upload-icon"><UploadCloud size={27}/></div><h4>Drop your audio file here</h4><p>or <button className="text-button" disabled={recording || requestingMicrophone || finalizingRecording} onClick={e=>{e.stopPropagation();inputRef.current?.click();}}>browse files</button> from your device</p><div className="file-hints"><span>WAV</span><span>FLAC</span><span>MP3</span><span>OGG</span><span>M4A</span><i>Up to 25 MiB</i></div>
        </div> : <div className="selected-file"><div className="file-icon"><FileAudio size={24}/></div><div className="selected-info"><strong>{file.name}</strong><span>{(file.size/1024/1024).toFixed(2)} MB · Ready for analysis</span></div><button className="icon-button" onClick={discardRecording} aria-label="Remove file"><X size={18}/></button></div>}
        {error && <div className="error-box"><AlertTriangle size={17}/>{error}</div>}
        <div className="upload-actions"><div className="privacy-note"><ShieldCheck size={15}/> File is sent to your configured backend for analysis.</div><button className="primary-button" disabled={!file||loading||recording||requestingMicrophone||finalizingRecording} onClick={onRun}>{loading?<><LoaderCircle className="spin" size={17}/> Analyzing…</>:<><Sparkles size={17}/> Run detection</>}</button></div>
        {loading&&<div className="progress-wrap"><div className="progress-bar"><span/></div><small>Processing audio and generating a prediction…</small></div>}
      </section>
      <aside className="content-card guidance-card"><div className="card-heading"><div><h3>Before you begin</h3><p>Input requirements</p></div><FileCheck2 size={18}/></div>
        <div className="guidance-item"><span className="guidance-check"><CheckCircle2 size={16}/></span><div><strong>Supported formats</strong><p>WAV, FLAC, MP3, OGG and M4A files.</p></div></div>
        <div className="guidance-item"><span className="guidance-check"><CheckCircle2 size={16}/></span><div><strong>File size</strong><p>Maximum upload size is 25 MiB.</p></div></div>
        <div className="guidance-item"><span className="guidance-check"><CheckCircle2 size={16}/></span><div><strong>Clear speech</strong><p>Use recordings with audible speech and minimal background noise.</p></div></div>
        <div className="notice-box"><AlertTriangle size={16}/><p>Detection is a machine learning estimate, not forensic proof of audio authenticity.</p></div>
      </aside>
    </div>
    {result&&<section className="content-card result-card"><div className="card-heading"><div><h3>Detection result</h3><p>Analysis completed for {result.filename}</p></div><span className="success-tag"><CheckCircle2 size={14}/> COMPLETE</span></div>
      <div className={`result-main ${result.label==="human"?"human-result":"synthetic-result"}`}><div className="result-symbol">{result.label==="human"?<ShieldCheck size={30}/>:<AudioWaveform size={30}/>}</div><div><div className="result-overline">CLASSIFICATION</div><h2>{result.label==="human"?"Human speech detected":"Synthetic speech detected"}</h2><p>The model classified this recording as <strong>{result.label}</strong>.</p></div></div>
      <div className="result-details">
        {result.confidence!==undefined&&<div className="result-detail"><span>Model confidence</span><strong>{(result.confidence*100).toFixed(2)}%</strong><div className="confidence-track"><span style={{width:`${Math.max(0,Math.min(100,result.confidence*100))}%`}}/></div></div>}
        {result.probabilities&&<><div className="result-detail"><span>Human probability</span><strong>{result.probabilities.human!==undefined?`${(result.probabilities.human*100).toFixed(2)}%`:"—"}</strong></div><div className="result-detail"><span>Synthetic probability</span><strong>{result.probabilities.synthetic!==undefined?`${(result.probabilities.synthetic*100).toFixed(2)}%`:"—"}</strong></div></>}
        <div className="result-detail"><span>Model</span><strong>{result.model||"XGBoost"}</strong></div>
        {result.processing_time_ms!==undefined&&<div className="result-detail"><span>Processing time</span><strong>{result.processing_time_ms} ms</strong></div>}
      </div>
    </section>}
  </>;
}

function ResultBadge({label}:{label:string}) {
  return <span className={`result-badge ${label==="human"?"human-badge":"synthetic-badge"}`}>{label}</span>;
}

function HistoryPage({history,onAnalyze}:{history:Prediction[];onAnalyze:()=>void}) {
  return <>
    <PageHeading eyebrow="ANALYSIS HISTORY" title="Your recent analyses." description="Review recordings analyzed during this browser session." action={<button className="primary-button" onClick={onAnalyze}><UploadCloud size={17}/> New analysis</button>}/>
    <div className="content-card table-card">
      {history.length ? <div className="table-scroll"><table><thead><tr><th>FILE</th><th>CLASSIFICATION</th><th>CONFIDENCE</th><th>MODEL</th><th>DATE</th></tr></thead><tbody>{history.map((item,i)=><tr key={`${item.filename}-${i}`}><td><div className="table-file"><FileAudio size={17}/>{item.filename}</div></td><td><ResultBadge label={item.label}/></td><td>{item.confidence!==undefined?`${(item.confidence*100).toFixed(1)}%`:"Not provided"}</td><td>{item.model||"XGBoost"}</td><td>{new Date(item.analyzed_at).toLocaleString()}</td></tr>)}</tbody></table></div> : <div className="empty-state"><History size={34}/><strong>No analysis history</strong><span>Upload an audio file to create your first result.</span><button className="primary-button" onClick={onAnalyze}>Analyze audio</button></div>}
    </div>
    <p className="footnote">History is stored in component state and will reset when this page is refreshed. Persistent history requires backend storage.</p>
  </>;
}

function ModelPage() {
  return <>
    <PageHeading eyebrow="MODEL DETAILS" title="Inside the detection engine." description="Evaluation details for the currently selected SonicSense classifier."/>
    <div className="model-hero"><div className="model-icon"><Microscope size={28}/></div><div><span className="tag">SELECTED MODEL</span><h2>XGBoost classifier</h2><p>Gradient-boosted decision trees trained on engineered audio features.</p></div><div className="model-status"><span className="status-dot"/> TRAINED MODEL</div></div>
    <div className="section-title"><div><h3>Held-out test performance</h3><p>Metrics recorded by the Member 3 training pipeline.</p></div></div>
    <div className="metric-grid">
      <Metric icon={Gauge} label="Accuracy" value="95.96%" note="Overall correct predictions"/>
      <Metric icon={Activity} label="Macro F1" value="94.44%" note="Unweighted class F1 average"/>
      <Metric icon={CheckCircle2} label="Human recall" value="96.57%" note="Human samples correctly identified"/>
      <Metric icon={Microscope} label="ROC-AUC" value="99.30%" note="Human as positive class"/>
    </div>
    <div className="bottom-grid model-bottom">
      <div className="content-card"><div className="card-heading"><div><h3>Evaluation dataset</h3><p>Stratified random split</p></div><Activity size={18}/></div><div className="dataset-stats"><div><span>Training samples</span><strong>33,180</strong></div><div><span>Validation samples</span><strong>7,110</strong></div><div><span>Test samples</span><strong>7,110</strong></div><div><span>Input features</span><strong>134</strong></div></div></div>
      <div className="content-card"><div className="card-heading"><div><h3>Important limitation</h3><p>Interpret results responsibly</p></div><ShieldCheck size={18}/></div><div className="notice-box large-notice"><AlertTriangle size={17}/><p>The available metadata does not include speaker or source-recording identifiers. Evaluation used a stratified random split and should not be described as speaker-independent. Performance on unseen speakers, recording devices or datasets may differ.</p></div></div>
    </div>
  </>;
}

export default App;
