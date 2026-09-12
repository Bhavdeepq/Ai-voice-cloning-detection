import { useEffect, useRef, useState } from 'react'
import { createWavBlob } from './wav.js'

const API_URL = 'http://127.0.0.1:8000/detect'
const CHUNK_DURATION_MS = 4000
const MIN_FINAL_CHUNK_SAMPLES = 4000

const initialResult = null

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

export default function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [pendingRequests, setPendingRequests] = useState(0)
  const [result, setResult] = useState(initialResult)
  const [chunksAnalyzed, setChunksAnalyzed] = useState(0)
  const [error, setError] = useState('')
  const audioRef = useRef(null)
  const framesRef = useRef([])
  const intervalRef = useRef(null)
  const requestNumberRef = useRef(0)
  const fileInputRef = useRef(null)

  const status = isRecording ? (pendingRequests > 0 ? 'Analyzing' : 'Listening') : isUploading ? 'Analyzing' : 'Stopped'
  const isSpoof = result?.prediction === 'AI_SPOOF'

  useEffect(() => () => stopAudioCapture(), [])

  function stopAudioCapture() {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    const audio = audioRef.current
    if (!audio) return
    audio.processor.disconnect()
    audio.source.disconnect()
    audio.silentGain.disconnect()
    audio.stream.getTracks().forEach((track) => track.stop())
    audio.context.close()
    audioRef.current = null
  }

  async function analyzeChunk(frames, sampleRate) {
    const sampleCount = frames.reduce((total, frame) => total + frame.length, 0)
    if (!sampleCount) return

    setPendingRequests((count) => count + 1)
    requestNumberRef.current += 1
    const formData = new FormData()
    formData.append('file', createWavBlob(frames, sampleRate), `microphone-chunk-${requestNumberRef.current}.wav`)

    try {
      const response = await fetch(API_URL, { method: 'POST', body: formData })
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail || `Server returned ${response.status}`)
      }
      setResult(payload)
      setChunksAnalyzed((count) => count + 1)
      setError('')
    } catch (requestError) {
      setError(`Unable to analyze this audio chunk. ${requestError.message}`)
    } finally {
      setPendingRequests((count) => Math.max(0, count - 1))
    }
  }

  function submitCurrentChunk({ includeShortChunk = false } = {}) {
    const audio = audioRef.current
    if (!audio) return
    const frames = framesRef.current
    const sampleCount = frames.reduce((total, frame) => total + frame.length, 0)
    if (!sampleCount || (!includeShortChunk && sampleCount < audio.context.sampleRate * 2)) return
    framesRef.current = []
    analyzeChunk(frames, audio.context.sampleRate)
  }

  async function startDetection() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone recording is not supported by this browser. Use a current desktop browser.')
      return
    }

    setError('')
    setResult(null)
    setChunksAnalyzed(0)
    requestNumberRef.current = 0
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const AudioContextClass = window.AudioContext || window.webkitAudioContext
      const context = new AudioContextClass()
      await context.resume()
      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(4096, 1, 1)
      const silentGain = context.createGain()
      silentGain.gain.value = 0

      processor.onaudioprocess = (event) => {
        framesRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)))
      }
      source.connect(processor)
      processor.connect(silentGain)
      silentGain.connect(context.destination)
      audioRef.current = { context, processor, silentGain, source, stream }
      framesRef.current = []
      intervalRef.current = window.setInterval(() => submitCurrentChunk(), CHUNK_DURATION_MS)
      setIsRecording(true)
    } catch (microphoneError) {
      setError(`Microphone access was not granted. ${microphoneError.message}`)
      stopAudioCapture()
    }
  }

  function stopDetection() {
    const audio = audioRef.current
    if (audio) {
      const sampleCount = framesRef.current.reduce((total, frame) => total + frame.length, 0)
      if (sampleCount >= MIN_FINAL_CHUNK_SAMPLES) submitCurrentChunk({ includeShortChunk: true })
    }
    stopAudioCapture()
    setIsRecording(false)
  }

  async function uploadAudio(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file || isRecording) return

    setError('')
    setResult(null)
    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', file, file.name)

    try {
      const response = await fetch(API_URL, { method: 'POST', body: formData })
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail || `Server returned ${response.status}`)
      }
      setResult(payload)
      setChunksAnalyzed(1)
    } catch (requestError) {
      setError(`Unable to analyze the selected audio file. ${requestError.message}`)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="hero" aria-labelledby="page-title">
        <div className="brand"><span className="brand-mark">◈</span> VoiceShield</div>
        <p className="eyebrow">SIH PROJECT DEMONSTRATION</p>
        <h1 id="page-title">Real-Time AI Voice Detection</h1>
        <p className="hero-copy">Listen continuously, analyze short voice windows, and identify potential AI-cloned speech as it is captured.</p>
      </section>

      <section className="dashboard" aria-label="Live detection dashboard">
        <div className="control-card">
          <div className="status-row">
            <span className={`status-dot ${status.toLowerCase()}`} />
            <span>System status</span>
            <strong>{status}</strong>
          </div>
          <div className="microphone-visual" aria-hidden="true">⌁</div>
          <h2>{isRecording ? 'Microphone is active' : isUploading ? 'Audio is being analyzed' : 'Ready to listen'}</h2>
          <p>Each completed window is securely sent to the local detector for analysis.</p>
          <div className="controls">
            <button className="button primary" onClick={startDetection} disabled={isRecording || isUploading}>Start Detection</button>
            <button className="button secondary" onClick={stopDetection} disabled={!isRecording}>Stop Detection</button>
          </div>
          <div className="upload-section">
            <div className="upload-heading"><span>OR</span></div>
            <h3>Analyze a completed audio file</h3>
            <p>Upload a WAV or other supported audio file for a one-time detection result.</p>
            <input ref={fileInputRef} id="audio-upload" className="file-input" type="file" accept="audio/wav,audio/x-wav,audio/*,.wav" onChange={uploadAudio} disabled={isRecording || isUploading} />
            <label className={`upload-button ${isRecording || isUploading ? 'disabled' : ''}`} htmlFor="audio-upload">{isUploading ? 'Analyzing audio…' : 'Upload Audio'}</label>
          </div>
          {error && <p className="error-message" role="alert">{error}</p>}
        </div>

        <div className={`result-card ${result ? (isSpoof ? 'warning' : 'safe') : 'idle'}`}>
          <p className="result-label">Latest detection</p>
          {result ? <>
            <div className="result-heading"><span className="result-icon">{isSpoof ? '!' : '✓'}</span><h2>{result.prediction === 'AI_SPOOF' ? 'AI voice detected' : 'Likely genuine voice'}</h2></div>
            <p className="prediction">{result.prediction}</p>
            <div className="confidence"><span>Confidence</span><strong>{formatPercent(result.confidence)}</strong></div>
            <div className="segments"><span>Segments analyzed</span><strong>{result.segments_analyzed}</strong></div>
            <div className="probabilities">
              <Probability label="Real probability" value={result.real_probability} tone="real" />
              <Probability label="AI/Fake probability" value={result.fake_probability} tone="fake" />
            </div>
          </> : <div className="empty-result"><span>◌</span><p>Start detection to see the latest result.</p></div>}
        </div>
      </section>

      <section className="details">
        <article><span className="detail-value">{chunksAnalyzed}</span><span>Chunks analyzed</span></article>
        <article><span className="detail-value">~4 sec</span><span>Analysis window</span></article>
        <article><span className="detail-value">HTTP</span><span>Local detector connection</span></article>
      </section>
    </main>
  )
}

function Probability({ label, value, tone }) {
  const percentage = Math.max(0, Math.min(100, value * 100))
  return <div className="probability"><div><span>{label}</span><strong>{formatPercent(value)}</strong></div><div className="meter"><i className={tone} style={{ width: `${percentage}%` }} /></div></div>
}
