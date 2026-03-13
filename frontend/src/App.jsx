import { useState, useEffect, useRef, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend,
  LineChart, Line,
} from 'recharts'

const API = '/api'

const CRITERIA = [
  { key: 'accuracy',  label: 'Accuracy',  icon: '🎯' },
  { key: 'clarity',   label: 'Clarity',   icon: '💡' },
  { key: 'depth',     label: 'Depth',     icon: '🔬' },
  { key: 'concision', label: 'Concision', icon: '✂️' },
  { key: 'examples',  label: 'Examples',  icon: '📐' },
]

const PLAYGROUND_PRESETS = [
  { label: 'Ollama (local)',  endpoint: 'http://localhost:11434/v1/chat/completions', model: 'mistral:7b' },
  { label: 'OpenAI',         endpoint: 'https://api.openai.com/v1/chat/completions', model: 'gpt-4o-mini' },
  { label: 'Groq',           endpoint: 'https://api.groq.com/openai/v1/chat/completions', model: 'llama3-8b-8192' },
  { label: 'Together AI',    endpoint: 'https://api.together.xyz/v1/chat/completions', model: 'meta-llama/Llama-3-8b-chat-hf' },
  { label: 'LM Studio',      endpoint: 'http://localhost:1234/v1/chat/completions', model: 'local-model' },
]

// ── Tiny UI pieces ─────────────────────────────────────────────────────────────

function Spinner({ size = 14, color = 'var(--cyan)' }) {
  return <span style={{ display:'inline-block', width:size, height:size,
    border:`2px solid var(--border2)`, borderTopColor:color,
    borderRadius:'50%', animation:'spin 0.7s linear infinite', flexShrink:0 }} />
}

function Dot({ color, pulse }) {
  return <span style={{ display:'inline-block', width:8, height:8, borderRadius:'50%',
    background: color, boxShadow: `0 0 6px ${color}`,
    animation: pulse ? 'pulse 1.5s ease-in-out infinite' : 'none', flexShrink:0 }} />
}

function Tag({ children, color = 'var(--muted)' }) {
  return <span style={{ display:'inline-block', padding:'1px 7px', borderRadius:3,
    fontFamily:'var(--mono)', fontSize:10, background:color+'22', color, border:`1px solid ${color}44` }}>
    {children}
  </span>
}

function ScoreBar({ value, max = 5, color }) {
  const pct = ((value || 0) / max) * 100
  return (
    <div style={{ display:'flex', alignItems:'center', gap:8 }}>
      <div style={{ flex:1, height:5, background:'var(--border)', borderRadius:3, overflow:'hidden' }}>
        <div style={{ width:`${pct}%`, height:'100%', background:color, borderRadius:3,
          transition:'width 0.8s ease', boxShadow:`0 0 5px ${color}88` }} />
      </div>
      <span style={{ fontFamily:'var(--mono)', fontSize:11, color, minWidth:24 }}>
        {value?.toFixed ? value.toFixed(1) : value}
      </span>
    </div>
  )
}

function Stars({ value, onChange, color }) {
  const [hover, setHover] = useState(null)
  const active = hover ?? value
  return (
    <div style={{ display:'flex', gap:2 }}>
      {[1,2,3,4,5].map(n => (
        <span key={n}
          onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(null)}
          onClick={() => onChange?.(n)}
          style={{ fontSize:18, cursor:onChange?'pointer':'default',
            color: n <= active ? color : 'var(--border2)', transition:'color 0.1s',
            filter: n <= active ? `drop-shadow(0 0 4px ${color})` : 'none' }}>★</span>
      ))}
    </div>
  )
}

function DarkTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background:'var(--surface2)', border:'1px solid var(--border2)',
      borderRadius:6, padding:'8px 12px', fontFamily:'var(--mono)', fontSize:11 }}>
      <div style={{ color:'var(--muted)', marginBottom:4 }}>{label}</div>
      {payload.map((p,i) => (
        <div key={i} style={{ color:p.color||'var(--text)', display:'flex', gap:12, justifyContent:'space-between' }}>
          <span>{p.name || p.dataKey}</span>
          <span>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</span>
        </div>
      ))}
    </div>
  )
}

function SectionHeader({ children }) {
  return <div style={{ fontFamily:'var(--mono)', fontSize:10, color:'var(--muted)',
    textTransform:'uppercase', letterSpacing:2, marginBottom:14,
    paddingBottom:8, borderBottom:'1px solid var(--border)' }}>{children}</div>
}

function StatCard({ label, value, sub, color='var(--cyan)' }) {
  return (
    <div style={{ background:'var(--surface)', border:'1px solid var(--border)',
      borderTop:`2px solid ${color}`, borderRadius:6, padding:'14px 16px' }}>
      <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:1, marginBottom:5 }}>{label}</div>
      <div style={{ fontFamily:'var(--mono)', fontSize:24, color, lineHeight:1 }}>{value}</div>
      {sub && <div style={{ fontSize:10, color:'var(--muted)', marginTop:4, fontFamily:'var(--mono)' }}>{sub}</div>}
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────────

const TABS = [
  { id:'run',        label:'⚡ Run' },
  { id:'metrics',    label:'📊 Metrics' },
  { id:'responses',  label:'📝 Responses' },
  { id:'overall',    label:'🏆 Overall' },
  { id:'stream',     label:'💬 Stream' },
  { id:'playground', label:'🔌 Playground' },
  { id:'history',    label:'📚 History' },
]

const DEFAULT_PANEL = (idx) => ({
  endpoint: idx === 0
    ? 'http://localhost:11434/v1/chat/completions'
    : 'http://localhost:11434/v1/chat/completions',
  model:    idx === 0 ? 'mistral:7b' : 'llama3:8b',
  apiKey:   '',
  sysPrompt:'',
  label:    idx === 0 ? 'Model A' : 'Model B',
  color:    idx === 0 ? 'var(--cyan)' : 'var(--purple)',
})

export default function App() {
  const [tab, setTab] = useState('run')

  // ── Benchmark state ────────────────────────────────────────────────────────
  const [catalog, setCatalog] = useState([])
  const [selected, setSelected] = useState([])
  const [customModel, setCustomModel] = useState('')
  const [customModels, setCustomModels] = useState([])
  const [prompt, setPrompt] = useState("Explain Einstein's theory of relativity in simple terms.")
  const [nRuns, setNRuns] = useState(3)
  const [nTokens, setNTokens] = useState(512)
  const [autoJudge, setAutoJudge] = useState(true)
  const [keepAlive, setKeepAlive] = useState(false)
  const [sequential, setSequential] = useState(false)
  const [judgeSystemPrompt, setJudgeSystemPrompt] = useState('')
  const [judgeModel, setJudgeModel] = useState('qwen2.5:7b')
  const judgeConfigSaveTimer = useRef(null)
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState({})
  const [judgeScores, setJudgeScores] = useState({})
  const [humanScores, setHumanScores] = useState({})
  const [leaderboard, setLeaderboard] = useState([])
  const [containerStatus, setContainerStatus] = useState({})
  const [log, setLog] = useState([])
  const [judging, setJudging] = useState({})
  const logRef = useRef(null)

  // ── Stream tab state ───────────────────────────────────────────────────────
  const [streamRunning, setStreamRunning] = useState(false)
  const [streamTexts, setStreamTexts]     = useState({})   // model → accumulated text
  const [streamStats, setStreamStats]     = useState({})   // model → { tps, ttft_ms, tokens }
  const [streamStatus, setStreamStatus]   = useState({})   // model → 'spawning'|'streaming'|'done'|'error'
  const [streamErrors, setStreamErrors]   = useState({})   // model → error string
  const streamPanelRefs = useRef({})
  const streamAbortRef  = useRef(null)

  // ── Model pull state ────────────────────────────────────────────────────────
  const [localModels, setLocalModels]     = useState([])         // [{name, size, ...}]
  const [pullStatus, setPullStatus]       = useState({})         // model → 'pulling'|'done'|'error'
  const [pullProgress, setPullProgress]   = useState({})         // model → 0-100
  const [pullMsg, setPullMsg]             = useState({})         // model → last status string

  // ── History state ───────────────────────────────────────────────────────────
  const [history, setHistory]             = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)

  // ── Last-run config (captured at run time for metrics scope card) ────────────
  const [lastRunConfig, setLastRunConfig] = useState({ prompt:'', nTokens:512, nRuns:3 })

  // ── Playground tab state ───────────────────────────────────────────────────
  const [playPanels, setPlayPanels]   = useState([DEFAULT_PANEL(0), DEFAULT_PANEL(1)])
  const [playPrompt, setPlayPrompt]   = useState('')
  const [playRunning, setPlayRunning] = useState(false)
  const [playTexts, setPlayTexts]     = useState(['', ''])
  const [playErrors, setPlayErrors]   = useState(['', ''])
  const [playDone, setPlayDone]       = useState([false, false])
  const [playStats, setPlayStats]     = useState([null, null])  // { ttft_ms, chars }
  const playT0 = useRef([null, null])
  const playPanelRefs = useRef([null, null])

  // ── Helpers ────────────────────────────────────────────────────────────────
  const addLog = useCallback((msg, type='info') => {
    const ts = new Date().toLocaleTimeString('en-US', { hour12:false })
    setLog(prev => [...prev.slice(-80), { ts, msg, type }])
  }, [])

  const fetchLocalModels = useCallback(() => {
    fetch(`${API}/models/local`)
      .then(r => r.json())
      .then(data => setLocalModels(Array.isArray(data) ? data : []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetch(`${API}/models/catalog`)
      .then(r => r.json())
      .then(data => {
        setCatalog(data)
        const defaults = ['mistral:7b','llama3:8b','gemma2:9b','deepseek-r1:7b']
        setSelected(data.filter(m => defaults.includes(m.name)).map(m => m.name))
      })
      .catch(() => addLog('Cannot reach orchestrator — is Docker running?', 'error'))
    fetch(`${API}/judge/config`)
      .then(r => r.json())
      .then(data => { setJudgeSystemPrompt(data.system_prompt); setJudgeModel(data.model) })
      .catch(() => {})
    fetchLocalModels()
  }, [])

  // Refresh local model list when switching to History tab, and fetch history
  useEffect(() => {
    if (tab === 'run') fetchLocalModels()
    if (tab === 'history') {
      setHistoryLoading(true)
      fetch(`${API}/history`)
        .then(r => r.json())
        .then(data => setHistory(Array.isArray(data) ? data : []))
        .catch(() => setHistory([]))
        .finally(() => setHistoryLoading(false))
    }
  }, [tab])

  const saveJudgeConfig = (prompt, model) => {
    clearTimeout(judgeConfigSaveTimer.current)
    judgeConfigSaveTimer.current = setTimeout(() => {
      fetch(`${API}/judge/config`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ system_prompt: prompt, model }),
      }).catch(() => {})
    }, 600)
  }

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log])

  // Auto-scroll streaming panels
  useEffect(() => {
    Object.values(streamPanelRefs.current).forEach(el => {
      if (el) el.scrollTop = el.scrollHeight
    })
  }, [streamTexts])

  useEffect(() => {
    playPanelRefs.current.forEach(el => {
      if (el) el.scrollTop = el.scrollHeight
    })
  }, [playTexts])

  // Models downloaded in Ollama but not in the preset catalog or custom list
  const catalogBaseNames = new Set(catalog.map(m => m.name.split(':')[0]))
  const customNames = new Set(customModels)
  const localOnlyModels = localModels
    .filter(lm => {
      if (customNames.has(lm.name)) return false
      const base = lm.name.split(':')[0]
      return !catalogBaseNames.has(base)
    })
    .map(lm => ({
      name: lm.name,
      color: '#5ccfe6',
      local: true,
      size_gb: lm.size ? (lm.size / 1e9).toFixed(1) : null,
    }))

  const allModels = [
    ...catalog,
    ...customModels.map(n => ({ name:n, color:'#aaaaaa', custom:true })),
    ...localOnlyModels,
  ]

  // ── Model pull ─────────────────────────────────────────────────────────────
  const pullModel = async (modelName) => {
    setPullStatus(p => ({ ...p, [modelName]: 'pulling' }))
    setPullProgress(p => ({ ...p, [modelName]: 0 }))
    setPullMsg(p => ({ ...p, [modelName]: 'connecting…' }))
    addLog(`↓ Pulling ${modelName}…`, 'cyan')
    try {
      const res = await fetch(`${API}/models/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
      })
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const chunk = JSON.parse(line.slice(6))
            if (chunk.status === 'error') {
              setPullStatus(p => ({ ...p, [modelName]: 'error' }))
              setPullMsg(p => ({ ...p, [modelName]: chunk.error || 'Pull failed' }))
              addLog(`  ✗ Pull failed: ${chunk.error}`, 'error')
              return
            }
            setPullMsg(p => ({ ...p, [modelName]: chunk.status || '' }))
            if (chunk.total && chunk.completed != null) {
              setPullProgress(p => ({ ...p, [modelName]: Math.round((chunk.completed / chunk.total) * 100) }))
            }
            if (chunk.status === 'success') {
              setPullStatus(p => ({ ...p, [modelName]: 'done' }))
              setPullProgress(p => ({ ...p, [modelName]: 100 }))
              addLog(`  ✓ ${modelName} ready`, 'green')
              fetchLocalModels()
              return
            }
          } catch {}
        }
      }
    } catch (e) {
      setPullStatus(p => ({ ...p, [modelName]: 'error' }))
      setPullMsg(p => ({ ...p, [modelName]: e.message }))
      addLog(`  ✗ Pull error: ${e.message}`, 'error')
    }
  }

  // ── History restore ────────────────────────────────────────────────────────
  const restoreHistory = (entry) => {
    setResults(entry.results || {})
    setJudgeScores(entry.judge_scores || {})
    setLeaderboard(entry.leaderboard || [])
    setSelected(entry.models || [])
    if (entry.prompt) {
      setPrompt(entry.prompt)
      setLastRunConfig({ prompt: entry.prompt, nTokens: entry.n_tokens ?? 512, nRuns: entry.n_runs ?? 3 })
    }
    addLog(`📚 Restored run from ${new Date(entry.timestamp * 1000).toLocaleString()}`, 'cyan')
    setTab('metrics')
  }

  // ── PDF export ─────────────────────────────────────────────────────────────
  const exportPDF = async () => {
    try {
      const res = await fetch(`${API}/export/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt:       lastRunConfig.prompt || prompt,
          n_tokens:     lastRunConfig.nTokens ?? nTokens,
          n_runs:       lastRunConfig.nRuns ?? nRuns,
          results,
          judge_scores: judgeScores,
          leaderboard,
        }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `judgegpt-report-${Date.now()}.pdf`
      a.click()
      URL.revokeObjectURL(url)
      addLog('↓ PDF report downloaded', 'green')
    } catch (e) {
      addLog(`PDF export failed: ${e.message}`, 'error')
    }
  }

  const addCustomModel = () => {
    const name = customModel.trim()
    if (!name || allModels.find(m => m.name === name)) return
    setCustomModels(prev => [...prev, name])
    setSelected(prev => [...prev, name])
    setCustomModel('')
    addLog(`Added custom model: ${name}`, 'yellow')
  }

  // ── Benchmark runner ───────────────────────────────────────────────────────
  const runBenchmark = async () => {
    if (!selected.length || running) return
    setRunning(true)
    setResults({})
    setJudgeScores({})
    setLeaderboard([])
    setContainerStatus({})
    setLastRunConfig({ prompt, nTokens, nRuns })
    addLog(`▶ Starting benchmark: ${selected.join(', ')}`, 'cyan')
    addLog(`  Prompt: "${prompt.slice(0,55)}…"`)
    addLog(`  ${nRuns} runs · ${nTokens === -1 ? '∞' : nTokens} tokens · judge: ${autoJudge} · ${sequential ? 'sequential' : 'concurrent'}`)

    try {
      const res = await fetch(`${API}/benchmark/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_names:selected, prompt, n_tokens:nTokens,
                               n_runs:nRuns, auto_judge:autoJudge, keep_alive:keepAlive,
                               sequential }),
      })
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream:true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try { handleSSEEvent(JSON.parse(line.slice(6))) } catch {}
        }
      }
    } catch (e) {
      addLog(`Error: ${e.message}`, 'error')
    } finally {
      setRunning(false)
    }
  }

  const handleSSEEvent = (evt) => {
    switch (evt.event) {
      case 'spawn_start':
        setContainerStatus(p => ({ ...p, [evt.model]: 'spawning' }))
        addLog(`  Spawning container for ${evt.model}…`, 'muted')
        break
      case 'spawn_done':
        setContainerStatus(p => ({ ...p, [evt.model]: 'running' }))
        addLog(`  ✓ ${evt.model} ready on :${evt.port}`, 'green')
        break
      case 'warmup':
        addLog(`  Warming up ${evt.model}…`, 'muted')
        break
      case 'run_start':
        addLog(`  ${evt.model} — run ${evt.run}/${evt.total}`, 'muted')
        break
      case 'run_done':
        addLog(`  ${evt.model}  run ${evt.run}: ${evt.tps} tok/s  ${evt.ttft_ms}ms`, 'green')
        break
      case 'run_error':
        addLog(`  ✗ ${evt.model} run ${evt.run}: ${evt.error}`, 'error')
        break
      case 'model_complete':
        setResults(p => ({ ...p, [evt.model]: evt.result }))
        addLog(`✓ ${evt.model}  avg ${evt.result.tps_mean} tok/s  ${evt.result.ttft_mean}ms TTFT`, 'cyan')
        setTab('metrics')
        break
      case 'judging_start':
        addLog('⚖ Running LLM judge…', 'purple')
        break
      case 'judge_done':
        setJudgeScores(p => ({ ...p, [evt.model]: evt.score }))
        const avg = ['accuracy','clarity','depth','concision','examples']
          .reduce((s,k) => s + (evt.score[k] || 0), 0) / 5
        addLog(`  ⚖ ${evt.model}: ${avg.toFixed(2)}/5`, 'purple')
        break
      case 'judge_error':
        addLog(`  ⚖ Judge failed for ${evt.model}: ${evt.error}`, 'error')
        break
      case 'error':
        addLog(`✗ ${evt.model}: ${evt.error}`, 'error')
        setContainerStatus(p => ({ ...p, [evt.model]: 'error' }))
        break
      case 'done':
        if (evt.leaderboard) setLeaderboard(evt.leaderboard)
        addLog('✓ Benchmark complete', 'cyan')
        if (evt.leaderboard?.length) setTab('overall')
        break
    }
  }

  const runJudge = async (modelName) => {
    setJudging(p => ({ ...p, [modelName]: true }))
    addLog(`⚖ Judging ${modelName}…`, 'purple')
    try {
      const data = await fetch(`${API}/judge/run/${modelName}`, { method:'POST' }).then(r => r.json())
      setJudgeScores(p => ({ ...p, [modelName]: data.score }))
      if (data.leaderboard) setLeaderboard(data.leaderboard)
      addLog(`  ✓ ${modelName} scored`, 'purple')
    } catch(e) {
      addLog(`  ✗ Judge error: ${e.message}`, 'error')
    } finally {
      setJudging(p => ({ ...p, [modelName]: false }))
    }
  }

  const setHuman = async (modelName, score) => {
    setHumanScores(p => ({ ...p, [modelName]: score }))
    const data = await fetch(`${API}/judge/human-score`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ model_name:modelName, score }),
    }).then(r => r.json())
    if (data.leaderboard) setLeaderboard(data.leaderboard)
    addLog(`⭐ Human score: ${modelName} → ${score}/5`, 'yellow')
  }

  // ── Stream runner ──────────────────────────────────────────────────────────
  const stopStream = () => {
    streamAbortRef.current?.abort()
    streamAbortRef.current = null
    setStreamRunning(false)
  }

  const runStream = async () => {
    if (!selected.length || streamRunning) return
    const ctrl = new AbortController()
    streamAbortRef.current = ctrl
    setStreamRunning(true)
    setStreamTexts({})
    setStreamStats({})
    setStreamErrors({})
    setStreamStatus(Object.fromEntries(selected.map(m => [m, 'spawning'])))

    try {
      const res = await fetch(`${API}/benchmark/stream-live`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_names:selected, prompt, n_tokens:nTokens, keep_alive:false, sequential }),
        signal: ctrl.signal,
      })
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream:true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const evt = JSON.parse(line.slice(6))
            if (evt.event === 'model_ready') {
              setStreamStatus(p => ({ ...p, [evt.model]: 'streaming' }))
            } else if (evt.event === 'token') {
              setStreamTexts(p => ({ ...p, [evt.model]: (p[evt.model] || '') + evt.text }))
            } else if (evt.event === 'model_done') {
              setStreamStats(p => ({ ...p, [evt.model]: { tps:evt.tps, ttft_ms:evt.ttft_ms, tokens:evt.tokens } }))
              setStreamStatus(p => ({ ...p, [evt.model]: 'done' }))
            } else if (evt.event === 'model_error') {
              setStreamErrors(p => ({ ...p, [evt.model]: evt.error || 'Unknown error' }))
              setStreamStatus(p => ({ ...p, [evt.model]: 'error' }))
            } else if (evt.event === 'stream_done') {
              setStreamRunning(false)
            }
          } catch {}
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') addLog(`Stream error: ${e.message}`, 'error')
    } finally {
      streamAbortRef.current = null
      setStreamRunning(false)
    }
  }

  // ── Playground runner ──────────────────────────────────────────────────────
  const runPlayground = async () => {
    if (playRunning || !playPrompt.trim()) return
    setPlayRunning(true)
    setPlayTexts(['', ''])
    setPlayErrors(['', ''])
    setPlayDone([false, false])
    setPlayStats([null, null])
    playT0.current = [performance.now(), performance.now()]

    const runPanel = async (idx) => {
      const panel = playPanels[idx]
      let ttft = null
      let chars = 0
      try {
        const res = await fetch(`${API}/proxy/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            endpoint:     panel.endpoint,
            model:        panel.model,
            messages:     [{ role:'user', content:playPrompt }],
            api_key:      panel.apiKey,
            system_prompt:panel.sysPrompt,
            max_tokens:   512,
          }),
        })
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream:true })
          const lines = buffer.split('\n')
          buffer = lines.pop()
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const evt = JSON.parse(line.slice(6))
              if (evt.event === 'token') {
                if (ttft === null) ttft = Math.round(performance.now() - playT0.current[idx])
                chars += evt.text.length
                setPlayTexts(p => { const n=[...p]; n[idx]=n[idx]+evt.text; return n })
              } else if (evt.event === 'done') {
                setPlayStats(p => { const n=[...p]; n[idx]={ ttft_ms:ttft??0, chars }; return n })
                setPlayDone(p => { const n=[...p]; n[idx]=true; return n })
              } else if (evt.event === 'error') {
                setPlayErrors(p => { const n=[...p]; n[idx]=evt.error; return n })
                setPlayDone(p => { const n=[...p]; n[idx]=true; return n })
              }
            } catch {}
          }
        }
      } catch (e) {
        setPlayErrors(p => { const n=[...p]; n[idx]=e.message; return n })
      } finally {
        setPlayDone(p => { const n=[...p]; n[idx]=true; return n })
      }
    }

    await Promise.all([runPanel(0), runPanel(1)])
    setPlayRunning(false)
  }

  // ── Derived data ───────────────────────────────────────────────────────────
  const resultList = Object.values(results)
  const hasResults = resultList.length > 0
  const tpsData  = resultList.map(r => ({ model:r.model.replace(/:.*/, ''), tps:r.tps_mean,  fill:r.color }))
  const ttftData = resultList.map(r => ({ model:r.model.replace(/:.*/, ''), ttft:r.ttft_mean, fill:r.color }))
  const radarData = CRITERIA.map(c => {
    const e = { criterion:c.label }
    for (const [n, s] of Object.entries(judgeScores)) e[n] = s[c.key] || 0
    return e
  })
  const statusColor = { spawning:'var(--yellow)', running:'var(--green)', error:'var(--orange)' }
  const streamCols = Math.min(selected.length, 3)

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100vh', overflow:'hidden' }}>

      {/* Header */}
      <header style={{ background:'var(--surface)', borderBottom:'1px solid var(--border)',
        padding:'0 20px', display:'flex', alignItems:'center', gap:20, height:50, flexShrink:0 }}>
        <div style={{ fontFamily:'var(--mono)', fontSize:15, letterSpacing:2, fontWeight:700, display:'flex', alignItems:'center', gap:6 }}>
          <span style={{ fontSize:16 }}>⚖</span>
          <span style={{ color:'var(--purple)' }}>Judge</span>
          <span style={{ color:'var(--cyan)' }}>GPT</span>
        </div>
        <nav style={{ display:'flex', gap:2, flex:1, overflowX:'auto' }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background: tab===t.id ? 'var(--surface2)' : 'transparent',
              border:'none', color: tab===t.id ? 'var(--text)' : 'var(--muted)',
              padding:'6px 12px', borderRadius:4, fontSize:13, whiteSpace:'nowrap',
              borderBottom: tab===t.id ? '2px solid var(--cyan)' : '2px solid transparent',
              transition:'all 0.15s',
            }}>{t.label}</button>
          ))}
        </nav>
        <div style={{ display:'flex', gap:6 }}>
          {Object.entries(containerStatus).map(([m, phase]) => (
            <div key={m} style={{ display:'flex', alignItems:'center', gap:5,
              background:'var(--surface2)', border:`1px solid ${statusColor[phase]||'var(--border)'}33`,
              borderRadius:4, padding:'2px 8px', fontSize:10, fontFamily:'var(--mono)' }}>
              <Dot color={statusColor[phase]||'var(--muted)'} pulse={phase==='spawning'} />
              <span style={{ color:'var(--muted)' }}>{m.replace(/:.*/, '')}</span>
            </div>
          ))}
        </div>
        <Tag color="var(--green)">live · ollama</Tag>
      </header>

      {/* Body */}
      <main style={{ flex:1, overflow:'auto', padding:20 }}>

        {/* ── RUN ── */}
        {tab === 'run' && (
          <div style={{ maxWidth:860, margin:'0 auto', display:'flex', flexDirection:'column', gap:20 }}>
            <SectionHeader>Configure Benchmark</SectionHeader>

            <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:18 }}>
              {/* ── Model directory header ── */}
              <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:16 }}>
                <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:1 }}>
                  Model Directory
                </div>
                <div style={{ display:'flex', gap:6, alignItems:'center' }}>
                  <span style={{ fontSize:10, color:'var(--green)', fontFamily:'var(--mono)' }}>
                    {localModels.length} installed
                  </span>
                  <span style={{ color:'var(--border2)' }}>·</span>
                  <span style={{ fontSize:10, color:'var(--muted)', fontFamily:'var(--mono)' }}>
                    {selected.length} selected
                  </span>
                </div>
              </div>

              {/* ── Installed section ── */}
              {allModels.filter(m => {
                const base = m.name.split(':')[0]
                return m.local || pullStatus[m.name]==='done' || localModels.some(lm => lm.name===m.name || lm.name.startsWith(base+':'))
              }).length > 0 && (
                <div style={{ marginBottom:14 }}>
                  <div style={{ fontSize:9, color:'var(--green)', fontFamily:'var(--mono)',
                    textTransform:'uppercase', letterSpacing:1.5, marginBottom:8,
                    display:'flex', alignItems:'center', gap:6 }}>
                    <span style={{ width:6, height:6, borderRadius:'50%', background:'var(--green)', display:'inline-block' }} />
                    Installed
                  </div>
                  <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
                    {allModels.filter(m => {
                      const base = m.name.split(':')[0]
                      return m.local || pullStatus[m.name]==='done' || localModels.some(lm => lm.name===m.name || lm.name.startsWith(base+':'))
                    }).map(m => {
                      const on  = selected.includes(m.name)
                      const col = m.color || '#888'
                      const phase = containerStatus[m.name]
                      return (
                        <div key={m.name} onClick={() => setSelected(p =>
                          p.includes(m.name) ? p.filter(x=>x!==m.name) : [...p,m.name]
                        )} style={{
                          display:'flex', alignItems:'center', gap:10,
                          padding:'9px 12px', borderRadius:6, cursor:'pointer',
                          background: on ? col+'14' : 'var(--surface2)',
                          border:`1px solid ${on ? col+'66' : 'var(--border)'}`,
                          transition:'all 0.15s',
                        }}>
                          {/* Color dot */}
                          <div style={{ width:8, height:8, borderRadius:'50%', flexShrink:0,
                            background:col, boxShadow:`0 0 6px ${col}88`,
                            opacity: on ? 1 : 0.5 }} />
                          {/* Name + meta */}
                          <div style={{ flex:1, minWidth:0 }}>
                            <span style={{ fontFamily:'var(--mono)', fontSize:12,
                              color: on ? col : 'var(--text)' }}>{m.name}</span>
                            <span style={{ fontSize:10, color:'var(--muted)', marginLeft:8, fontFamily:'var(--mono)' }}>
                              {m.size_gb ? `${m.size_gb} GB` : ''}
                              {m.custom ? ' · custom' : ''}
                              {m.local ? ' · local' : ''}
                            </span>
                          </div>
                          {/* Status */}
                          <div style={{ display:'flex', alignItems:'center', gap:6, flexShrink:0 }}>
                            {phase && <Dot color={statusColor[phase]||'var(--muted)'} pulse={phase==='spawning'} />}
                            <span style={{ fontSize:9, color:'var(--green)', fontFamily:'var(--mono)' }}>✓ ready</span>
                          </div>
                          {/* Select indicator */}
                          <div style={{ width:16, height:16, borderRadius:4, flexShrink:0,
                            background: on ? col : 'transparent',
                            border:`1.5px solid ${on ? col : 'var(--border2)'}`,
                            display:'flex', alignItems:'center', justifyContent:'center',
                            transition:'all 0.15s' }}>
                            {on && <span style={{ color:'#000', fontSize:9, fontWeight:'bold', lineHeight:1 }}>✓</span>}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* ── Available section ── */}
              {allModels.filter(m => {
                const base = m.name.split(':')[0]
                return !m.local && pullStatus[m.name]!=='done' && !localModels.some(lm => lm.name===m.name || lm.name.startsWith(base+':'))
              }).length > 0 && (
                <div>
                  <div style={{ fontSize:9, color:'var(--muted)', fontFamily:'var(--mono)',
                    textTransform:'uppercase', letterSpacing:1.5, marginBottom:8,
                    display:'flex', alignItems:'center', gap:6 }}>
                    <span style={{ width:6, height:6, borderRadius:'50%', background:'var(--border2)', display:'inline-block' }} />
                    Available — not downloaded
                  </div>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(240px,1fr))', gap:6 }}>
                    {allModels.filter(m => {
                      const base = m.name.split(':')[0]
                      return !m.local && pullStatus[m.name]!=='done' && !localModels.some(lm => lm.name===m.name || lm.name.startsWith(base+':'))
                    }).map(m => {
                      const on      = selected.includes(m.name)
                      const col     = m.color || '#888'
                      const phase   = containerStatus[m.name]
                      const pulling = pullStatus[m.name] === 'pulling'
                      const pullErr = pullStatus[m.name] === 'error'
                      const pct     = pullProgress[m.name] ?? 0
                      return (
                        <div key={m.name} style={{ borderRadius:6, overflow:'hidden',
                          border:`1px solid ${on ? col+'55' : 'var(--border)'}`,
                          background: on ? col+'0d' : 'var(--surface2)',
                          transition:'all 0.15s' }}>
                          {/* Top row — clickable for select */}
                          <div onClick={() => setSelected(p =>
                            p.includes(m.name) ? p.filter(x=>x!==m.name) : [...p,m.name]
                          )} style={{ display:'flex', alignItems:'center', gap:8,
                            padding:'9px 12px', cursor:'pointer' }}>
                            <div style={{ width:8, height:8, borderRadius:'50%', flexShrink:0,
                              background:col, opacity: on ? 0.9 : 0.35 }} />
                            <div style={{ flex:1, minWidth:0 }}>
                              <div style={{ fontFamily:'var(--mono)', fontSize:11,
                                color: on ? col : 'var(--muted)', overflow:'hidden',
                                textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{m.name}</div>
                              {m.size_gb && <div style={{ fontSize:9, color:'var(--border2)',
                                fontFamily:'var(--mono)' }}>{m.size_gb} GB</div>}
                            </div>
                            {phase && <Dot color={statusColor[phase]||'var(--muted)'} pulse={phase==='spawning'} />}
                            <div style={{ width:14, height:14, borderRadius:3, flexShrink:0,
                              background: on ? col : 'transparent',
                              border:`1.5px solid ${on ? col : 'var(--border2)'}`,
                              display:'flex', alignItems:'center', justifyContent:'center' }}>
                              {on && <span style={{ color:'#000', fontSize:8, fontWeight:'bold' }}>✓</span>}
                            </div>
                          </div>
                          {/* Pull status bar */}
                          <div style={{ padding:'0 12px 9px', borderTop:'1px solid var(--border)22' }}>
                            {pulling ? (
                              <>
                                <div style={{ height:2, background:'var(--border)', borderRadius:1, overflow:'hidden', marginBottom:4 }}>
                                  <div style={{ width:`${pct}%`, height:'100%', background:col, transition:'width 0.4s' }} />
                                </div>
                                <div style={{ fontSize:9, color:'var(--muted)', fontFamily:'var(--mono)',
                                  display:'flex', justifyContent:'space-between' }}>
                                  <span style={{ overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', maxWidth:130 }}>
                                    {pullMsg[m.name] || 'pulling…'}
                                  </span>
                                  <span>{pct}%</span>
                                </div>
                              </>
                            ) : pullErr ? (
                              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                                <span style={{ fontSize:9, color:'var(--orange)', fontFamily:'var(--mono)' }}>pull failed</span>
                                <span onClick={e => { e.stopPropagation(); pullModel(m.name) }}
                                  style={{ fontSize:9, cursor:'pointer', color:'var(--cyan)',
                                    fontFamily:'var(--mono)', textDecoration:'underline' }}>retry</span>
                              </div>
                            ) : (
                              <button onClick={e => { e.stopPropagation(); pullModel(m.name) }} style={{
                                fontSize:9, color:'var(--cyan)',
                                background:'transparent', border:'1px solid var(--cyan)44',
                                borderRadius:3, padding:'2px 10px', fontFamily:'var(--mono)',
                                cursor:'pointer', width:'100%',
                              }}>↓ pull to install</button>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* ── Add custom model ── */}
              <div style={{ marginTop:14, paddingTop:12, borderTop:'1px solid var(--border)', display:'flex', gap:8 }}>
                <input value={customModel} onChange={e => setCustomModel(e.target.value)}
                  onKeyDown={e => e.key==='Enter' && addCustomModel()}
                  placeholder="Add custom model tag (e.g. llama3.2:3b)"
                  style={{ flex:1, background:'var(--surface2)', border:'1px solid var(--border)',
                    borderRadius:6, padding:'7px 12px', color:'var(--text)',
                    fontFamily:'var(--mono)', fontSize:12, outline:'none' }} />
                <button onClick={addCustomModel} style={{
                  background:'var(--yellow)18', border:'1px solid var(--yellow)',
                  color:'var(--yellow)', padding:'7px 16px', borderRadius:6,
                  fontFamily:'var(--mono)', fontSize:12, cursor:'pointer',
                }}>+ Add</button>
              </div>
            </div>

            <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:18 }}>
              <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:1, marginBottom:10 }}>Prompt</div>
              <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={3}
                style={{ width:'100%', background:'var(--surface2)', border:'1px solid var(--border)',
                  borderRadius:6, padding:'10px 14px', color:'var(--text)',
                  fontFamily:'var(--sans)', fontSize:13, resize:'vertical', outline:'none' }} />
            </div>

            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:10 }}>
              <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:14 }}>
                <div style={{ fontSize:10, color:'var(--muted)', marginBottom:8 }}>RUNS / MODEL</div>
                <div style={{ display:'flex', gap:6 }}>
                  {[1,3,5].map(n => (
                    <button key={n} onClick={() => setNRuns(n)} style={{
                      background: nRuns===n ? 'var(--cyan)22' : 'var(--surface2)',
                      border:`1px solid ${nRuns===n ? 'var(--cyan)' : 'var(--border)'}`,
                      color: nRuns===n ? 'var(--cyan)' : 'var(--muted)',
                      padding:'4px 12px', borderRadius:4, fontFamily:'var(--mono)', fontSize:12,
                    }}>{n}</button>
                  ))}
                </div>
              </div>
              <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:14 }}>
                <div style={{ fontSize:10, color:'var(--muted)', marginBottom:8 }}>
                  MAX TOKENS {nTokens === -1 ? '— unlimited' : ''}
                </div>
                <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
                  {[256,512,1024,2048,4096,-1].map(n => (
                    <button key={n} onClick={() => setNTokens(n)} style={{
                      background: nTokens===n ? 'var(--purple)22' : 'var(--surface2)',
                      border:`1px solid ${nTokens===n ? 'var(--purple)' : 'var(--border)'}`,
                      color: nTokens===n ? 'var(--purple)' : 'var(--muted)',
                      padding:'4px 10px', borderRadius:4, fontFamily:'var(--mono)', fontSize:11,
                      cursor:'pointer',
                    }}>{n === -1 ? '∞' : n}</button>
                  ))}
                </div>
              </div>
              <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:14, display:'flex', flexDirection:'column', gap:10 }}>
                {[
                  { label:'Auto-Judge', val:autoJudge, set:setAutoJudge, col:'var(--purple)' },
                  { label:'Keep Alive', val:keepAlive, set:setKeepAlive, col:'var(--yellow)' },
                  { label:'Sequential', val:sequential, set:setSequential, col:'var(--orange)' },
                ].map(({ label, val, set, col }) => (
                  <div key={label} style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
                    <div>
                      <span style={{ fontSize:10, color:'var(--muted)' }}>{label}</span>
                      {label === 'Sequential' && (
                        <div style={{ fontSize:9, color:'var(--border2)', fontFamily:'var(--mono)', marginTop:1 }}>
                          one model at a time
                        </div>
                      )}
                    </div>
                    <div onClick={() => set(v => !v)} style={{
                      width:36, height:20, borderRadius:10, cursor:'pointer',
                      background: val ? col : 'var(--border)', position:'relative', transition:'background 0.2s',
                      boxShadow: val ? `0 0 8px ${col}66` : 'none',
                    }}>
                      <div style={{ position:'absolute', top:2, left: val ? 18 : 2,
                        width:16, height:16, borderRadius:'50%', background:'white', transition:'left 0.2s' }} />
                    </div>
                  </div>
                ))}
                {autoJudge && (
                  <div style={{ borderTop:'1px solid var(--border)', paddingTop:10, display:'flex', flexDirection:'column', gap:8 }}>
                    <div style={{ fontSize:10, color:'var(--purple)', letterSpacing:1 }}>JUDGE CONFIG</div>
                    <div>
                      <div style={{ fontSize:9, color:'var(--muted)', marginBottom:4 }}>MODEL</div>
                      <input
                        value={judgeModel}
                        onChange={e => { setJudgeModel(e.target.value); saveJudgeConfig(judgeSystemPrompt, e.target.value) }}
                        style={{ width:'100%', background:'var(--surface2)', border:'1px solid var(--border)',
                          color:'var(--fg)', fontFamily:'var(--mono)', fontSize:11, padding:'5px 8px',
                          borderRadius:4, boxSizing:'border-box' }}
                      />
                    </div>
                    <div>
                      <div style={{ fontSize:9, color:'var(--muted)', marginBottom:4 }}>SYSTEM PROMPT</div>
                      <textarea
                        value={judgeSystemPrompt}
                        onChange={e => { setJudgeSystemPrompt(e.target.value); saveJudgeConfig(e.target.value, judgeModel) }}
                        rows={6}
                        style={{ width:'100%', background:'var(--surface2)', border:'1px solid var(--border)',
                          color:'var(--fg)', fontFamily:'var(--mono)', fontSize:10, padding:'6px 8px',
                          borderRadius:4, resize:'vertical', boxSizing:'border-box', lineHeight:1.5 }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            <button onClick={runBenchmark} disabled={running || !selected.length} style={{
              background: running ? 'var(--surface2)' : 'linear-gradient(135deg,var(--cyan)18,var(--purple)18)',
              border:`1px solid ${running ? 'var(--border)' : 'var(--cyan)'}`,
              color: running ? 'var(--muted)' : 'var(--cyan)',
              padding:'13px', borderRadius:8, fontSize:14, fontFamily:'var(--mono)', letterSpacing:1,
              display:'flex', alignItems:'center', justifyContent:'center', gap:10,
              transition:'all 0.2s', width:'100%',
              boxShadow: (!running && selected.length) ? '0 0 24px var(--cyan)22' : 'none',
            }}>
              {running ? <><Spinner />{' Running benchmark…'}</> : '▶  Run Benchmark'}
            </button>

            {log.length > 0 && (
              <div ref={logRef} style={{ background:'var(--surface)', border:'1px solid var(--border)',
                borderRadius:8, padding:14, fontFamily:'var(--mono)', fontSize:11,
                maxHeight:240, overflowY:'auto', animation:'slideIn 0.2s ease' }}>
                {log.map((l,i) => (
                  <div key={i} style={{
                    color: l.type==='error' ? 'var(--orange)' : l.type==='green' ? 'var(--green)'
                         : l.type==='cyan' ? 'var(--cyan)' : l.type==='purple' ? 'var(--purple)'
                         : l.type==='yellow' ? 'var(--yellow)' : 'var(--muted)',
                    lineHeight:1.9,
                  }}>
                    <span style={{ color:'var(--border2)', marginRight:10 }}>{l.ts}</span>
                    {l.msg}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── METRICS ── */}
        {tab === 'metrics' && (
          <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
            {!hasResults
              ? <div style={{ color:'var(--muted)', textAlign:'center', marginTop:80, fontFamily:'var(--mono)' }}>No results yet — run a benchmark first.</div>
              : (() => {
                  // Prompt scope analysis
                  const scopePrompt = lastRunConfig.prompt || prompt
                  const scopeWords = scopePrompt.trim().split(/\s+/).filter(Boolean).length
                  const scopeChars = scopePrompt.length
                  const lower = scopePrompt.toLowerCase()
                  const scopeType =
                    /\b(write|create|generate|compose|draft)\b/.test(lower) ? 'Creative' :
                    /\b(code|function|program|implement|algorithm|script)\b/.test(lower) ? 'Technical' :
                    /\b(compare|difference|vs|versus|which is better)\b/.test(lower) ? 'Comparative' :
                    /\b(list|enumerate|steps|how to|procedure)\b/.test(lower) ? 'Instructional' :
                    /\b(explain|describe|what is|why|how does)\b/.test(lower) ? 'Explanatory' :
                    scopePrompt.includes('?') ? 'Question' : 'General'
                  const scopeComplexity = scopeWords < 20 ? 'Simple' : scopeWords < 60 ? 'Moderate' : 'Complex'
                  const scopeTokLabel = (lastRunConfig.nTokens || nTokens) === -1
                    ? '∞ unlimited' : `${lastRunConfig.nTokens || nTokens} max`

                  return <>
                  {/* ── Scope card ── */}
                  <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:16 }}>
                    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:12 }}>
                      <SectionHeader>Benchmark Scope</SectionHeader>
                      <div style={{ display:'flex', gap:8 }}>
                        <Tag color="var(--purple)">{scopeType}</Tag>
                        <Tag color="var(--muted)">{scopeComplexity}</Tag>
                      </div>
                    </div>
                    <div style={{ display:'flex', gap:16, alignItems:'flex-start', flexWrap:'wrap' }}>
                      {/* Prompt quote */}
                      <div style={{ flex:'1 1 340px', background:'var(--surface2)',
                        borderLeft:'2px solid var(--purple)', borderRadius:'0 6px 6px 0',
                        padding:'10px 14px', fontSize:13, color:'var(--text)', lineHeight:1.65,
                        fontStyle:'italic', fontFamily:'var(--sans)' }}>
                        "{scopePrompt}"
                      </div>
                      {/* Stats chips */}
                      <div style={{ display:'flex', flexWrap:'wrap', gap:8, alignContent:'flex-start', paddingTop:4 }}>
                        {[
                          { label:'Words',     val:scopeWords,                            col:'var(--cyan)'   },
                          { label:'Chars',     val:scopeChars,                            col:'var(--cyan)'   },
                          { label:'Runs',      val:lastRunConfig.nRuns || nRuns,          col:'var(--yellow)' },
                          { label:'Max tokens',val:scopeTokLabel,                         col:'var(--purple)' },
                          { label:'Models',    val:resultList.length,                     col:'var(--green)'  },
                        ].map(({ label, val, col }) => (
                          <div key={label} style={{ background:'var(--surface2)',
                            border:`1px solid ${col}33`, borderRadius:6,
                            padding:'6px 12px', minWidth:70 }}>
                            <div style={{ fontSize:9, color:'var(--muted)', textTransform:'uppercase',
                              letterSpacing:1, marginBottom:3 }}>{label}</div>
                            <div style={{ fontFamily:'var(--mono)', fontSize:13, color:col }}>{val}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))', gap:10 }}>
                    <StatCard label="Models" value={resultList.length} color="var(--cyan)" />
                    <StatCard label="Fastest" value={Math.max(...resultList.map(r=>r.tps_mean)).toFixed(1)+'t/s'}
                      sub={[...resultList].sort((a,b)=>b.tps_mean-a.tps_mean)[0]?.model.replace(/:.*/, '')}
                      color="var(--green)" />
                    <StatCard label="Best TTFT"
                      value={Math.min(...resultList.map(r=>r.ttft_mean)).toFixed(0)+'ms'}
                      sub={[...resultList].sort((a,b)=>a.ttft_mean-b.ttft_mean)[0]?.model.replace(/:.*/, '')}
                      color="var(--yellow)" />
                    {Object.keys(judgeScores).length > 0 && (() => {
                      const avg = s => CRITERIA.reduce((a,c) => a+(s[c.key]||0), 0)/CRITERIA.length
                      const [name, score] = Object.entries(judgeScores).sort((a,b)=>avg(b[1])-avg(a[1]))[0]
                      return <StatCard label="Best Quality" value={avg(score).toFixed(2)+'/5'} sub={name.replace(/:.*/, '')} color="var(--purple)" />
                    })()}
                  </div>

                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16 }}>
                    <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:18 }}>
                      <SectionHeader>Tokens Per Second</SectionHeader>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={tpsData} barCategoryGap="40%">
                          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
                          <XAxis dataKey="model" tick={{ fill:'#7d8590', fontFamily:'Space Mono', fontSize:10 }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fill:'#7d8590', fontFamily:'Space Mono', fontSize:10 }} axisLine={false} tickLine={false} />
                          <Tooltip content={<DarkTooltip />} />
                          <Bar dataKey="tps" radius={[4,4,0,0]} label={{ position:'top', fill:'#7d8590', fontFamily:'Space Mono', fontSize:9 }}>
                            {tpsData.map((e,i) => <rect key={i} fill={e.fill} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:18 }}>
                      <SectionHeader>Time to First Token (ms)</SectionHeader>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={ttftData} barCategoryGap="40%">
                          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
                          <XAxis dataKey="model" tick={{ fill:'#7d8590', fontFamily:'Space Mono', fontSize:10 }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fill:'#7d8590', fontFamily:'Space Mono', fontSize:10 }} axisLine={false} tickLine={false} />
                          <Tooltip content={<DarkTooltip />} />
                          <Bar dataKey="ttft" radius={[4,4,0,0]}>
                            {ttftData.map((e,i) => <rect key={i} fill={e.fill} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {Object.keys(judgeScores).length > 0 && (
                    <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:18 }}>
                      <SectionHeader>Quality — LLM Judge</SectionHeader>
                      <div style={{ display:'flex', gap:20, alignItems:'flex-start' }}>
                        {/* Radar chart */}
                        <div style={{ flex:'1 1 52%', minWidth:0 }}>
                          <ResponsiveContainer width="100%" height={280}>
                            <RadarChart data={radarData}>
                              <PolarGrid stroke="#21262d" />
                              <PolarAngleAxis dataKey="criterion" tick={{ fill:'#7d8590', fontFamily:'Space Mono', fontSize:10 }} />
                              <PolarRadiusAxis domain={[0,5]} tick={{ fill:'#7d8590', fontSize:9 }} />
                              {Object.keys(judgeScores).map(name => (
                                <Radar key={name} name={name} dataKey={name}
                                  stroke={results[name]?.color || '#888'}
                                  fill={results[name]?.color || '#888'} fillOpacity={0.12} />
                              ))}
                              <Legend formatter={v => <span style={{ fontFamily:'Space Mono', fontSize:10, color:results[v]?.color||'#888' }}>{v}</span>} />
                              <Tooltip content={<DarkTooltip />} />
                            </RadarChart>
                          </ResponsiveContainer>
                        </div>
                        {/* Scores table */}
                        <div style={{ flex:'1 1 44%', minWidth:0 }}>
                          <table style={{ width:'100%', borderCollapse:'collapse', fontFamily:'var(--mono)', fontSize:11 }}>
                            <thead>
                              <tr>
                                <th style={{ textAlign:'left', padding:'5px 8px', color:'var(--muted)', fontWeight:400, borderBottom:'1px solid var(--border)', fontSize:10 }}>Criterion</th>
                                {Object.keys(judgeScores).map(name => (
                                  <th key={name} style={{ textAlign:'center', padding:'5px 8px', color:results[name]?.color||'var(--cyan)', fontWeight:600, borderBottom:'1px solid var(--border)', fontSize:10 }}>
                                    {name.replace(/:.*/, '')}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {CRITERIA.map((c, i) => (
                                <tr key={c.key} style={{ background: i%2===0 ? 'transparent' : 'var(--surface2)' }}>
                                  <td style={{ padding:'6px 8px', color:'var(--muted)' }}>{c.icon} {c.label}</td>
                                  {Object.entries(judgeScores).map(([name, score]) => (
                                    <td key={name} style={{ textAlign:'center', padding:'6px 8px', color:results[name]?.color||'var(--fg)', fontWeight:500 }}>
                                      {score[c.key] ?? '—'}<span style={{ color:'var(--muted)', fontSize:9 }}>/5</span>
                                    </td>
                                  ))}
                                </tr>
                              ))}
                              <tr style={{ borderTop:'1px solid var(--border)' }}>
                                <td style={{ padding:'7px 8px', color:'var(--muted)', fontWeight:600 }}>⌀ Average</td>
                                {Object.entries(judgeScores).map(([name, score]) => {
                                  const avg = CRITERIA.reduce((s, c) => s + (score[c.key] || 0), 0) / CRITERIA.length
                                  return (
                                    <td key={name} style={{ textAlign:'center', padding:'7px 8px', color:results[name]?.color||'var(--fg)', fontWeight:700 }}>
                                      {avg.toFixed(1)}<span style={{ color:'var(--muted)', fontSize:9 }}>/5</span>
                                    </td>
                                  )
                                })}
                              </tr>
                            </tbody>
                          </table>
                          {/* Reasoning snippets */}
                          <div style={{ marginTop:12, display:'flex', flexDirection:'column', gap:6 }}>
                            {Object.entries(judgeScores).map(([name, score]) => score.reasoning && (
                              <div key={name} style={{ padding:'7px 10px', background:'var(--surface2)', borderLeft:`2px solid ${results[name]?.color||'var(--border)'}`, borderRadius:'0 4px 4px 0', fontSize:10, color:'var(--muted)', lineHeight:1.5 }}>
                                <span style={{ color:results[name]?.color||'var(--cyan)', fontWeight:600 }}>{name.replace(/:.*/, '')} </span>
                                {score.reasoning}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {resultList[0]?.runs?.length > 1 && (
                    <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:18 }}>
                      <SectionHeader>TPS per Run — Variance</SectionHeader>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart>
                          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                          <XAxis dataKey="run" type="number" domain={[1,nRuns]}
                            tick={{ fill:'#7d8590', fontFamily:'Space Mono', fontSize:10 }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fill:'#7d8590', fontFamily:'Space Mono', fontSize:10 }} axisLine={false} tickLine={false} />
                          <Tooltip content={<DarkTooltip />} />
                          {resultList.map(r => (
                            <Line key={r.model} data={r.runs} dataKey="tps" name={r.model}
                              stroke={r.color} strokeWidth={2} dot={{ r:3, fill:r.color }} type="monotone" />
                          ))}
                          <Legend formatter={v => <span style={{ fontFamily:'Space Mono', fontSize:10 }}>{v}</span>} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* ── Export PDF button ── */}
                  <div style={{ display:'flex', justifyContent:'flex-end' }}>
                    <button onClick={exportPDF} style={{
                      background:'linear-gradient(135deg,var(--purple)18,var(--cyan)18)',
                      border:'1px solid var(--purple)',
                      color:'var(--purple)', padding:'9px 20px', borderRadius:6,
                      fontFamily:'var(--mono)', fontSize:12, letterSpacing:0.5,
                      display:'flex', alignItems:'center', gap:8, cursor:'pointer',
                      boxShadow:'0 0 16px var(--purple)22',
                    }}>
                      ↓ Export PDF Report
                    </button>
                  </div>
                  </>
              })()}
          </div>
        )}

        {/* ── RESPONSES ── */}
        {tab === 'responses' && (
          <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
            {!hasResults
              ? <div style={{ color:'var(--muted)', textAlign:'center', marginTop:80, fontFamily:'var(--mono)' }}>No results yet.</div>
              : Object.entries(results).map(([name, r]) => {
                  const col = r.color || '#888'
                  const j = judgeScores[name]
                  const h = humanScores[name]
                  const wc = r.response?.split(' ').length || 0
                  return (
                    <div key={name} style={{ background:'var(--surface)',
                      border:'1px solid var(--border)', borderLeft:`3px solid ${col}`,
                      borderRadius:8, padding:18, animation:'slideIn 0.3s ease' }}>
                      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:12 }}>
                        <div style={{ display:'flex', alignItems:'center', gap:8, flexWrap:'wrap' }}>
                          <span style={{ fontFamily:'var(--mono)', fontSize:12, color:col }}>{name}</span>
                          <Tag color="var(--cyan)">{r.tps_mean} tok/s</Tag>
                          <Tag color="var(--yellow)">{r.ttft_mean}ms</Tag>
                          <Tag color="var(--muted)">{wc}w</Tag>
                        </div>
                        {!j && (
                          <button onClick={() => runJudge(name)} disabled={judging[name]} style={{
                            background:'var(--purple)18', border:'1px solid var(--purple)',
                            color:'var(--purple)', padding:'5px 12px', borderRadius:5,
                            fontFamily:'var(--mono)', fontSize:11,
                            display:'flex', alignItems:'center', gap:5,
                          }}>
                            {judging[name] ? <><Spinner size={11} color="var(--purple)" /> judging…</> : '⚖ Judge'}
                          </button>
                        )}
                      </div>
                      <div style={{ background:'var(--surface2)', borderRadius:6, padding:'10px 14px',
                        fontSize:13, lineHeight:1.75, color:'var(--text)', marginBottom: j ? 14 : 0 }}>
                        {r.response || '—'}
                      </div>
                      {j && (
                        <div style={{ marginTop:14, display:'grid', gridTemplateColumns:'1fr 1fr', gap:16 }}>
                          <div>
                            <div style={{ fontSize:10, color:'var(--purple)', textTransform:'uppercase', letterSpacing:1, marginBottom:10 }}>
                              LLM Judge
                            </div>
                            {CRITERIA.map(c => (
                              <div key={c.key} style={{ display:'flex', alignItems:'center', gap:8, marginBottom:7 }}>
                                <span style={{ width:18, fontSize:13 }}>{c.icon}</span>
                                <span style={{ width:68, fontSize:11, color:'var(--muted)' }}>{c.label}</span>
                                <div style={{ flex:1 }}><ScoreBar value={j[c.key]} color={col} /></div>
                              </div>
                            ))}
                          </div>
                          <div>
                            {j.reasoning && (
                              <div style={{ background:'var(--surface2)', borderRadius:6, padding:'10px 12px',
                                fontSize:12, color:'var(--muted)', lineHeight:1.7, fontStyle:'italic', marginBottom:14 }}>
                                "{j.reasoning}"
                              </div>
                            )}
                            <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:1, marginBottom:8 }}>Your Score</div>
                            <Stars value={h||0} onChange={s => setHuman(name, s)} color={col} />
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })
            }
          </div>
        )}

        {/* ── OVERALL ── */}
        {tab === 'overall' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            {leaderboard.length === 0
              ? <div style={{ color:'var(--muted)', textAlign:'center', marginTop:80, fontFamily:'var(--mono)' }}>
                  {hasResults ? 'Run judge to see combined leaderboard.' : 'No results yet.'}
                </div>
              : <>
                  <div style={{ fontSize:11, color:'var(--muted)', marginBottom:4 }}>
                    Combined = TPS×35% + TTFT×15% + Quality×50%
                  </div>
                  {leaderboard.map((e, idx) => {
                    const col = results[e.model]?.color || '#888'
                    return (
                      <div key={e.model} style={{ background:'var(--surface)',
                        border:'1px solid var(--border)', borderLeft:`3px solid ${col}`,
                        borderRadius:8, padding:'14px 18px', animation:'slideIn 0.3s ease' }}>
                        <div style={{ display:'flex', alignItems:'center', gap:14, marginBottom:12 }}>
                          <span style={{ fontFamily:'var(--mono)', fontSize:18,
                            color: idx===0?'var(--yellow)':idx===1?'#c0c0c0':idx===2?'#cd7f32':'var(--muted)',
                            minWidth:26 }}>#{e.rank}</span>
                          <span style={{ fontFamily:'var(--mono)', fontSize:13, color:col, flex:1 }}>{e.model}</span>
                          <div style={{ textAlign:'right' }}>
                            <div style={{ fontFamily:'var(--mono)', fontSize:22, color:col,
                              textShadow:`0 0 16px ${col}88`, lineHeight:1 }}>
                              {e.combined_score?.toFixed(1) ?? '—'}
                            </div>
                            <div style={{ fontSize:10, color:'var(--muted)' }}>/100</div>
                          </div>
                        </div>
                        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:10 }}>
                          {[
                            { label:'⚡ TPS',     val:e.tps_score,     w:'35%', col:'var(--cyan)' },
                            { label:'🕐 TTFT',    val:e.ttft_score,    w:'15%', col:'var(--yellow)' },
                            { label:'🧠 Quality', val:e.quality_score, w:'50%', col:'var(--purple)' },
                          ].map(seg => (
                            <div key={seg.label}>
                              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
                                <span style={{ fontSize:10, color:'var(--muted)' }}>{seg.label}</span>
                                <span style={{ fontFamily:'var(--mono)', fontSize:10, color:seg.col }}>
                                  {seg.val?.toFixed(1) ?? '—'} <span style={{ color:'var(--border2)' }}>×{seg.w}</span>
                                </span>
                              </div>
                              <div style={{ height:4, background:'var(--border)', borderRadius:2 }}>
                                {seg.val != null && <div style={{ width:`${seg.val}%`, height:'100%',
                                  background:seg.col, borderRadius:2, transition:'width 0.8s ease',
                                  boxShadow:`0 0 6px ${seg.col}88` }} />}
                              </div>
                            </div>
                          ))}
                        </div>
                        {!e.quality_score && (
                          <div style={{ marginTop:10, fontSize:11, color:'var(--muted)', fontStyle:'italic' }}>
                            Quality pending — run judge or add human score
                          </div>
                        )}
                      </div>
                    )
                  })}
                  <div style={{ display:'flex', gap:10, marginTop:4 }}>
                    <a href={`${API}/export/json`} style={{ fontSize:11, color:'var(--cyan)',
                      background:'var(--surface)', border:'1px solid var(--border)',
                      padding:'6px 14px', borderRadius:5, fontFamily:'var(--mono)' }}>
                      ↓ export json
                    </a>
                    <a href={`${API}/export/csv`} style={{ fontSize:11, color:'var(--green)',
                      background:'var(--surface)', border:'1px solid var(--border)',
                      padding:'6px 14px', borderRadius:5, fontFamily:'var(--mono)' }}>
                      ↓ export csv
                    </a>
                  </div>
                </>
            }
          </div>
        )}

        {/* ── STREAM ── */}
        {tab === 'stream' && (
          <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
            {/* Header row */}
            <div style={{ display:'flex', alignItems:'flex-start', gap:12 }}>
              <div style={{ flex:1, background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:14 }}>
                <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:1, marginBottom:8 }}>
                  Prompt · uses models & prompt from Run tab
                </div>
                <div style={{ fontFamily:'var(--sans)', fontSize:13, color:'var(--text)',
                  background:'var(--surface2)', borderRadius:6, padding:'8px 12px', lineHeight:1.6 }}>
                  {prompt}
                </div>
              </div>
              <button onClick={streamRunning ? stopStream : runStream} disabled={!selected.length} style={{
                background: streamRunning ? 'var(--orange)18' : 'var(--cyan)18',
                border:`1px solid ${streamRunning ? 'var(--orange)' : 'var(--cyan)'}`,
                color: streamRunning ? 'var(--orange)' : 'var(--cyan)',
                padding:'14px 22px', borderRadius:8, fontFamily:'var(--mono)', fontSize:12,
                display:'flex', alignItems:'center', gap:8, whiteSpace:'nowrap', flexShrink:0,
                cursor: selected.length ? 'pointer' : 'not-allowed',
                boxShadow: selected.length ? (streamRunning ? '0 0 16px var(--orange)22' : '0 0 16px var(--cyan)22') : 'none',
              }}>
                {streamRunning ? <><Spinner color="var(--orange)" /> ⏹ Stop</> : '▶ Stream Live'}
              </button>
            </div>

            {!selected.length ? (
              <div style={{ color:'var(--muted)', textAlign:'center', marginTop:60, fontFamily:'var(--mono)' }}>
                Select models on the ⚡ Run tab first.
              </div>
            ) : (
              <div style={{
                display:'grid',
                gridTemplateColumns:`repeat(${streamCols}, 1fr)`,
                gap:12,
              }}>
                {selected.map(model => {
                  const col  = allModels.find(m => m.name === model)?.color || '#888'
                  const text = streamTexts[model] || ''
                  const stats  = streamStats[model]
                  const status = streamStatus[model]
                  const isStreaming = status === 'streaming'
                  const isDone = status === 'done'
                  const isError = status === 'error'

                  return (
                    <div key={model} style={{
                      background:'var(--surface)',
                      border:`1px solid ${isStreaming ? col+'55' : isDone ? col+'33' : 'var(--border)'}`,
                      borderTop:`2px solid ${col}`,
                      borderRadius:8, padding:16,
                      display:'flex', flexDirection:'column', gap:10,
                      boxShadow: isStreaming ? `0 0 24px ${col}18` : 'none',
                      transition:'border-color 0.4s, box-shadow 0.4s',
                      minHeight: 280,
                    }}>
                      {/* Panel header */}
                      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
                        <div style={{ display:'flex', alignItems:'center', gap:7 }}>
                          {isStreaming && <Dot color={col} pulse />}
                          {isDone     && <span style={{ fontSize:11, color:'var(--green)' }}>✓</span>}
                          {isError    && <span style={{ fontSize:11, color:'var(--orange)' }}>✗</span>}
                          {!status && <span style={{ fontSize:11, color:'var(--border2)' }}>○</span>}
                          <span style={{ fontFamily:'var(--mono)', fontSize:12, color:col }}>{model}</span>
                        </div>
                        <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                          {stats && <><Tag color="var(--cyan)">{stats.tps} t/s</Tag><Tag color="var(--yellow)">{stats.ttft_ms}ms TTFT</Tag></>}
                          {isStreaming && !stats && <Spinner size={12} color={col} />}
                          {status === 'spawning' && <span style={{ fontSize:10, color:'var(--yellow)', fontFamily:'var(--mono)' }}>spawning…</span>}
                        </div>
                      </div>

                      {/* Live text */}
                      <div
                        ref={el => streamPanelRefs.current[model] = el}
                        style={{
                          flex:1, background:'var(--surface2)', borderRadius:6,
                          padding:'10px 14px', fontSize:13, lineHeight:1.85,
                          color: text ? 'var(--text)' : 'var(--muted)',
                          overflowY:'auto', wordBreak:'break-word',
                          minHeight:200, maxHeight:420,
                          fontFamily:'var(--sans)',
                        }}>
                        {isError
                          ? <span style={{ color:'var(--orange)', fontFamily:'var(--mono)', fontSize:11 }}>{streamErrors[model] || 'Error spawning container'}</span>
                          : text
                            ? <>{text}{isStreaming && <span style={{ color:col, fontWeight:'bold' }}>▋</span>}</>
                            : isStreaming
                              ? <span style={{ color:col }}>▋</span>
                              : <span style={{ color:'var(--border2)', fontFamily:'var(--mono)', fontSize:11 }}>
                                  {streamRunning ? 'waiting…' : 'press ▶ Stream Live'}
                                </span>
                        }
                      </div>

                      {/* Token counter */}
                      {text && (
                        <div style={{ fontSize:10, color:'var(--muted)', fontFamily:'var(--mono)',
                          display:'flex', gap:12 }}>
                          <span>{text.split(/\s+/).filter(Boolean).length} words</span>
                          {stats && <span>{stats.tokens} tokens</span>}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {/* ── PLAYGROUND ── */}
        {tab === 'playground' && (
          <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
            <div style={{ fontSize:10, color:'var(--muted)', fontFamily:'var(--mono)', textTransform:'uppercase', letterSpacing:2 }}>
              OpenAPI Endpoint Playground — compare any two models side by side
            </div>

            {/* Two config + response panels */}
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
              {[0,1].map(idx => {
                const panel = playPanels[idx]
                const accentColor = idx === 0 ? 'var(--cyan)' : 'var(--purple)'
                const text = playTexts[idx]
                const err  = playErrors[idx]
                const done = playDone[idx]
                const stat = playStats[idx]
                return (
                  <div key={idx} style={{
                    background:'var(--surface)', border:`1px solid var(--border)`,
                    borderTop:`2px solid ${accentColor}`, borderRadius:8, padding:16,
                    display:'flex', flexDirection:'column', gap:12,
                  }}>
                    {/* Panel label + preset picker */}
                    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
                      <span style={{ fontFamily:'var(--mono)', fontSize:11, color:accentColor, textTransform:'uppercase', letterSpacing:1 }}>
                        {panel.label}
                      </span>
                      <select
                        onChange={e => {
                          const preset = PLAYGROUND_PRESETS[parseInt(e.target.value)]
                          if (!preset) return
                          setPlayPanels(p => {
                            const n = [...p]
                            n[idx] = { ...n[idx], endpoint:preset.endpoint, model:preset.model }
                            return n
                          })
                        }}
                        style={{ background:'var(--surface2)', border:'1px solid var(--border)',
                          color:'var(--muted)', borderRadius:4, padding:'3px 8px',
                          fontFamily:'var(--mono)', fontSize:10, outline:'none' }}>
                        <option value="">— preset —</option>
                        {PLAYGROUND_PRESETS.map((p,i) => <option key={i} value={i}>{p.label}</option>)}
                      </select>
                    </div>

                    {/* Config fields */}
                    {[
                      { label:'Endpoint URL', key:'endpoint', placeholder:'http://localhost:11434/v1/chat/completions', type:'text' },
                      { label:'Model',        key:'model',    placeholder:'mistral:7b',                                type:'text' },
                      { label:'API Key',      key:'apiKey',   placeholder:'sk-… (leave blank for local)',             type:'password' },
                      { label:'System Prompt',key:'sysPrompt',placeholder:'Optional system context…',                type:'text' },
                    ].map(({ label, key, placeholder, type }) => (
                      <div key={key}>
                        <div style={{ fontSize:10, color:'var(--muted)', marginBottom:4 }}>{label}</div>
                        <input
                          type={type}
                          value={panel[key]}
                          onChange={e => setPlayPanels(p => {
                            const n=[...p]; n[idx]={...n[idx],[key]:e.target.value}; return n
                          })}
                          placeholder={placeholder}
                          style={{ width:'100%', background:'var(--surface2)',
                            border:`1px solid var(--border)`, borderRadius:5,
                            padding:'6px 10px', color:'var(--text)',
                            fontFamily:'var(--mono)', fontSize:11, outline:'none',
                          }}
                        />
                      </div>
                    ))}

                    {/* Response area header */}
                    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
                      <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:1 }}>Response</div>
                      {stat && (
                        <div style={{ display:'flex', gap:6 }}>
                          <Tag color="var(--yellow)">{stat.ttft_ms}ms TTFT</Tag>
                          <Tag color="var(--muted)">{stat.chars} chars</Tag>
                        </div>
                      )}
                      {playRunning && !done && <Spinner size={12} color={accentColor} />}
                      {done && text && <span style={{ fontSize:11, color:'var(--green)' }}>✓ done</span>}
                    </div>

                    {/* Response text */}
                    <div
                      ref={el => playPanelRefs.current[idx] = el}
                      style={{
                        flex:1, background:'var(--surface2)', borderRadius:6,
                        padding:'10px 14px', fontSize:13, lineHeight:1.85,
                        color: (text || err) ? 'var(--text)' : 'var(--muted)',
                        minHeight:200, maxHeight:380, overflowY:'auto',
                        wordBreak:'break-word', fontFamily:'var(--sans)',
                        border:`1px solid ${err ? 'var(--orange)33' : 'var(--border)'}`,
                      }}>
                      {err
                        ? <span style={{ color:'var(--orange)', fontFamily:'var(--mono)', fontSize:11 }}>
                            Error: {err}
                          </span>
                        : text
                          ? <>{text}{playRunning && !done && <span style={{ color:accentColor, fontWeight:'bold' }}>▋</span>}</>
                          : <span style={{ color:'var(--border2)', fontFamily:'var(--mono)', fontSize:11 }}>
                              {playRunning ? 'waiting for first token…' : 'response will appear here'}
                            </span>
                      }
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Shared prompt + Compare button */}
            <div style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:8, padding:16 }}>
              <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:1, marginBottom:8 }}>
                Prompt — sent to both endpoints simultaneously
              </div>
              <div style={{ display:'flex', gap:12, alignItems:'flex-end' }}>
                <textarea
                  value={playPrompt}
                  onChange={e => setPlayPrompt(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runPlayground() }}
                  rows={3}
                  placeholder="Enter a prompt to compare across both endpoints… (⌘↵ to send)"
                  style={{ flex:1, background:'var(--surface2)', border:'1px solid var(--border)',
                    borderRadius:6, padding:'10px 14px', color:'var(--text)',
                    fontFamily:'var(--sans)', fontSize:13, resize:'vertical', outline:'none' }}
                />
                <button
                  onClick={runPlayground}
                  disabled={playRunning || !playPrompt.trim()}
                  style={{
                    background: playRunning
                      ? 'var(--surface2)'
                      : 'linear-gradient(135deg, var(--cyan)18, var(--purple)18)',
                    border:`1px solid ${playRunning ? 'var(--border)' : 'var(--purple)'}`,
                    color: playRunning ? 'var(--muted)' : 'var(--purple)',
                    padding:'12px 24px', borderRadius:8,
                    fontFamily:'var(--mono)', fontSize:13, letterSpacing:1,
                    display:'flex', alignItems:'center', gap:8, whiteSpace:'nowrap',
                    boxShadow: (!playRunning && playPrompt.trim()) ? '0 0 20px var(--purple)22' : 'none',
                    transition:'all 0.2s',
                  }}>
                  {playRunning
                    ? <><Spinner color="var(--purple)" /> Comparing…</>
                    : '⚡ Compare'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── HISTORY ── */}
        {tab === 'history' && (
          <div style={{ display:'flex', flexDirection:'column', gap:14, maxWidth:900, margin:'0 auto' }}>
            <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
              <SectionHeader>Benchmark History</SectionHeader>
              {history.length > 0 && (
                <button onClick={() => {
                  fetch(`${API}/history`, { method:'DELETE' }).then(() => setHistory([]))
                }} style={{
                  fontSize:10, color:'var(--orange)', background:'transparent',
                  border:'1px solid var(--orange)44', borderRadius:4,
                  padding:'3px 10px', fontFamily:'var(--mono)', cursor:'pointer',
                }}>✕ clear all</button>
              )}
            </div>

            {historyLoading ? (
              <div style={{ display:'flex', justifyContent:'center', marginTop:60 }}>
                <Spinner size={20} />
              </div>
            ) : history.length === 0 ? (
              <div style={{ color:'var(--muted)', textAlign:'center', marginTop:80, fontFamily:'var(--mono)' }}>
                No history yet — completed benchmarks appear here automatically.
              </div>
            ) : history.map((entry) => {
              const date     = new Date(entry.timestamp * 1000)
              const topModel = entry.leaderboard?.[0]
              const topCol   = entry.results?.[topModel?.model]?.color || 'var(--cyan)'
              const hasJudge = Object.keys(entry.judge_scores || {}).length > 0
              return (
                <div key={entry.id} style={{ background:'var(--surface)',
                  border:'1px solid var(--border)', borderRadius:8, padding:16,
                  animation:'slideIn 0.2s ease' }}>

                  {/* Header row */}
                  <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', gap:12, marginBottom:10 }}>
                    <div style={{ flex:1, minWidth:0 }}>
                      <div style={{ fontFamily:'var(--mono)', fontSize:10, color:'var(--muted)', marginBottom:4 }}>
                        {date.toLocaleString()} · {entry.models?.length} model{entry.models?.length !== 1 ? 's' : ''} · {entry.n_runs} run{entry.n_runs !== 1 ? 's' : ''} · {entry.n_tokens === -1 ? '∞' : entry.n_tokens} tok
                      </div>
                      <div style={{ fontSize:12, color:'var(--text)', fontStyle:'italic',
                        overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                        "{entry.prompt}"
                      </div>
                    </div>
                    <button onClick={() => restoreHistory(entry)} style={{
                      flexShrink:0, background:'var(--cyan)18', border:'1px solid var(--cyan)',
                      color:'var(--cyan)', padding:'6px 14px', borderRadius:5,
                      fontFamily:'var(--mono)', fontSize:11, cursor:'pointer',
                    }}>↺ restore</button>
                  </div>

                  {/* Model tags */}
                  <div style={{ display:'flex', gap:6, flexWrap:'wrap', marginBottom:10 }}>
                    {(entry.models || []).map(m => {
                      const col = entry.results?.[m]?.color || '#888'
                      const tps = entry.results?.[m]?.tps_mean
                      return (
                        <span key={m} style={{ fontFamily:'var(--mono)', fontSize:10,
                          color:col, background:col+'18', border:`1px solid ${col}44`,
                          borderRadius:3, padding:'2px 8px', display:'flex', alignItems:'center', gap:5 }}>
                          {m}
                          {tps != null && <span style={{ color:'var(--muted)' }}>{tps}t/s</span>}
                        </span>
                      )
                    })}
                  </div>

                  {/* Winner + scores */}
                  {topModel && (
                    <div style={{ display:'flex', alignItems:'center', gap:14,
                      background:'var(--surface2)', borderRadius:6, padding:'8px 12px' }}>
                      <span style={{ fontSize:14 }}>🏆</span>
                      <span style={{ fontFamily:'var(--mono)', fontSize:11, color:topCol }}>{topModel.model}</span>
                      <div style={{ display:'flex', gap:8, marginLeft:'auto' }}>
                        <Tag color="var(--cyan)">{topModel.tps_mean} t/s</Tag>
                        <Tag color="var(--yellow)">{topModel.ttft_mean}ms</Tag>
                        {topModel.combined_score != null && (
                          <Tag color="var(--purple)">{topModel.combined_score.toFixed(1)}/100</Tag>
                        )}
                        {!hasJudge && (
                          <span style={{ fontSize:10, color:'var(--muted)', fontFamily:'var(--mono)',
                            alignSelf:'center' }}>no judge score</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}

      </main>
    </div>
  )
}
