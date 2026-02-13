import React, { useEffect, useRef, useState } from 'react'
import { DATA_BASE } from '../config'

const COLORS = {
  point: 'rgba(141, 160, 203, 0.5)',
  density: (v) => `rgba(141, 160, 203, ${Math.min(0.92, 0.08 + v * 0.84)})`,
  line: '#fc8d62',
  band: 'rgba(252, 141, 98, 0.18)',
}

const Scatter = ({ data, reg, mode, xKey, yKey, xLabel, yLabel, centered }) => {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    const W = canvas.clientWidth
    const H = canvas.clientHeight
    canvas.width = W * dpr
    canvas.height = H * dpr
    const ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, W, H)

    if (!data || !data.length) {
      ctx.fillStyle = '#9a948e'
      ctx.font = '12px DM Sans, system-ui'
      ctx.textAlign = 'center'
      ctx.fillText('No data available', W / 2, H / 2)
      return
    }

    const pad = { top: 12, right: 12, bottom: 28, left: centered ? 42 : 36 }
    const xVals = data.map((d) => d[xKey])
    const yVals = data.map((d) => d[yKey])
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals)
    const yMin = Math.min(...yVals), yMax = Math.max(...yVals)
    const pw = W - pad.left - pad.right
    const ph = H - pad.top - pad.bottom
    const sx = (x) => pad.left + ((x - xMin) / (xMax - xMin || 1)) * pw
    const sy = (y) => H - pad.bottom - ((y - yMin) / (yMax - yMin || 1)) * ph

    // Axis lines
    ctx.strokeStyle = '#ebe7e1'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(pad.left, pad.top)
    ctx.lineTo(pad.left, H - pad.bottom)
    ctx.lineTo(W - pad.right, H - pad.bottom)
    ctx.stroke()

    // Tick labels
    ctx.fillStyle = '#9a948e'
    ctx.font = '10px DM Sans, system-ui'
    ctx.textAlign = 'center'
    const xTicks = centered ? niceRangeCentered(xMin, xMax, 5) : niceRange(xMin, xMax, 5)
    xTicks.forEach((v) => {
      const x = sx(v)
      ctx.fillText(centered ? v.toFixed(1) : `10${superscript(v)}`, x, H - pad.bottom + 16)
      ctx.strokeStyle = '#f0ede9'
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, H - pad.bottom); ctx.stroke()
    })
    ctx.textAlign = 'right'
    const yTicks = centered ? niceRangeCentered(yMin, yMax, 4) : niceRange(yMin, yMax, 4)
    yTicks.forEach((v) => {
      const y = sy(v)
      ctx.fillText(centered ? v.toFixed(1) : `10${superscript(v)}`, pad.left - 6, y + 3)
      ctx.strokeStyle = '#f0ede9'
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke()
    })

    // Zero lines for centered view
    if (centered) {
      ctx.strokeStyle = '#d0cdc8'
      ctx.lineWidth = 1
      ctx.setLineDash([4, 3])
      if (xMin < 0 && xMax > 0) {
        const zx = sx(0)
        ctx.beginPath(); ctx.moveTo(zx, pad.top); ctx.lineTo(zx, H - pad.bottom); ctx.stroke()
      }
      if (yMin < 0 && yMax > 0) {
        const zy = sy(0)
        ctx.beginPath(); ctx.moveTo(pad.left, zy); ctx.lineTo(W - pad.right, zy); ctx.stroke()
      }
      ctx.setLineDash([])
    }

    // Axis labels
    ctx.fillStyle = '#6b6560'
    ctx.font = '10px DM Sans, system-ui'
    ctx.textAlign = 'center'
    ctx.fillText(xLabel, pad.left + pw / 2, H - 2)
    ctx.save()
    ctx.translate(10, pad.top + ph / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText(yLabel, 0, 0)
    ctx.restore()

    if (mode === 'density') {
      const gw = 120, gh = 90
      const grid = new Uint32Array(gw * gh)
      for (const d of data) {
        const xi = Math.min(gw - 1, Math.max(0, Math.floor(((d[xKey] - xMin) / (xMax - xMin || 1)) * gw)))
        const yi = Math.min(gh - 1, Math.max(0, Math.floor(((d[yKey] - yMin) / (yMax - yMin || 1)) * gh)))
        grid[yi * gw + xi]++
      }
      const max = Math.max(1, ...grid)
      for (let yi = 0; yi < gh; yi++) {
        for (let xi = 0; xi < gw; xi++) {
          const v = grid[yi * gw + xi] / max
          if (v <= 0) continue
          ctx.fillStyle = COLORS.density(v)
          const x0 = pad.left + (xi / gw) * pw
          const y0 = H - pad.bottom - ((yi + 1) / gh) * ph
          ctx.fillRect(x0, y0, pw / gw + 1, ph / gh + 1)
        }
      }
    } else {
      ctx.fillStyle = COLORS.point
      for (const d of data) {
        ctx.fillRect(sx(d[xKey]) - 1, sy(d[yKey]) - 1, 2.5, 2.5)
      }
    }

    // Regression
    if (reg && isFinite(reg.slope)) {
      const xs = [xMin, xMax]
      const line = (m) => xs.map((x) => ({ x, y: reg.y0 + m * (x - reg.x0) }))
      if (isFinite(reg.slope_lo) && isFinite(reg.slope_hi)) {
        ctx.fillStyle = COLORS.band
        ctx.beginPath()
        const upper = line(reg.slope_hi)
        const lower = line(reg.slope_lo).reverse()
        ;[...upper, ...lower].forEach((p, i) => {
          const X = sx(p.x), Y = sy(p.y)
          i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y)
        })
        ctx.closePath()
        ctx.fill()
      }
      ctx.strokeStyle = COLORS.line
      ctx.lineWidth = 2
      ctx.beginPath()
      const L = line(reg.slope)
      ctx.moveTo(sx(L[0].x), sy(L[0].y))
      ctx.lineTo(sx(L[1].x), sy(L[1].y))
      ctx.stroke()
    }
  }, [data, reg, mode, xKey, yKey, xLabel, yLabel, centered])

  return (
    <div className="chart-wrap">
      <canvas ref={canvasRef} />
    </div>
  )
}

const CityPanel = ({ scope, onSelectCity, countryName }) => {
  const [data, setData] = useState([])
  const [reg, setReg] = useState(null)
  const [mode, setMode] = useState('density')
  const [debugInfo, setDebugInfo] = useState('loading...')

  // Global: show de-centered; Country/City: show original
  const isCentered = scope.level === 'global'

  useEffect(() => {
    let cityUrl = `${DATA_BASE}/cities_agg/global.json`
    let regUrl = `${DATA_BASE}/regression/global_city.json`
    if (scope.level === 'country' || scope.level === 'city') {
      cityUrl = `${DATA_BASE}/cities_agg/country=${scope.iso}.json`
      regUrl = `${DATA_BASE}/regression/country/${scope.iso}.json`
    }
    setDebugInfo(`fetching ${cityUrl}...`)
    Promise.all([fetch(cityUrl), fetch(regUrl)]).then(async ([cRes, rRes]) => {
      if (cRes.ok) {
        const parsed = await cRes.json()
        setDebugInfo(`OK: ${cRes.status}, ${Array.isArray(parsed) ? parsed.length : typeof parsed} items from ${cityUrl}`)
        setData(parsed)
      } else {
        setDebugInfo(`FAIL: HTTP ${cRes.status} from ${cityUrl}`)
        setData([])
      }
      if (rRes.ok) setReg(await rRes.json())
      else setReg(null)
    }).catch((err) => {
      setDebugInfo(`ERROR: ${err.message} fetching ${cityUrl}`)
      setData([])
      setReg(null)
    })
  }, [scope])

  const title = (scope.level === 'country' || scope.level === 'city')
    ? `Cities in ${countryName || scope.iso}`
    : 'All Cities'

  const xKey = isCentered ? 'log_pop_c' : 'log_pop'
  const yKey = isCentered ? 'log_mass_c' : 'log_mass'
  const xLabel = isCentered ? '\u0394 log\u2081\u2080 Population' : 'log\u2081\u2080 Population'
  const yLabel = isCentered ? '\u0394 log\u2081\u2080 Built Mass' : 'log\u2081\u2080 Built Mass (t)'

  return (
    <>
      <div className="section-title">
        {title}
        <span className="badge city">City scale</span>
      </div>
      <div style={{ fontSize: 9, color: '#999', padding: '2px 0', fontFamily: 'monospace' }}>{debugInfo}</div>
      <div className="chart-controls">
        <span className="control-label">View</span>
        <div className="toggle-group">
          <button onClick={() => setMode('points')} disabled={mode === 'points'}>Points</button>
          <button onClick={() => setMode('density')} disabled={mode === 'density'}>Density</button>
        </div>
        {mode === 'density' && (
          <div className="density-ramp">
            <span className="label">low</span>
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="swatch" style={{ background: COLORS.density(i / 5) }} />
            ))}
            <span className="label">high</span>
          </div>
        )}
      </div>
      <Scatter data={data} reg={reg} mode={mode} xKey={xKey} yKey={yKey} xLabel={xLabel} yLabel={yLabel} centered={isCentered} />
      <div className="stat-row">
        {reg && isFinite(reg.slope) && (
          <>
            <span><span className="stat-label">Slope</span> <span className="stat-value">{reg.slope.toFixed(3)}</span></span>
            <span><span className="stat-label">95% CI</span> {reg.slope_lo.toFixed(3)}–{reg.slope_hi.toFixed(3)}</span>
            <span><span className="stat-label">n</span> {reg.n?.toLocaleString()}</span>
            <span><span className="stat-label">R&sup2;</span> {reg.r2?.toFixed(3)}</span>
          </>
        )}
      </div>
      <div className="legend-row">
        <span><span className="legend-swatch" style={{ display: 'inline-block', width: 16, height: 2, borderRadius: 1, background: COLORS.line }} /> OLS fit</span>
        <span><span className="legend-swatch" style={{ display: 'inline-block', width: 16, height: 8, borderRadius: 2, background: COLORS.band, border: `1px solid ${COLORS.band}` }} /> 95% CI</span>
      </div>
      <div className="city-select">
        <label>Jump to city (ID)</label>
        <input
          type="number"
          placeholder="e.g. 474"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              const id = parseInt(e.target.value)
              if (id) onSelectCity(id)
            }
          }}
        />
      </div>
    </>
  )
}

function niceRange(lo, hi, count) {
  const step = Math.ceil((hi - lo) / count)
  const start = Math.ceil(lo)
  const ticks = []
  for (let v = start; v <= hi; v += Math.max(1, step)) ticks.push(v)
  return ticks
}

function niceRangeCentered(lo, hi, count) {
  const range = hi - lo
  const rawStep = range / count
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const steps = [1, 2, 5, 10]
  const step = steps.find((s) => s * mag >= rawStep) * mag || rawStep
  const start = Math.ceil(lo / step) * step
  const ticks = []
  for (let v = start; v <= hi + step * 0.01; v += step) ticks.push(Math.round(v / step) * step)
  return ticks
}

function superscript(n) {
  const sup = { '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3', '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079', '-': '\u207B', '.': '\u00B7' }
  return String(Math.round(n)).split('').map((c) => sup[c] || c).join('')
}

export default CityPanel
