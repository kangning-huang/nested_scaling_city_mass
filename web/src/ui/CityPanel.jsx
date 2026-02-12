import React, { useEffect, useMemo, useRef, useState } from 'react'
import { DATA_BASE } from '../config'

const Panel = ({ title, children }) => (
  <div style={{ padding: 12 }}>
    <div style={{ fontWeight: 600, marginBottom: 8 }}>{title}</div>
    {children}
  </div>
)

const Scatter = ({ data, reg, mode }) => {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = canvas.clientWidth * dpr
    canvas.height = canvas.clientHeight * dpr
    const ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight)

    if (!data || !data.length) return
    // Compute scales
    const xVals = data.map((d) => d.log_pop)
    const yVals = data.map((d) => d.log_mass)
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals)
    const yMin = Math.min(...yVals), yMax = Math.max(...yVals)
    const W = canvas.clientWidth, H = canvas.clientHeight
    const pad = 24
    const sx = (x) => pad + ((x - xMin) / (xMax - xMin || 1)) * (W - 2 * pad)
    const sy = (y) => H - pad - ((y - yMin) / (yMax - yMin || 1)) * (H - 2 * pad)

    if (mode === 'density') {
      // Simple grid density
      const gw = 140, gh = 100
      const grid = new Uint32Array(gw * gh)
      for (const d of data) {
        const xi = Math.min(gw - 1, Math.max(0, Math.floor(((d.log_pop - xMin) / (xMax - xMin || 1)) * gw)))
        const yi = Math.min(gh - 1, Math.max(0, Math.floor(((d.log_mass - yMin) / (yMax - yMin || 1)) * gh)))
        grid[yi * gw + xi]++
      }
      const max = Math.max(1, ...grid)
      for (let yi = 0; yi < gh; yi++) {
        for (let xi = 0; xi < gw; xi++) {
          const v = grid[yi * gw + xi] / max
          if (v <= 0) continue
          const x0 = pad + (xi / gw) * (W - 2 * pad)
          const y0 = H - pad - ((yi + 1) / gh) * (H - 2 * pad)
          ctx.fillStyle = `rgba(30,90,156,${Math.min(0.9, 0.1 + v)})`
          ctx.fillRect(x0, y0, (W - 2 * pad) / gw + 1, (H - 2 * pad) / gh + 1)
        }
      }
    } else {
      // Points
      ctx.fillStyle = 'rgba(30, 90, 156, 0.6)'
      for (const d of data) {
        const x = sx(d.log_pop), y = sy(d.log_mass)
        ctx.fillRect(x, y, 2, 2)
      }
    }

    // Regression line + CI band if provided
    if (reg && isFinite(reg.slope)) {
      const xs = [xMin, xMax]
      const line = (m) => xs.map((x) => ({ x, y: reg.y0 + m * (x - reg.x0) }))
      // CI band
      if (isFinite(reg.slope_lo) && isFinite(reg.slope_hi)) {
        ctx.fillStyle = 'rgba(70,126,187,0.15)'
        ctx.beginPath()
        const upper = line(reg.slope_hi)
        const lower = line(reg.slope_lo).reverse()
        const all = upper.concat(lower)
        all.forEach((p, i) => { const X = sx(p.x), Y = sy(p.y); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y) })
        ctx.closePath()
        ctx.fill()
      }
      // Line
      ctx.strokeStyle = '#1e5a9c'
      ctx.lineWidth = 2
      ctx.beginPath()
      const L = line(reg.slope)
      ctx.moveTo(sx(L[0].x), sy(L[0].y))
      ctx.lineTo(sx(L[1].x), sy(L[1].y))
      ctx.stroke()
    }
  }, [data, reg])

  return <div style={{ height: 260, border: '1px solid #eee', borderRadius: 6 }}><canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} /></div>
}

const CityPanel = ({ scope, onSelectCity }) => {
  const [data, setData] = useState([])
  const [reg, setReg] = useState(null)
  const [mode, setMode] = useState('points')

  useEffect(() => {
    const load = async () => {
      let cityUrl = `${DATA_BASE}/cities_agg/global.json`
      let regUrl = `${DATA_BASE}/regression/global_city.json`
      if (scope.level === 'country') {
        cityUrl = `${DATA_BASE}/cities_agg/country=${scope.iso}.json`
        regUrl = `${DATA_BASE}/regression/country/${scope.iso}.json`
      }
      const [cityRes, regRes] = await Promise.all([fetch(cityUrl), fetch(regUrl)])
      if (cityRes.ok) setData(await cityRes.json())
      if (regRes.ok) setReg(await regRes.json())
    }
    load()
  }, [scope])

  return (
    <Panel title={scope.level === 'country' ? `Cities in ${scope.iso}` : 'Global Cities'}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: '#666' }}>Mode:</span>
        <div className="btn-group">
          <button onClick={() => setMode('points')} disabled={mode==='points'}>Points</button>
          <button onClick={() => setMode('density')} disabled={mode==='density'}>Density</button>
        </div>
        {mode==='density' && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#666' }}>
            <span>Density</span>
            <div style={{ display: 'inline-flex', gap: 2 }}>
              {Array.from({length:6},(_,i)=>i).map(i => (
                <div key={i} style={{ width: 18, height: 10, background: `rgba(30,90,156,${0.15 + (i/5)*0.75})`, borderRadius: 2 }} />
              ))}
            </div>
            <span>higher</span>
          </div>
        )}
      </div>
      <Scatter data={data} reg={reg} mode={mode} />
      <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
        {reg && isFinite(reg.slope) && (
          <>Slope: {reg.slope.toFixed(3)} (95% CI {reg.slope_lo.toFixed(3)}–{reg.slope_hi.toFixed(3)}), n={reg.n}</>
        )}
      </div>
      <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 10, fontSize: 11, color: '#666' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 18, height: 2, background: '#1e5a9c' }} /> OLS fit
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 18, height: 10, background: 'rgba(30,90,156,0.15)', border: '1px solid rgba(30,90,156,0.25)' }} /> 95% CI band
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <label style={{ fontSize: 12 }}>Select city by ID: </label>
        <input type="number" placeholder="city_id" onKeyDown={(e) => { if (e.key === 'Enter') onSelectCity(parseInt(e.target.value)) }} />
      </div>
    </Panel>
  )
}

export default CityPanel
