import React, { useEffect, useRef, useState } from 'react'
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
    const xVals = data.map((d) => d.log_pop)
    const yVals = data.map((d) => d.log_mass)
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals)
    const yMin = Math.min(...yVals), yMax = Math.max(...yVals)
    const W = canvas.clientWidth, H = canvas.clientHeight
    const pad = 24
    const sx = (x) => pad + ((x - xMin) / (xMax - xMin || 1)) * (W - 2 * pad)
    const sy = (y) => H - pad - ((y - yMin) / (yMax - yMin || 1)) * (H - 2 * pad)

    if (mode === 'density') {
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
          ctx.fillStyle = `rgba(80,80,80,${Math.min(0.9, 0.1 + v)})`
          ctx.fillRect(x0, y0, (W - 2 * pad) / gw + 1, (H - 2 * pad) / gh + 1)
        }
      }
    } else {
      ctx.fillStyle = 'rgba(90, 90, 90, 0.55)'
      for (const d of data) {
        const x = sx(d.log_pop), y = sy(d.log_mass)
        ctx.fillRect(x, y, 2, 2)
      }
    }

    if (reg && isFinite(reg.slope)) {
      const xs = [xMin, xMax]
      const line = (m) => xs.map((x) => ({ x, y: reg.y0 + m * (x - reg.x0) }))
      if (isFinite(reg.slope_lo) && isFinite(reg.slope_hi)) {
        ctx.fillStyle = 'rgba(127,127,127,0.15)'
        ctx.beginPath()
        const upper = line(reg.slope_hi)
        const lower = line(reg.slope_lo).reverse()
        const all = upper.concat(lower)
        all.forEach((p, i) => { const X = sx(p.x), Y = sy(p.y); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y) })
        ctx.closePath(); ctx.fill()
      }
      ctx.strokeStyle = '#444'
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

const NeighborhoodPanel = ({ scope }) => {
  const [data, setData] = useState([])
  const [reg, setReg] = useState(null)
  const [mode, setMode] = useState('points')

  useEffect(() => {
    const load = async () => {
      let sampleUrl = `${DATA_BASE}/scatter_samples/global_neighborhood.json`
      let regUrl = `${DATA_BASE}/regression/global_neighborhood.json`
      if (scope.level === 'country') {
        sampleUrl = `${DATA_BASE}/scatter_samples/country=${scope.iso}.json`
        regUrl = `${DATA_BASE}/regression/country_neighborhood/${scope.iso}.json`
      } else if (scope.level === 'city') {
        // For city-level neighborhood scatter, build on the fly from the hex feed (small enough)
        const url = `${DATA_BASE}/hex/city=${scope.cityId}.json`
        const rows = await (await fetch(url)).json()
        const mapped = rows.map((r) => ({ log_pop: Math.log10(Math.max(1e-9, r.population_2015)), log_mass: Math.log10(Math.max(1e-9, r.total_built_mass_tons)) }))
        setData(mapped)
        const regRes = await fetch(`${DATA_BASE}/regression/city_neighborhood/${scope.cityId}.json`)
        if (regRes.ok) setReg(await regRes.json())
        return
      }
      const [sRes, rRes] = await Promise.all([fetch(sampleUrl), fetch(regUrl)])
      if (sRes.ok) setData(await sRes.json())
      if (rRes.ok) setReg(await rRes.json())
    }
    load()
  }, [scope])

  const title = scope.level === 'country' ? `Neighborhoods in ${scope.iso}` : scope.level === 'city' ? `Neighborhoods of city ${scope.cityId}` : 'Global Neighborhoods'
  return (
    <Panel title={title}>
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
                <div key={i} style={{ width: 18, height: 10, background: `rgba(80,80,80,${0.15 + (i/5)*0.75})`, borderRadius: 2 }} />
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
          <span style={{ display: 'inline-block', width: 18, height: 2, background: '#444' }} /> OLS fit
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 18, height: 10, background: 'rgba(127,127,127,0.15)', border: '1px solid rgba(127,127,127,0.25)' }} /> 95% CI band
        </div>
      </div>
    </Panel>
  )
}

export default NeighborhoodPanel
