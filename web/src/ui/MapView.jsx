import React, { useEffect, useMemo, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { H3HexagonLayer } from '@deck.gl/aggregation-layers'
import { interpolateViridis } from 'd3-scale-chromatic'
import * as h3 from 'h3-js'
import React, { useState } from 'react'
import { DATA_BASE } from '../config'

const MAPTILER_KEY = import.meta.env.VITE_MAPTILER_KEY || 'YOUR_MAPTILER_KEY'

const MapView = ({ scope, metric, onSelectCountry, onSelectCity }) => {
  const mapRef = useRef(null)
  const overlayRef = useRef(null)
  const hexDataRef = useRef([])
  const [legend, setLegend] = useState({ lo: null, hi: null })
  const [tooltip, setTooltip] = useState(null)
  const [countryCityCounts, setCountryCityCounts] = useState({})

  const cityPointsRef = useRef({ features: [] })

  useEffect(() => {
    const map = new maplibregl.Map({
      container: mapRef.current,
      style: `https://api.maptiler.com/maps/dataviz-light/style.json?key=${MAPTILER_KEY}`,
      center: [0, 20],
      zoom: 1.3,
    })

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }))
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 160, unit: 'metric' }), 'bottom-left')
    const overlay = new MapboxOverlay({ layers: [] })
    map.addControl(overlay)
    overlayRef.current = overlay

    // Add countries GeoJSON source and layer
    map.on('load', async () => {
      const res = await fetch(`${DATA_BASE}/countries.geojson`)
      if (res.ok) {
        const data = await res.json()
        map.addSource('countries', { type: 'geojson', data })
        map.addLayer({ id: 'country-fill', type: 'fill', source: 'countries', paint: { 'fill-color': '#e8eef6', 'fill-opacity': 0.4 } })
        map.addLayer({ id: 'country-outline', type: 'line', source: 'countries', paint: { 'line-color': '#5b6b7a', 'line-width': 0.8 } })
        map.on('click', 'country-fill', (e) => {
          const iso3 = e.features?.[0]?.properties?.iso3
          if (iso3) onSelectCountry(iso3)
          const bbox = turfBbox(e.features[0])
          map.fitBounds(bbox, { padding: 30, duration: 600 })
        })
        map.on('mousemove', 'country-fill', (e) => {
          const f = e.features?.[0]
          if (!f) { setTooltip(null); return }
          const iso3 = f.properties?.iso3
          const name = f.properties?.name
          const count = (countryCityCounts[iso3] || []).length
          setTooltip({ type: 'country', x: e.point.x, y: e.point.y, iso: iso3, name, count })
        })
        map.on('mouseleave', 'country-fill', () => setTooltip(null))
        map.getCanvas().style.cursor = 'default'
        map.on('mouseenter', 'country-fill', () => (map.getCanvas().style.cursor = 'pointer'))
        map.on('mouseleave', 'country-fill', () => (map.getCanvas().style.cursor = 'default'))
      }

      // Build city points from index/city_meta.json
      const metaRes = await fetch(`${DATA_BASE}/index/city_meta.json`)
      if (metaRes.ok) {
        const meta = await metaRes.json()
        const features = Object.entries(meta).map(([cityId, m]) => {
          const [lat, lng] = h3.cellToLatLng(m.sample_h3)
          return { type: 'Feature', geometry: { type: 'Point', coordinates: [lng, lat] }, properties: { city_id: parseInt(cityId), country_iso: m.country_iso, name: m.city } }
        })
        cityPointsRef.current = { type: 'FeatureCollection', features }
        map.addSource('cities', { type: 'geojson', data: cityPointsRef.current })
        map.addLayer({ id: 'city-pts', type: 'circle', source: 'cities', paint: { 'circle-color': '#1e5a9c', 'circle-radius': 3, 'circle-opacity': 0.7 } })
        map.on('click', 'city-pts', (e) => {
          const cid = e.features?.[0]?.properties?.city_id
          if (cid) onSelectCity(cid)
        })
        map.on('mousemove', 'city-pts', (e) => {
          const f = e.features?.[0]
          if (!f) { setTooltip(null); return }
          setTooltip({ type: 'city', x: e.point.x, y: e.point.y, name: f.properties?.name, city_id: f.properties?.city_id, iso: f.properties?.country_iso })
        })
        map.on('mouseleave', 'city-pts', () => setTooltip(null))
        map.on('mouseenter', 'city-pts', () => (map.getCanvas().style.cursor = 'pointer'))
        map.on('mouseleave', 'city-pts', () => (map.getCanvas().style.cursor = 'default'))
      }

      // Load country→cities index for counts in country tooltip
      const idxRes = await fetch(`${DATA_BASE}/index/country_to_cities.json`)
      if (idxRes.ok) {
        const idx = await idxRes.json()
        setCountryCityCounts(idx)
      }
    })

    return () => {
      overlay.remove()
      map.remove()
    }
  }, [])

  // Load and render per-city hex layer when a city is selected
  useEffect(() => {
    const overlay = overlayRef.current
    if (!overlay) return
    if (scope.level !== 'city') {
      overlay.setProps({ layers: [] })
      return
    }
    const url = `${DATA_BASE}/hex/city=${scope.cityId}.json`
    fetch(url)
      .then((r) => r.json())
      .then((rows) => {
        hexDataRef.current = rows
        // Compute bounds from H3 cells
        const pts = rows.map((d) => h3.cellToLatLng(d.h3index)).map(([lat, lng]) => [lng, lat])
        fitToPoints(pts)
        renderHexLayer()
      })
      .catch(() => overlay.setProps({ layers: [] }))
  }, [scope])

  useEffect(() => {
    // recolor on metric change
    if (scope.level === 'city') renderHexLayer()
  }, [metric])

  // Filter city points by country when country selected
  useEffect(() => {
    const map = mapRef.current && mapRef.current._map
    if (!map || !map.getSource('cities')) return
    if (scope.level === 'country') {
      const all = cityPointsRef.current
      const filtered = { type: 'FeatureCollection', features: all.features.filter((f) => f.properties.country_iso === scope.iso) }
      map.getSource('cities').setData(filtered)
    } else {
      map.getSource('cities').setData(cityPointsRef.current)
    }
  }, [scope])

  const fitToPoints = (coords) => {
    if (!coords.length) return
    const xs = coords.map((c) => c[0])
    const ys = coords.map((c) => c[1])
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    const map = mapRef.current && mapRef.current._map
    if (map && isFinite(minX)) {
      map.fitBounds([[minX, minY], [maxX, maxY]], { padding: 40, duration: 600 })
    }
  }

  const renderHexLayer = () => {
    const overlay = overlayRef.current
    if (!overlay) return
    const rows = hexDataRef.current
    if (!rows || !rows.length) {
      overlay.setProps({ layers: [] })
      return
    }
    // Build color scale domain from metric
    const vals = rows.map((d) => (metric === 'pop' ? d.population_2015 : d.total_built_mass_tons)).filter((v) => v > 0)
    const sorted = vals.sort((a, b) => a - b)
    const lo = sorted[Math.floor(sorted.length * 0.02)] || 1
    const hi = sorted[Math.floor(sorted.length * 0.98)] || lo * 10
    setLegend({ lo, hi })
    const color = (v) => {
      const t = Math.max(lo, Math.min(hi, v))
      const s = Math.log10(t) - Math.log10(lo)
      const d = Math.log10(hi) - Math.log10(lo)
      return interpolateViridis(d > 0 ? s / d : 0.5)
    }
    const layer = new H3HexagonLayer({
      id: 'city-h3',
      data: rows,
      pickable: true,
      getHexagon: (d) => d.h3index,
      getFillColor: (d) => {
        const v = metric === 'pop' ? d.population_2015 : d.total_built_mass_tons
        const [r, g, b] = cssToRgb(color(v))
        return [r, g, b, 200]
      },
      stroked: false,
      extruded: false,
      onHover: ({x, y, object}) => {
        if (!object) { setTooltip(null); return }
        const v = metric === 'pop' ? object.population_2015 : object.total_built_mass_tons
        setTooltip({ type: 'hex', x, y, h3: object.h3index, value: v, pop: object.population_2015, mass: object.total_built_mass_tons })
      },
    })
    overlay.setProps({ layers: [layer] })
  }

  // Build CSS gradient for legend
  const gradient = useMemo(() => {
    const stops = Array.from({ length: 11 }, (_, i) => i / 10)
    const colors = stops.map((t) => interpolateViridis(t)).join(',')
    return `linear-gradient(to right, ${colors})`
  }, [])

  const formatSI = (num) => {
    if (!isFinite(num)) return ''
    const abs = Math.abs(num)
    if (abs >= 1e9) return (num/1e9).toFixed(1)+'B'
    if (abs >= 1e6) return (num/1e6).toFixed(1)+'M'
    if (abs >= 1e3) return (num/1e3).toFixed(1)+'k'
    return num.toFixed(0)
  }

  const Legend = () => {
    if (!legend.lo || !legend.hi) return null
    const log10 = (x) => Math.log10(x)
    const kmin = Math.floor(log10(legend.lo))
    const kmax = Math.ceil(log10(legend.hi))
    const ticks = []
    for (let k = kmin; k <= kmax; k++) ticks.push(k)
    return (
      <div style={{ position: 'absolute', right: 12, bottom: 12, background: 'rgba(255,255,255,0.9)', padding: 8, borderRadius: 6, boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        <div style={{ fontSize: 12, marginBottom: 4 }}>Color scale: {metric === 'pop' ? 'Population (people)' : 'Built Mass (tons)'}</div>
        <div style={{ width: 200 }}>
          <div style={{ height: 10, background: gradient, borderRadius: 4 }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#555', marginTop: 4 }}>
            {ticks.map((k) => <span key={k}>10^{k}</span>)}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#888', marginTop: 2 }}>
            <span>{formatSI(legend.lo)}</span>
            <span>{formatSI(legend.hi)}</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ position: 'relative' }}>
      <div ref={mapRef} style={{ width: '100%', height: '100%' }} />
      <Legend />
      {tooltip && tooltip.type === 'hex' && (
        <div style={{ position: 'absolute', left: tooltip.x + 12, top: tooltip.y + 12, background: 'rgba(255,255,255,0.95)', border: '1px solid #e6e8eb', padding: '6px 8px', borderRadius: 6, fontSize: 12, boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <div><b>H3</b>: {tooltip.h3}</div>
          <div>Population: {formatSI(tooltip.pop)}</div>
          <div>Built Mass: {formatSI(tooltip.mass)} t</div>
        </div>
      )}
      {tooltip && tooltip.type === 'city' && (
        <div style={{ position: 'absolute', left: tooltip.x + 12, top: tooltip.y + 12, background: 'rgba(255,255,255,0.95)', border: '1px solid #e6e8eb', padding: '6px 8px', borderRadius: 6, fontSize: 12, boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <div><b>{tooltip.name}</b></div>
          <div>City ID: {tooltip.city_id}</div>
          <div>Country: {tooltip.iso}</div>
        </div>
      )}
      {tooltip && tooltip.type === 'country' && (
        <div style={{ position: 'absolute', left: tooltip.x + 12, top: tooltip.y + 12, background: 'rgba(255,255,255,0.95)', border: '1px solid #e6e8eb', padding: '6px 8px', borderRadius: 6, fontSize: 12, boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <div><b>{tooltip.name}</b></div>
          <div>ISO3: {tooltip.iso}</div>
          <div>Cities: {(countryCityCounts[tooltip.iso] || []).length}</div>
        </div>
      )}
    </div>
  )
}

// Helpers
function cssToRgb(css) {
  const c = css.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/)
  if (!c) return [160, 160, 160]
  return [parseInt(c[1]), parseInt(c[2]), parseInt(c[3])]
}

// Minimal bbox for a GeoJSON feature (no turf dependency)
function turfBbox(feat) {
  const coords = []
  const walk = (g) => {
    const t = g.type
    if (t === 'Polygon') coords.push(...g.coordinates.flat())
    else if (t === 'MultiPolygon') coords.push(...g.coordinates.flat(2))
    else if (t === 'GeometryCollection') g.geometries.forEach(walk)
  }
  walk(feat.geometry)
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  coords.forEach(([x, y]) => { if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y })
  return [[minX, minY], [maxX, maxY]]
}

export default MapView
