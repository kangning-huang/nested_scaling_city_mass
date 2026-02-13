import React, { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { H3HexagonLayer } from '@deck.gl/geo-layers'
import { interpolateViridis } from 'd3-scale-chromatic'
import * as h3 from 'h3-js'
import { DATA_BASE } from '../config'

const MAPTILER_KEY = import.meta.env.VITE_MAPTILER_KEY
const MAP_STYLE = MAPTILER_KEY
  ? `https://api.maptiler.com/maps/dataviz-dark/style.json?key=${MAPTILER_KEY}`
  : 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

const MapView = ({ scope, metric, onSelectCountry, onSelectCity }) => {
  const containerRef = useRef(null)
  const mapInstanceRef = useRef(null)
  const overlayRef = useRef(null)
  const hexDataRef = useRef([])
  const cityPointsRef = useRef({ type: 'FeatureCollection', features: [] })
  const [legend, setLegend] = useState({ lo: null, hi: null })
  const [tooltip, setTooltip] = useState(null)
  const [countryCityCounts, setCountryCityCounts] = useState({})

  useEffect(() => {
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: [0, 20],
      zoom: 1.3,
    })
    mapInstanceRef.current = map

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }))
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 160, unit: 'metric' }), 'bottom-left')
    const overlay = new MapboxOverlay({ layers: [] })
    map.addControl(overlay)
    overlayRef.current = overlay

    map.on('load', async () => {
      // Countries layer
      const cRes = await fetch(`${DATA_BASE}/countries.geojson`)
      if (cRes.ok) {
        const data = await cRes.json()
        map.addSource('countries', { type: 'geojson', data })
        map.addLayer({ id: 'country-fill', type: 'fill', source: 'countries', paint: { 'fill-color': '#3a3a42', 'fill-opacity': 0.35 } })
        map.addLayer({ id: 'country-outline', type: 'line', source: 'countries', paint: { 'line-color': '#5a5a65', 'line-width': 0.6 } })
        map.on('click', 'country-fill', (e) => {
          const f = e.features?.[0]
          if (!f) return
          const iso3 = f.properties?.iso3
          const name = f.properties?.name
          if (iso3) onSelectCountry(iso3, name)
          map.fitBounds(turfBbox(f), { padding: 30, duration: 600 })
        })
        map.on('mousemove', 'country-fill', (e) => {
          const f = e.features?.[0]
          if (!f) { setTooltip(null); return }
          setTooltip({ type: 'country', x: e.point.x, y: e.point.y, iso: f.properties?.iso3, name: f.properties?.name })
        })
        map.on('mouseleave', 'country-fill', () => setTooltip(null))
        map.on('mouseenter', 'country-fill', () => (map.getCanvas().style.cursor = 'pointer'))
        map.on('mouseleave', 'country-fill', () => (map.getCanvas().style.cursor = 'default'))
      }

      // City points from index
      const metaRes = await fetch(`${DATA_BASE}/index/city_meta.json`)
      if (metaRes.ok) {
        const meta = await metaRes.json()
        const features = Object.entries(meta).map(([cityId, m]) => {
          return { type: 'Feature', geometry: { type: 'Point', coordinates: [m.lon, m.lat] }, properties: { city_id: parseInt(cityId), country_iso: m.country_iso, name: m.city } }
        })
        cityPointsRef.current = { type: 'FeatureCollection', features }
        map.addSource('cities', { type: 'geojson', data: cityPointsRef.current })
        map.addLayer({
          id: 'city-pts', type: 'circle', source: 'cities',
          paint: { 'circle-color': '#c8873a', 'circle-radius': 3, 'circle-opacity': 0.75, 'circle-stroke-color': 'rgba(200,135,58,0.3)', 'circle-stroke-width': 2 },
        })
        map.on('click', 'city-pts', (e) => {
          const f = e.features?.[0]
          if (f) onSelectCity(f.properties?.city_id, f.properties?.name)
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

      // Country→cities index
      const idxRes = await fetch(`${DATA_BASE}/index/country_to_cities.json`)
      if (idxRes.ok) setCountryCityCounts(await idxRes.json())
    })

    return () => { overlay.remove(); map.remove() }
  }, [])

  // Hex layer on city selection
  useEffect(() => {
    const overlay = overlayRef.current
    if (!overlay) return
    if (scope.level !== 'city') { overlay.setProps({ layers: [] }); return }
    fetch(`${DATA_BASE}/hex/city=${scope.cityId}.json`)
      .then((r) => r.json())
      .then((rows) => {
        hexDataRef.current = rows
        const pts = rows.map((d) => h3.cellToLatLng(d.h3index)).map(([lat, lng]) => [lng, lat])
        fitToPoints(pts)
        renderHexLayer()
      })
      .catch(() => overlay.setProps({ layers: [] }))
  }, [scope])

  // Recolor on metric change
  useEffect(() => { if (scope.level === 'city') renderHexLayer() }, [metric])

  // Filter city points by country
  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || !map.getSource('cities')) return
    if (scope.level === 'country' || scope.level === 'city') {
      const filtered = { type: 'FeatureCollection', features: cityPointsRef.current.features.filter((f) => f.properties.country_iso === scope.iso) }
      map.getSource('cities').setData(filtered)
    } else {
      map.getSource('cities').setData(cityPointsRef.current)
    }
  }, [scope])

  const fitToPoints = (coords) => {
    if (!coords.length) return
    const xs = coords.map((c) => c[0]), ys = coords.map((c) => c[1])
    const map = mapInstanceRef.current
    if (map && isFinite(Math.min(...xs))) {
      map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 40, duration: 600 })
    }
  }

  const renderHexLayer = () => {
    const overlay = overlayRef.current
    if (!overlay) return
    const rows = hexDataRef.current
    if (!rows?.length) { overlay.setProps({ layers: [] }); return }
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
    overlay.setProps({
      layers: [new H3HexagonLayer({
        id: 'city-h3', data: rows, pickable: true,
        getHexagon: (d) => d.h3index,
        getFillColor: (d) => {
          const v = metric === 'pop' ? d.population_2015 : d.total_built_mass_tons
          const [r, g, b] = cssToRgb(color(v))
          return [r, g, b, 210]
        },
        stroked: false, extruded: false,
        onHover: ({ x, y, object }) => {
          if (!object) { setTooltip(null); return }
          setTooltip({ type: 'hex', x, y, h3: object.h3index, pop: object.population_2015, mass: object.total_built_mass_tons })
        },
      })],
    })
  }

  const gradient = useMemo(() => {
    const colors = Array.from({ length: 11 }, (_, i) => interpolateViridis(i / 10)).join(',')
    return `linear-gradient(to right, ${colors})`
  }, [])

  return (
    <div className="map-container">
      <div ref={containerRef} />

      {/* Legend */}
      {legend.lo && legend.hi && (
        <div className="map-legend">
          <div className="map-legend-title">{metric === 'pop' ? 'Population' : 'Built Mass (tons)'}</div>
          <div className="map-legend-bar" style={{ background: gradient }} />
          <div className="map-legend-ticks">
            {(() => {
              const kmin = Math.floor(Math.log10(legend.lo))
              const kmax = Math.ceil(Math.log10(legend.hi))
              const ticks = []
              for (let k = kmin; k <= kmax; k++) ticks.push(k)
              return ticks.map((k) => <span key={k}>10<sup>{k}</sup></span>)
            })()}
          </div>
          <div className="map-legend-range">
            <span>{formatSI(legend.lo)}</span>
            <span>{formatSI(legend.hi)}</span>
          </div>
        </div>
      )}

      {/* Tooltips */}
      {tooltip && tooltip.type === 'hex' && (
        <div className="map-tooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>H3 Cell</strong>
          <div className="tt-row"><span className="tt-label">Population</span><span className="tt-value">{formatSI(tooltip.pop)}</span></div>
          <div className="tt-row"><span className="tt-label">Built Mass</span><span className="tt-value">{formatSI(tooltip.mass)} t</span></div>
        </div>
      )}
      {tooltip && tooltip.type === 'city' && (
        <div className="map-tooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltip.name}</strong>
          <div className="tt-row"><span className="tt-label">Country</span><span className="tt-value">{tooltip.iso}</span></div>
        </div>
      )}
      {tooltip && tooltip.type === 'country' && (
        <div className="map-tooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltip.name}</strong>
          <div className="tt-row"><span className="tt-label">ISO</span><span className="tt-value">{tooltip.iso}</span></div>
          <div className="tt-row"><span className="tt-label">Cities</span><span className="tt-value">{(countryCityCounts[tooltip.iso] || []).length}</span></div>
        </div>
      )}
    </div>
  )
}

function formatSI(num) {
  if (!isFinite(num)) return ''
  const abs = Math.abs(num)
  if (abs >= 1e9) return (num / 1e9).toFixed(1) + 'B'
  if (abs >= 1e6) return (num / 1e6).toFixed(1) + 'M'
  if (abs >= 1e3) return (num / 1e3).toFixed(1) + 'k'
  return num.toFixed(0)
}

function cssToRgb(css) {
  const c = css.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/)
  return c ? [parseInt(c[1]), parseInt(c[2]), parseInt(c[3])] : [160, 160, 160]
}

function turfBbox(feat) {
  const coords = []
  const walk = (g) => {
    if (g.type === 'Polygon') coords.push(...g.coordinates.flat())
    else if (g.type === 'MultiPolygon') coords.push(...g.coordinates.flat(2))
    else if (g.type === 'GeometryCollection') g.geometries.forEach(walk)
  }
  walk(feat.geometry)
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  coords.forEach(([x, y]) => { if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y })
  return [[minX, minY], [maxX, maxY]]
}

export default MapView
