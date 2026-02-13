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
  const countriesDataRef = useRef(null)
  const countryCityCountsRef = useRef({})
  const [legend, setLegend] = useState({ lo: null, hi: null })
  const [tooltip, setTooltip] = useState(null)

  // Keep callback refs fresh to avoid stale closures in map event handlers
  const onSelectCountryRef = useRef(onSelectCountry)
  const onSelectCityRef = useRef(onSelectCity)
  useEffect(() => { onSelectCountryRef.current = onSelectCountry }, [onSelectCountry])
  useEffect(() => { onSelectCityRef.current = onSelectCity }, [onSelectCity])

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
        countriesDataRef.current = data
        map.addSource('countries', { type: 'geojson', data })
        map.addLayer({ id: 'country-fill', type: 'fill', source: 'countries', paint: { 'fill-color': '#3a3a42', 'fill-opacity': 0.15 } })
        map.addLayer({ id: 'country-outline', type: 'line', source: 'countries', paint: { 'line-color': '#5a5a65', 'line-width': 0.6, 'line-opacity': 0.5 } })
        map.on('click', 'country-fill', (e) => {
          const props = e.features?.[0]?.properties
          const iso3 = props?.iso3
          const name = props?.name
          if (iso3) onSelectCountryRef.current(iso3, name)
        })
        map.on('mousemove', 'country-fill', (e) => {
          const f = e.features?.[0]
          if (!f) { setTooltip(null); return }
          const iso3 = f.properties?.iso3
          const name = f.properties?.name
          const count = (countryCityCountsRef.current[iso3] || []).length
          setTooltip({ type: 'country', x: e.point.x, y: e.point.y, iso: iso3, name, count })
        })
        map.on('mouseleave', 'country-fill', () => setTooltip(null))
        map.on('mouseenter', 'country-fill', () => (map.getCanvas().style.cursor = 'pointer'))
        map.on('mouseleave', 'country-fill', () => (map.getCanvas().style.cursor = 'default'))
      }

      // City points from index + population data
      const [metaRes, aggRes] = await Promise.all([
        fetch(`${DATA_BASE}/index/city_meta.json`),
        fetch(`${DATA_BASE}/cities_agg/global.json`)
      ])

      let popLookup = {}
      if (aggRes.ok) {
        const aggData = await aggRes.json()
        for (const row of aggData) {
          const id = row.city_id ?? row.ID_HDC_G0
          const pop = row.pop_total ?? row.population_2015 ?? (row.log_pop ? Math.pow(10, row.log_pop) : 0)
          if (id != null) popLookup[id] = pop
        }
      }

      if (metaRes.ok) {
        const meta = await metaRes.json()
        const features = Object.entries(meta).map(([cityId, m]) => {
          const pop = popLookup[parseInt(cityId)] || 0
          return {
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [m.lon, m.lat] },
            properties: { city_id: parseInt(cityId), country_iso: m.country_iso, name: m.city, pop_total: pop }
          }
        })
        cityPointsRef.current = { type: 'FeatureCollection', features }
        map.addSource('cities', { type: 'geojson', data: cityPointsRef.current })
        map.addLayer({
          id: 'city-pts', type: 'circle', source: 'cities',
          paint: {
            'circle-color': '#c8873a',
            'circle-radius': [
              'interpolate', ['linear'],
              ['case',
                ['>', ['get', 'pop_total'], 0],
                ['log10', ['get', 'pop_total']],
                4
              ],
              4, 2,
              5, 4,
              6, 7,
              7, 12
            ],
            'circle-opacity': 0.75,
            'circle-stroke-color': 'rgba(200,135,58,0.3)',
            'circle-stroke-width': 2,
          },
        })
        map.on('click', 'city-pts', (e) => {
          const f = e.features?.[0]
          const cid = f?.properties?.city_id
          const iso = f?.properties?.country_iso
          if (cid) {
            const cntryName = countriesDataRef.current?.features?.find(
              cf => cf.properties?.iso3 === iso
            )?.properties?.name || iso
            onSelectCityRef.current(cid, iso, cntryName)
          }
        })
        map.on('mousemove', 'city-pts', (e) => {
          const f = e.features?.[0]
          if (!f) { setTooltip(null); return }
          setTooltip({
            type: 'city', x: e.point.x, y: e.point.y,
            name: f.properties?.name, city_id: f.properties?.city_id,
            iso: f.properties?.country_iso, pop: f.properties?.pop_total
          })
        })
        map.on('mouseleave', 'city-pts', () => setTooltip(null))
        map.on('mouseenter', 'city-pts', () => (map.getCanvas().style.cursor = 'pointer'))
        map.on('mouseleave', 'city-pts', () => (map.getCanvas().style.cursor = 'default'))
      }

      // Country→cities index
      const idxRes = await fetch(`${DATA_BASE}/index/country_to_cities.json`)
      if (idxRes.ok) {
        const idx = await idxRes.json()
        countryCityCountsRef.current = idx
      }
    })

    return () => {
      overlay.remove()
      map.remove()
      mapInstanceRef.current = null
    }
  }, [])

  // Zoom map when scope changes (handles breadcrumb navigation + country clicks)
  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || !map.loaded()) return
    if (scope.level === 'global') {
      map.flyTo({ center: [0, 20], zoom: 1.3, duration: 600 })
    } else if (scope.level === 'country' && countriesDataRef.current) {
      const feat = countriesDataRef.current.features.find(f => f.properties?.iso3 === scope.iso)
      if (feat) {
        const bbox = turfBbox(feat)
        map.fitBounds(bbox, { padding: 30, duration: 600 })
      }
    }
    // city-level zoom is handled by the hex loading effect below
  }, [scope.level, scope.iso])

  // Hex layer on city selection
  useEffect(() => {
    const overlay = overlayRef.current
    if (!overlay) return
    if (scope.level !== 'city') {
      overlay.setProps({ layers: [] })
      return
    }
    fetch(`${DATA_BASE}/hex/city=${scope.cityId}.json`)
      .then((r) => r.json())
      .then((rows) => {
        hexDataRef.current = rows
        const pts = rows.map((d) => h3.cellToLatLng(d.h3index)).map(([lat, lng]) => [lng, lat])
        fitToPoints(pts)
        renderHexLayer()
      })
      .catch(() => overlay.setProps({ layers: [] }))
  }, [scope.level, scope.cityId])

  // Recolor on metric change
  useEffect(() => {
    if (scope.level === 'city') renderHexLayer()
  }, [metric])

  // Filter city points by country
  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || !map.getSource('cities')) return
    if (scope.level === 'country') {
      const filtered = { type: 'FeatureCollection', features: cityPointsRef.current.features.filter((f) => f.properties.country_iso === scope.iso) }
      map.getSource('cities').setData(filtered)
    } else if (scope.level === 'global') {
      map.getSource('cities').setData(cityPointsRef.current)
    }
  }, [scope.level, scope.iso])

  const fitToPoints = (coords) => {
    if (!coords.length) return
    const xs = coords.map((c) => c[0])
    const ys = coords.map((c) => c[1])
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    const map = mapInstanceRef.current
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
          {tooltip.pop > 0 && <div className="tt-row"><span className="tt-label">Population</span><span className="tt-value">{formatSI(tooltip.pop)}</span></div>}
        </div>
      )}
      {tooltip && tooltip.type === 'country' && (
        <div className="map-tooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltip.name}</strong>
          <div className="tt-row"><span className="tt-label">ISO</span><span className="tt-value">{tooltip.iso}</span></div>
          <div className="tt-row"><span className="tt-label">Cities</span><span className="tt-value">{tooltip.count}</span></div>
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

// Parse CSS color string to [r, g, b] array
// Handles rgb(r,g,b), rgba(r,g,b,a), and hex (#rrggbb / #rgb) formats
function cssToRgb(css) {
  if (!css) return [160, 160, 160]
  const rgbMatch = css.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/)
  if (rgbMatch) return [parseInt(rgbMatch[1]), parseInt(rgbMatch[2]), parseInt(rgbMatch[3])]
  const hexMatch = css.match(/^#([0-9a-f]{3,8})$/i)
  if (hexMatch) {
    let hex = hexMatch[1]
    if (hex.length === 3 || hex.length === 4) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2]
    return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)]
  }
  return [160, 160, 160]
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
