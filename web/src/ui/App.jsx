import React, { useEffect, useMemo, useState } from 'react'
import MapView from './MapView.jsx'
import CityPanel from './CityPanel.jsx'
import NeighborhoodPanel from './NeighborhoodPanel.jsx'

const App = () => {
  const [scope, setScope] = useState({ level: 'global' }) // {level: 'global'|'country'|'city', iso?, cityId?}
  const [metric, setMetric] = useState('mass') // 'mass' | 'pop'

  const onSelectCountry = (iso3) => setScope({ level: 'country', iso: iso3 })
  const onSelectCity = (cityId) => setScope((s) => ({ level: 'city', iso: s.iso, cityId }))
  const onReset = () => setScope({ level: 'global' })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr 360px', height: '100%' }}>
      <div style={{ borderRight: '1px solid #ddd', overflow: 'auto', background: '#fff' }}>
        <Header scope={scope} metric={metric} onReset={onReset} onMetricChange={setMetric} />
        <CityPanel scope={scope} onSelectCity={onSelectCity} />
      </div>
      <MapView scope={scope} metric={metric} onSelectCountry={onSelectCountry} onSelectCity={onSelectCity} />
      <div style={{ borderLeft: '1px solid #ddd', overflow: 'auto', background: '#fff' }}>
        <NeighborhoodPanel scope={scope} />
      </div>
    </div>
  )
}

const Header = ({ scope, metric, onReset, onMetricChange }) => {
  return (
    <div className="header">
      <div className="title">Global Urban Scaling</div>
      <div className="crumbs">
        {scope.level === 'global' && 'Global'}
        {scope.level === 'country' && `Global ▸ ${scope.iso}`}
        {scope.level === 'city' && `Global ▸ ${scope.iso} ▸ City ${scope.cityId}`}
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
        <button onClick={onReset}>Reset to Global</button>
        <div style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
          Metric:
          <div className="btn-group">
            <button onClick={() => onMetricChange('mass')} disabled={metric==='mass'}>Built Mass</button>
            <button onClick={() => onMetricChange('pop')} disabled={metric==='pop'}>Population</button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
