import React, { useState } from 'react'
import MapView from './MapView.jsx'
import CityPanel from './CityPanel.jsx'
import NeighborhoodPanel from './NeighborhoodPanel.jsx'

const App = () => {
  const [scope, setScope] = useState({ level: 'global' })
  const [metric, setMetric] = useState('mass')
  const [cityName, setCityName] = useState(null)
  const [countryName, setCountryName] = useState(null)

  const onSelectCountry = (iso3, name) => {
    setScope({ level: 'country', iso: iso3 })
    setCountryName(name || iso3)
    setCityName(null)
  }
  const onSelectCity = (cityId, iso, cntryName, cityNameArg) => {
    setScope((s) => ({ level: 'city', iso: iso || s.iso, cityId }))
    setCityName(cityNameArg || `City ${cityId}`)
    if (cntryName) setCountryName(cntryName)
  }
  const onReset = () => {
    setScope({ level: 'global' })
    setCityName(null)
    setCountryName(null)
  }
  const onResetToCountry = () => {
    setScope((s) => ({ level: 'country', iso: s.iso }))
    setCityName(null)
  }

  return (
    <div className="app-layout">
      <MapView scope={scope} metric={metric} onSelectCountry={onSelectCountry} onSelectCity={onSelectCity} onReset={onReset} />

      <div className="panel panel-right">
        <div className="site-header">
          <div className="site-title">Nested Scaling of Urban Material Stocks</div>
          <div className="site-subtitle"><a href="https://kangning-huang.github.io/main/" target="_blank" rel="noopener noreferrer">Kangning Huang</a> (NYU Shanghai, <a href="mailto:kh3657@nyu.edu">kh3657@nyu.edu</a>) &amp; <a href="https://www.mingzhenlu-lab.com" target="_blank" rel="noopener noreferrer">Mingzhen Lu</a> (NYU, <a href="mailto:ml9120@nyu.edu">ml9120@nyu.edu</a>)</div>
          <div className="site-subtitle">Urban material stocks scale sublinearly with population at both city and neighborhood levels, revealing a universal pattern of resource efficiency in larger urban systems. <a href="https://arxiv.org/abs/2507.03960" target="_blank" rel="noopener noreferrer">Read the preprint &rarr;</a> &middot; <a href="https://github.com/kangning-huang/nested-scaling-city-mass" target="_blank" rel="noopener noreferrer">View on GitHub &rarr;</a></div>
          <div className="header-row">
            <div className="breadcrumbs">
              <span className={`crumb${scope.level === 'global' ? ' active' : ''}`} onClick={onReset}>Global</span>
              {scope.level !== 'global' && (
                <>
                  <span className="sep">/</span>
                  <span className={`crumb${scope.level === 'country' ? ' active' : ''}`} onClick={onResetToCountry}>
                    {countryName || scope.iso}
                  </span>
                </>
              )}
              {scope.level === 'city' && (
                <>
                  <span className="sep">/</span>
                  <span className="crumb active">{cityName}</span>
                </>
              )}
            </div>
            <div className="controls-bar">
              <span className="control-label">Map</span>
              <div className="toggle-group">
                <button onClick={() => setMetric('mass')} disabled={metric === 'mass'}>Mass</button>
                <button onClick={() => setMetric('pop')} disabled={metric === 'pop'}>Pop</button>
              </div>
            </div>
          </div>
        </div>

        <div className="section section-city">
          <CityPanel scope={scope} onSelectCity={onSelectCity} countryName={countryName} />
        </div>
        <div className="section-divider" />
        <div className="section section-neigh">
          <NeighborhoodPanel scope={scope} cityName={cityName} countryName={countryName} />
        </div>
      </div>
    </div>
  )
}

export default App
