import { useState, useEffect } from 'react'

const LIGHT = {
  // Canvas chrome
  canvasBg: '#ffffff',
  axis: '#d8d3cc',
  grid: '#f0ede9',
  tickText: '#8a857f',
  labelText: '#6b6560',
  emptyText: '#9a948e',
  zeroLine: '#d0cdc8',
  // City scatter
  cityPoint: 'rgba(141, 160, 203, 0.5)',
  cityDensity: (v) => `rgba(80, 100, 170, ${Math.min(0.92, 0.06 + v * 0.86)})`,
  cityLine: '#e07030',
  cityBand: 'rgba(224, 112, 48, 0.15)',
  // Neighborhood scatter
  neighPoint: 'rgba(141, 160, 203, 0.4)',
  neighDensity: (v) => `rgba(80, 100, 170, ${Math.min(0.92, 0.03 + v * 0.89)})`,
  neighLine: '#bf0000',
  neighBand: 'rgba(191, 0, 0, 0.15)',
}

const DARK = {
  // Canvas chrome
  canvasBg: '#222228',
  axis: '#3e3e48',
  grid: '#2c2c34',
  tickText: '#78736d',
  labelText: '#8a857f',
  emptyText: '#5a5550',
  zeroLine: '#484850',
  // City scatter
  cityPoint: 'rgba(160, 180, 230, 0.55)',
  cityDensity: (v) => `rgba(130, 160, 230, ${Math.min(0.92, 0.06 + v * 0.86)})`,
  cityLine: '#f09050',
  cityBand: 'rgba(240, 144, 80, 0.2)',
  // Neighborhood scatter
  neighPoint: 'rgba(160, 180, 230, 0.45)',
  neighDensity: (v) => `rgba(130, 160, 230, ${Math.min(0.92, 0.03 + v * 0.89)})`,
  neighLine: '#e04040',
  neighBand: 'rgba(224, 64, 64, 0.2)',
}

export function useTheme() {
  const [isDark, setIsDark] = useState(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches
  )

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = (e) => setIsDark(e.matches)
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [])

  return { isDark, chartColors: isDark ? DARK : LIGHT }
}
