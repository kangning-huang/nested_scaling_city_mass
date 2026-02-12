import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './ui/App.jsx'
import './ui/styles.css'

const root = createRoot(document.getElementById('root'))
root.render(<App />)
