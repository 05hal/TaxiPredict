import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import './App.css'
import { DashboardLayout } from './components/layout/DashboardLayout'
import { FeaturesPage } from './pages/FeaturesPage'
import { OverviewPage } from './pages/OverviewPage'
import { PredictionPage } from './pages/PredictionPage'
import { SpatialPage } from './pages/SpatialPage'
import { WeatherPage } from './pages/WeatherPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<DashboardLayout />}>
          <Route index element={<OverviewPage />} />
          <Route path="spatial" element={<SpatialPage />} />
          <Route path="weather" element={<WeatherPage />} />
          <Route path="features" element={<FeaturesPage />} />
          <Route path="prediction" element={<PredictionPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
