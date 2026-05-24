import { useEffect, useMemo, useState } from 'react'
import { type DashboardData, fetchDashboardData } from '../data/dashboard'

export type ActiveFilter =
  | { type: 'borough'; value: string }
  | { type: 'weekday'; value: string }
  | { type: 'weather'; value: string }
  | null

export type DashboardViewModel = {
  dashboardData: DashboardData
  activeFilter: ActiveFilter
  activeFilterText: string
  selectedBorough: string | null
  selectedWeekday: string | null
  selectedWeather: string | null
  displayedDemandTrend: DashboardData['demandTrend']
  displayedHourlyDemand: DashboardData['hourlyDemand']
  displayedWeekdayDemand: DashboardData['weekdayDemand']
  displayedTopRegions: DashboardData['topRegions']
  displayedWeatherScatter: DashboardData['weatherScatter']
  displayedPrecipImpact: DashboardData['precipImpact']
  mapHotspots: DashboardData['regionPoints']
  actualVsPredictedSeries: DashboardData['demandTrend']
  toggleBoroughFilter: (label: string) => void
  toggleWeekdayFilter: (label: string) => void
  toggleWeatherFilter: (label: string) => void
  handleTopRegionSelect: (label: string) => void
  clearFilter: () => void
}

function formatFilterLabel(activeFilter: ActiveFilter) {
  if (!activeFilter) {
    return '当前展示全量数据。点击地图、星期柱或天气柱可联动筛选。'
  }

  if (activeFilter.type === 'borough') {
    return `当前筛选：${activeFilter.value}，时空、天气和热点区域已联动更新。`
  }

  if (activeFilter.type === 'weekday') {
    return `当前筛选：${activeFilter.value}，小时分布已切换到该星期画像。`
  }

  return `当前筛选：${activeFilter.value}，天气散点已切换到该天气类别。`
}

export function useDashboardState() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [activeFilter, setActiveFilter] = useState<ActiveFilter>(null)

  useEffect(() => {
    let active = true

    fetchDashboardData()
      .then((data) => {
        if (active) {
          setDashboardData(data)
        }
      })
      .catch((error: Error) => {
        if (active) {
          setErrorMessage(error.message)
        }
      })

    return () => {
      active = false
    }
  }, [])

  const viewModel = useMemo<DashboardViewModel | null>(() => {
    if (!dashboardData) {
      return null
    }

    const selectedBorough = activeFilter?.type === 'borough' ? activeFilter.value : null
    const selectedWeekday = activeFilter?.type === 'weekday' ? activeFilter.value : null
    const selectedWeather = activeFilter?.type === 'weather' ? activeFilter.value : null

    const displayedDemandTrend = selectedBorough
      ? (dashboardData.demandTrendByBorough[selectedBorough] ?? dashboardData.demandTrend)
      : dashboardData.demandTrend

    const displayedHourlyDemand = selectedBorough
      ? (dashboardData.hourlyDemandByBorough[selectedBorough] ?? dashboardData.hourlyDemand)
      : selectedWeekday
        ? (dashboardData.weekdayProfiles[selectedWeekday] ?? dashboardData.hourlyDemand)
        : dashboardData.hourlyDemand

    const displayedWeekdayDemand = selectedBorough
      ? (dashboardData.weekdayDemandByBorough[selectedBorough] ?? dashboardData.weekdayDemand)
      : dashboardData.weekdayDemand

    const displayedTopRegions = selectedBorough
      ? (dashboardData.topRegionsByBorough[selectedBorough] ?? dashboardData.topRegions)
      : dashboardData.topRegions

    const displayedWeatherScatter = selectedBorough
      ? (dashboardData.weatherScatterByBorough[selectedBorough] ?? dashboardData.weatherScatter)
      : selectedWeather
        ? (dashboardData.weatherScatterByCategory[selectedWeather] ?? dashboardData.weatherScatter)
        : dashboardData.weatherScatter

    const displayedPrecipImpact = selectedBorough
      ? (dashboardData.precipImpactByBorough[selectedBorough] ?? dashboardData.precipImpact)
      : dashboardData.precipImpact

    const mapHotspots = selectedBorough
      ? (dashboardData.topRegionsByBorough[selectedBorough] ?? dashboardData.regionPoints)
      : dashboardData.regionPoints

    const actualVsPredictedSeries = dashboardData.actualVsPredicted.labels.map((label, index) => ({
      label,
      value: dashboardData.actualVsPredicted.actual[index] ?? 0,
    }))

    function toggleBoroughFilter(label: string) {
      setActiveFilter((current) => {
        if (current?.type === 'borough' && current.value === label) {
          return null
        }

        return { type: 'borough', value: label }
      })
    }

    function toggleWeekdayFilter(label: string) {
      setActiveFilter((current) => {
        if (current?.type === 'weekday' && current.value === label) {
          return null
        }

        return { type: 'weekday', value: label }
      })
    }

    function toggleWeatherFilter(label: string) {
      setActiveFilter((current) => {
        if (current?.type === 'weather' && current.value === label) {
          return null
        }

        return { type: 'weather', value: label }
      })
    }

    function handleTopRegionSelect(label: string) {
      const matched = displayedTopRegions.find((item) => item.label === label)
      if (matched?.borough) {
        toggleBoroughFilter(matched.borough)
      }
    }

    return {
      dashboardData,
      activeFilter,
      activeFilterText: formatFilterLabel(activeFilter),
      selectedBorough,
      selectedWeekday,
      selectedWeather,
      displayedDemandTrend,
      displayedHourlyDemand,
      displayedWeekdayDemand,
      displayedTopRegions,
      displayedWeatherScatter,
      displayedPrecipImpact,
      mapHotspots,
      actualVsPredictedSeries,
      toggleBoroughFilter,
      toggleWeekdayFilter,
      toggleWeatherFilter,
      handleTopRegionSelect,
      clearFilter: () => setActiveFilter(null),
    }
  }, [activeFilter, dashboardData])

  return {
    dashboardData,
    errorMessage,
    viewModel,
  }
}
