import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import booleanPointInPolygon from '@turf/boolean-point-in-polygon'
import { point } from '@turf/helpers'
import JSZip from 'jszip'
import Papa from 'papaparse'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const rootDir = path.resolve(__dirname, '..')
const cacheDir = path.join(rootDir, '.cache')
const outputPath = path.join(rootDir, 'public', 'data', 'dashboard.json')
const publicBoroughGeoPath = path.join(rootDir, 'public', 'data', 'nyc-boroughs.geojson')

const sources = {
  mayZip: {
    url: 'https://raw.githubusercontent.com/05hal/TaxiPredict/dev/new/may14.zip',
    fileName: 'may14.zip',
  },
  junZip: {
    url: 'https://raw.githubusercontent.com/05hal/TaxiPredict/dev/new/jun14.zip',
    fileName: 'jun14.zip',
  },
  weatherCsv: {
    url: 'https://raw.githubusercontent.com/05hal/TaxiPredict/dev/new/LCD_USW00094728_2014.csv',
    fileName: 'LCD_USW00094728_2014.csv',
  },
  featureReport: {
    url: 'https://raw.githubusercontent.com/05hal/TaxiPredict/dev/new/feature_analysis_report.md',
    fileName: 'feature_analysis_report.md',
  },
  boroughGeo: {
    url: 'https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson',
    fileName: 'boroughs.geojson',
  },
}

const MAP_ROWS = 5
const MAP_COLS = 6
const LAT_MIN = 40.6
const LAT_MAX = 40.82
const LON_MIN = -74.05
const LON_MAX = -73.82
const WEEKDAY_LABELS = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
const WEATHER_CATEGORIES = ['晴天', '雨天', '雪天', '雾天']
const BOROUGH_NAMES = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
const GEOHASH_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'

function ensureNumber(value) {
  const num = Number.parseFloat(String(value ?? '').trim())
  return Number.isFinite(num) ? num : null
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function round(value, digits = 2) {
  return Number(value.toFixed(digits))
}

function encodeGeohash(lat, lon, precision = 6) {
  let latMin = -90
  let latMax = 90
  let lonMin = -180
  let lonMax = 180
  let bit = 0
  let ch = 0
  let evenBit = true
  let geohash = ''

  while (geohash.length < precision) {
    if (evenBit) {
      const lonMid = (lonMin + lonMax) / 2
      if (lon >= lonMid) {
        ch = (ch << 1) | 1
        lonMin = lonMid
      } else {
        ch <<= 1
        lonMax = lonMid
      }
    } else {
      const latMid = (latMin + latMax) / 2
      if (lat >= latMid) {
        ch = (ch << 1) | 1
        latMin = latMid
      } else {
        ch <<= 1
        latMax = latMid
      }
    }

    evenBit = !evenBit
    bit += 1

    if (bit === 5) {
      geohash += GEOHASH_BASE32[ch]
      bit = 0
      ch = 0
    }
  }

  return geohash
}

function mean(values) {
  if (!values.length) {
    return 0
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function increment(map, key, amount = 1) {
  map.set(key, (map.get(key) ?? 0) + amount)
}

function createArrayMap(keys, factory) {
  return new Map(keys.map((key) => [key, factory()]))
}

function parseUberDate(value) {
  const [datePart, timePart] = String(value ?? '').trim().split(' ')
  if (!datePart || !timePart) {
    return null
  }

  const [month, day, year] = datePart.split('/').map((item) => Number.parseInt(item, 10))
  const [hour, minute, second] = timePart.split(':').map((item) => Number.parseInt(item, 10))
  if (![month, day, year, hour, minute, second].every(Number.isFinite)) {
    return null
  }

  const dateKey = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`
  return {
    year,
    month,
    day,
    hour,
    minute,
    second,
    dateKey,
    hourKey: `${dateKey} ${String(hour).padStart(2, '0')}:00`,
    utcMs: Date.UTC(year, month - 1, day, hour, minute, second),
  }
}

function parseWeatherHourKey(value) {
  const text = String(value ?? '').trim()
  if (!text) {
    return null
  }

  const [datePart, timePart] = text.split('T')
  if (!datePart || !timePart) {
    return null
  }

  const [year, month, day] = datePart.split('-').map((item) => Number.parseInt(item, 10))
  const [hour] = timePart.split(':').map((item) => Number.parseInt(item, 10))
  if (![year, month, day, hour].every(Number.isFinite)) {
    return null
  }

  return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')} ${String(hour).padStart(2, '0')}:00`
}

function buildRegionCell(lat, lon) {
  if (lat == null || lon == null) {
    return null
  }

  if (lat < LAT_MIN || lat > LAT_MAX || lon < LON_MIN || lon > LON_MAX) {
    return null
  }

  const row = clamp(Math.floor(((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * MAP_ROWS), 0, MAP_ROWS - 1)
  const col = clamp(Math.floor(((lon - LON_MIN) / (LON_MAX - LON_MIN)) * MAP_COLS), 0, MAP_COLS - 1)
  const latCenter = LAT_MAX - (row + 0.5) * ((LAT_MAX - LAT_MIN) / MAP_ROWS)
  const lonCenter = LON_MIN + (col + 0.5) * ((LON_MAX - LON_MIN) / MAP_COLS)

  return {
    key: `${row}-${col}`,
    row,
    col,
    lat: round(latCenter, 3),
    lon: round(lonCenter, 3),
    label: `${latCenter.toFixed(3)}, ${lonCenter.toFixed(3)}`,
  }
}

function buildHistogram(values, binEdges) {
  const counts = new Array(binEdges.length - 1).fill(0)

  for (const value of values) {
    for (let index = 0; index < binEdges.length - 1; index += 1) {
      const left = binEdges[index]
      const right = binEdges[index + 1]
      const isLast = index === binEdges.length - 2
      if (value >= left && (value < right || (isLast && value <= right))) {
        counts[index] += 1
        break
      }
    }
  }

  return counts.map((count, index) => ({ label: `${binEdges[index]}~${binEdges[index + 1]}`, value: count }))
}

function parseMarkdownTable(sectionText) {
  const lines = sectionText
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('|'))

  if (lines.length < 3) {
    return []
  }

  const headers = lines[0]
    .split('|')
    .map((cell) => cell.trim())
    .filter(Boolean)

  return lines.slice(2).map((line) => {
    const values = line
      .split('|')
      .map((cell) => cell.trim())
      .filter(Boolean)

    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']))
  })
}

function sampleSeries(entries, maxPoints) {
  if (entries.length <= maxPoints) {
    return entries
  }

  const sampled = []
  const lastIndex = entries.length - 1
  for (let index = 0; index < maxPoints; index += 1) {
    const sourceIndex = Math.round((index / (maxPoints - 1)) * lastIndex)
    sampled.push(entries[sourceIndex])
  }
  return sampled
}

async function downloadIfMissing({ url, fileName }) {
  await fs.mkdir(cacheDir, { recursive: true })
  const filePath = path.join(cacheDir, fileName)

  try {
    await fs.access(filePath)
    return filePath
  } catch {
    console.log(`Downloading ${fileName}...`)
  }

  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status}`)
  }

  const buffer = Buffer.from(await response.arrayBuffer())
  await fs.writeFile(filePath, buffer)
  return filePath
}

async function unzipTextEntry(zipPath, entryNamePart) {
  const buffer = await fs.readFile(zipPath)
  const zip = await JSZip.loadAsync(buffer)
  const fileEntry = Object.values(zip.files).find((entry) => entry.name.endsWith(entryNamePart))
  if (!fileEntry) {
    throw new Error(`No ${entryNamePart} found in ${zipPath}`)
  }
  return fileEntry.async('string')
}

async function loadBoroughFeatures() {
  const boroughPath = await downloadIfMissing(sources.boroughGeo)
  await fs.mkdir(path.dirname(publicBoroughGeoPath), { recursive: true })
  await fs.copyFile(boroughPath, publicBoroughGeoPath)
  const geoJson = JSON.parse(await fs.readFile(boroughPath, 'utf8'))
  return geoJson.features
}

function createAggregates() {
  return {
    totalOrders: 0,
    dailyCounts: new Map(),
    hourlyCounts: new Array(24).fill(0),
    weekdayCounts: new Array(7).fill(0),
    heatCounts: Array.from({ length: MAP_ROWS }, () => new Array(MAP_COLS).fill(0)),
    regionStats: new Map(),
    geohashStats: new Map(),
    juneDateKeys: [],
    hourlyDemandMap: new Map(),
    boroughCounts: new Map(BOROUGH_NAMES.map((name) => [name, 0])),
    dailyCountsByBorough: createArrayMap(BOROUGH_NAMES, () => new Map()),
    hourlyCountsByBorough: createArrayMap(BOROUGH_NAMES, () => new Array(24).fill(0)),
    weekdayCountsByBorough: createArrayMap(BOROUGH_NAMES, () => new Array(7).fill(0)),
    hourlyDemandMapByBorough: createArrayMap(BOROUGH_NAMES, () => new Map()),
    weekdayHourProfiles: createArrayMap(WEEKDAY_LABELS, () => new Array(24).fill(0)),
  }
}

function classifyBorough(lat, lon, boroughFeatures, coordCache) {
  if (lat == null || lon == null) {
    return null
  }

  const cacheKey = `${lat.toFixed(3)}|${lon.toFixed(3)}`
  if (coordCache.has(cacheKey)) {
    return coordCache.get(cacheKey)
  }

  const pt = point([lon, lat])
  const feature = boroughFeatures.find((item) => booleanPointInPolygon(pt, item))
  const boroughName = feature?.properties?.BoroName ?? null
  coordCache.set(cacheKey, boroughName)
  return boroughName
}

function updateRegionStats(regionStats, cell, boroughName, count = 1) {
  if (!cell) {
    return
  }

  const existing = regionStats.get(cell.key) ?? {
    label: cell.label,
    lat: cell.lat,
    lon: cell.lon,
    count: 0,
    boroughCounts: new Map(),
  }

  existing.count += count
  if (boroughName) {
    increment(existing.boroughCounts, boroughName, count)
  }
  regionStats.set(cell.key, existing)
}

function updateGeohashStats(geohashStats, lat, lon, boroughName, count = 1) {
  if (lat == null || lon == null) {
    return
  }

  const geohash = encodeGeohash(lat, lon, 6)
  const existing = geohashStats.get(geohash) ?? {
    label: geohash,
    count: 0,
    latSum: 0,
    lonSum: 0,
    boroughCounts: new Map(),
  }

  existing.count += count
  existing.latSum += lat * count
  existing.lonSum += lon * count
  if (boroughName) {
    increment(existing.boroughCounts, boroughName, count)
  }
  geohashStats.set(geohash, existing)
}

async function parseCleanedPickupsCsv(csvText, aggregates, boroughFeatures, coordCache) {
  return new Promise((resolve, reject) => {
    Papa.parse(csvText, {
      header: true,
      skipEmptyLines: true,
      fastMode: true,
      step: ({ data }) => {
        const dt = parseUberDate(data.datetime)
        const lat = ensureNumber(data.latitude)
        const lon = ensureNumber(data.longitude)
        if (!dt) {
          return
        }

        aggregates.totalOrders += 1
        increment(aggregates.dailyCounts, dt.dateKey)
        aggregates.hourlyCounts[dt.hour] += 1
        const weekdayIndex = (new Date(dt.utcMs).getUTCDay() + 6) % 7
        aggregates.weekdayCounts[weekdayIndex] += 1
        aggregates.weekdayHourProfiles.get(WEEKDAY_LABELS[weekdayIndex])[dt.hour] += 1

        const boroughName = classifyBorough(lat, lon, boroughFeatures, coordCache)
        if (boroughName) {
          aggregates.boroughCounts.set(boroughName, (aggregates.boroughCounts.get(boroughName) ?? 0) + 1)
          increment(aggregates.dailyCountsByBorough.get(boroughName), dt.dateKey)
          aggregates.hourlyCountsByBorough.get(boroughName)[dt.hour] += 1
          aggregates.weekdayCountsByBorough.get(boroughName)[weekdayIndex] += 1
        }

        const cell = buildRegionCell(lat, lon)
        if (cell) {
          aggregates.heatCounts[cell.row][cell.col] += 1
          updateRegionStats(aggregates.regionStats, cell, boroughName, 1)
        }

        updateGeohashStats(aggregates.geohashStats, lat, lon, boroughName, 1)
      },
      complete: resolve,
      error: reject,
    })
  })
}

async function parseAggregatedCsv(csvText, aggregates, boroughFeatures, coordCache) {
  return new Promise((resolve, reject) => {
    Papa.parse(csvText, {
      header: true,
      skipEmptyLines: true,
      fastMode: true,
      step: ({ data }) => {
        const year = Number.parseInt(String(data.year ?? ''), 10)
        const month = Number.parseInt(String(data.month ?? ''), 10)
        const day = Number.parseInt(String(data.day ?? ''), 10)
        const pickups = Number.parseInt(String(data.pickups ?? ''), 10)
        const hour = Number.parseInt(String(data.time_cat ?? '').slice(0, 2), 10)
        if (![year, month, day, pickups, hour].every(Number.isFinite)) {
          return
        }

        const dateKey = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`
        const hourKey = `${dateKey} ${String(hour).padStart(2, '0')}:00`
        increment(aggregates.hourlyDemandMap, hourKey, pickups)

        const lat = ensureNumber(data.latitude)
        const lon = ensureNumber(data.longitude)
        const boroughName = classifyBorough(lat, lon, boroughFeatures, coordCache)
        if (boroughName) {
          increment(aggregates.hourlyDemandMapByBorough.get(boroughName), hourKey, pickups)
        }

        const cell = buildRegionCell(lat, lon)
        if (cell) {
          updateRegionStats(aggregates.regionStats, cell, boroughName, pickups)
        }

        updateGeohashStats(aggregates.geohashStats, lat, lon, boroughName, pickups)

        if (month === 6) {
          aggregates.juneDateKeys.push(dateKey)
        }
      },
      complete: resolve,
      error: reject,
    })
  })
}

async function buildRawAggregates() {
  const aggregates = createAggregates()
  const mayZipPath = await downloadIfMissing(sources.mayZip)
  const junZipPath = await downloadIfMissing(sources.junZip)
  const boroughFeatures = await loadBoroughFeatures()
  const coordCache = new Map()

  console.log('Parsing May cleaned pickups...')
  await parseCleanedPickupsCsv(await unzipTextEntry(mayZipPath, 'cleaned_pickups.csv'), aggregates, boroughFeatures, coordCache)
  console.log('Parsing June cleaned pickups...')
  await parseCleanedPickupsCsv(await unzipTextEntry(junZipPath, 'cleaned_pickups.csv'), aggregates, boroughFeatures, coordCache)
  console.log('Parsing May aggregated demand...')
  await parseAggregatedCsv(await unzipTextEntry(mayZipPath, 'taxi_prediction_style_aggregated.csv'), aggregates, boroughFeatures, coordCache)
  console.log('Parsing June aggregated demand...')
  await parseAggregatedCsv(await unzipTextEntry(junZipPath, 'taxi_prediction_style_aggregated.csv'), aggregates, boroughFeatures, coordCache)

  return aggregates
}

function buildWeatherScatter(rows) {
  const tempBins = new Map()
  for (const row of rows) {
    const tempKey = Math.round(row.temp)
    if (!tempBins.has(tempKey)) {
      tempBins.set(tempKey, [])
    }
    tempBins.get(tempKey).push(row.demand)
  }

  return sampleSeries(
    [...tempBins.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([temp, values]) => ({ x: temp, y: round(mean(values)), size: 5 })),
    10,
  )
}

function buildPrecipImpact(rows) {
  const categoryValues = new Map(WEATHER_CATEGORIES.map((category) => [category, []]))
  for (const row of rows) {
    categoryValues.get(row.category).push(row.demand)
  }

  return WEATHER_CATEGORIES.map((label) => ({
    label,
    value: Math.round(mean(categoryValues.get(label) ?? [])),
  }))
}

function buildWeatherTempBins(rows) {
  const bins = new Map()
  for (const row of rows) {
    const binStart = Math.floor(row.temp / 5) * 5
    const binLabel = `${binStart}~${binStart + 5}°C`
    if (!bins.has(binLabel)) {
      bins.set(binLabel, [])
    }
    bins.get(binLabel).push(row.demand)
  }

  return [...bins.entries()]
    .sort((a, b) => Number.parseInt(a[0], 10) - Number.parseInt(b[0], 10))
    .map(([label, values]) => ({
      label,
      value: Math.round(mean(values)),
    }))
}

async function buildWeatherAggregates(hourlyDemandMap, hourlyDemandMapByBorough) {
  const weatherPath = await downloadIfMissing(sources.weatherCsv)
  const weatherText = await fs.readFile(weatherPath, 'utf8')
  const weatherHourly = new Map()

  await new Promise((resolve, reject) => {
    Papa.parse(weatherText, {
      header: true,
      skipEmptyLines: true,
      step: ({ data }) => {
        const reportType = String(data.REPORT_TYPE ?? '').trim()
        const hourKey = parseWeatherHourKey(data.DATE)
        if (!hourKey || (reportType && !['FM-15', 'FM-16'].includes(reportType))) {
          return
        }

        weatherHourly.set(hourKey, {
          temp: ensureNumber(data.HourlyDryBulbTemperature),
          vis: ensureNumber(data.HourlyVisibility),
          precip: (() => {
            const text = String(data.HourlyPrecipitation ?? '').trim()
            if (!text || text === 'T') {
              return 0
            }
            return ensureNumber(text) ?? 0
          })(),
          weatherType: String(data.HourlyPresentWeatherType ?? '').toUpperCase(),
        })
      },
      complete: resolve,
      error: reject,
    })
  })

  function joinRows(demandMap) {
    const joinedRows = []
    for (const [hourKey, demand] of demandMap.entries()) {
      const weather = weatherHourly.get(hourKey)
      if (!weather || weather.temp == null) {
        continue
      }

      let category = '晴天'
      if (weather.weatherType.includes('FG')) {
        category = '雾天'
      } else if (weather.weatherType.includes('SN')) {
        category = '雪天'
      } else if (weather.weatherType.includes('RA') || weather.precip > 0) {
        category = '雨天'
      }

      joinedRows.push({
        demand,
        temp: weather.temp,
        vis: weather.vis,
        precip: weather.precip,
        category,
      })
    }
    return joinedRows
  }

  const globalRows = joinRows(hourlyDemandMap)
  const weatherScatterByBorough = {}
  const precipImpactByBorough = {}

  for (const boroughName of BOROUGH_NAMES) {
    const boroughRows = joinRows(hourlyDemandMapByBorough.get(boroughName))
    weatherScatterByBorough[boroughName] = buildWeatherScatter(boroughRows)
    precipImpactByBorough[boroughName] = buildPrecipImpact(boroughRows)
  }

  const weatherScatterByCategory = Object.fromEntries(
    WEATHER_CATEGORIES.map((category) => [
      category,
      buildWeatherScatter(globalRows.filter((row) => row.category === category)),
    ]),
  )

  return {
    weatherScatter: buildWeatherScatter(globalRows),
    precipImpact: buildPrecipImpact(globalRows),
    weatherTempBins: buildWeatherTempBins(globalRows),
    weatherScatterByBorough,
    precipImpactByBorough,
    weatherScatterByCategory,
  }
}

async function buildFeatureAggregates() {
  const reportPath = await downloadIfMissing(sources.featureReport)
  const markdown = await fs.readFile(reportPath, 'utf8')
  const importanceSection = markdown.split('## 2. 与 pickups 相关性最高的前 20 个特征')[1]?.split('## 3.')[0] ?? ''
  const entropySection = markdown.split('## 4. 归一化熵值最高的前 20 个特征')[1]?.split('## 5.')[0] ?? ''
  const importanceRows = parseMarkdownTable(importanceSection)
  const entropyRows = parseMarkdownTable(entropySection)

  const topImportance = importanceRows.slice(0, 5)
  const maxImportance = Math.max(...topImportance.map((row) => Number.parseFloat(row.importance_score || '0')), 1)
  const featureImportance = topImportance.map((row) => {
    const score = Number.parseFloat(row.importance_score || '0')
    return {
      label: row.feature,
      value: score.toFixed(3),
      score: Math.round((score / maxImportance) * 100),
    }
  })

  const entropyScatter = entropyRows.slice(0, 8).map((row) => ({
    x: round(Number.parseFloat(row.normalized_entropy || '0') * 100, 2),
    y: round(Number.parseFloat(row.importance_score || '0') * 800, 2),
    size: 5,
  }))

  return { featureImportance, entropyScatter }
}

function buildPredictionAggregates(hourlyDemandMap, juneDateKeys) {
  const uniqueJuneDates = [...new Set(juneDateKeys)].sort()
  const maxJuneDate = uniqueJuneDates.at(-1)
  if (!maxJuneDate) {
    return {
      predictionMetrics: [],
      actualVsPredicted: { labels: [], actual: [], predicted: [] },
      actualPredictedScatter: [],
      errorHistogram: [],
      residualTrend: [],
      predictionTopErrors: [],
    }
  }

  const maxDate = new Date(`${maxJuneDate}T00:00:00Z`)
  const testStart = new Date(maxDate.getTime() - 6 * 24 * 60 * 60 * 1000)
  const testStartKey = testStart.toISOString().slice(0, 10)
  const hourlyEntries = [...hourlyDemandMap.entries()].sort(([a], [b]) => a.localeCompare(b))
  const predictions = []
  const dailyActual = new Map()
  const dailyPred = new Map()

  for (const [hourKey, actual] of hourlyEntries) {
    const dateKey = hourKey.slice(0, 10)
    if (dateKey < testStartKey || !dateKey.startsWith('2014-06')) {
      continue
    }

    const prevDate = new Date(`${dateKey}T00:00:00Z`)
    prevDate.setUTCDate(prevDate.getUTCDate() - 1)
    const previousHourKey = `${prevDate.toISOString().slice(0, 10)} ${hourKey.slice(11)}`
    const predicted = hourlyDemandMap.get(previousHourKey)
    if (predicted == null) {
      continue
    }

    const error = predicted - actual
    predictions.push({ actual, predicted, error, dateKey })
    increment(dailyActual, dateKey, actual)
    increment(dailyPred, dateKey, predicted)
  }

  const actualValues = predictions.map((item) => item.actual)
  const errors = predictions.map((item) => item.error)
  const mae = mean(errors.map((value) => Math.abs(value)))
  const rmse = Math.sqrt(mean(errors.map((value) => value * value)))
  const actualMean = mean(actualValues)
  const ssRes = errors.reduce((sum, value) => sum + value * value, 0)
  const ssTot = actualValues.reduce((sum, value) => sum + (value - actualMean) ** 2, 0)
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot

  const sortedDays = [...dailyActual.keys()].sort()
  const maxDailyError = Math.max(...sortedDays.map((key) => Math.abs((dailyPred.get(key) ?? 0) - (dailyActual.get(key) ?? 0))), 1)

  return {
    predictionMetrics: [
      { label: 'Baseline RMSE', value: rmse.toFixed(2), change: '前一日同小时预测', tone: 'violet' },
      { label: 'Baseline MAE', value: mae.toFixed(2), change: '最后一周小时级误差', tone: 'teal' },
      { label: 'Baseline R2', value: r2.toFixed(2), change: '基于六月最后一周', tone: 'amber' },
      { label: '预测样本数', value: String(predictions.length), change: '小时粒度', tone: 'sky' },
    ],
    actualVsPredicted: {
      labels: sortedDays.map((dateKey) => dateKey.slice(5)),
      actual: sortedDays.map((dateKey) => Math.round(dailyActual.get(dateKey) ?? 0)),
      predicted: sortedDays.map((dateKey) => Math.round(dailyPred.get(dateKey) ?? 0)),
    },
    actualPredictedScatter: sortedDays.map((dateKey) => ({
      x: Math.round(dailyActual.get(dateKey) ?? 0),
      y: Math.round(dailyPred.get(dateKey) ?? 0),
      size: 6,
    })),
    errorHistogram: buildHistogram(errors, [-80, -40, -20, 0, 20, 40, 80]),
    residualTrend: sortedDays.map((dateKey) => ({
      label: dateKey.slice(5),
      value: Math.round((dailyPred.get(dateKey) ?? 0) - (dailyActual.get(dateKey) ?? 0)),
    })),
    predictionTopErrors: sortedDays
      .map((dateKey) => {
        const actual = dailyActual.get(dateKey) ?? 0
        const predicted = dailyPred.get(dateKey) ?? 0
        const absError = Math.abs(predicted - actual)
        return {
          label: dateKey.slice(5),
          value: `${absError}`,
          score: Math.round((absError / maxDailyError) * 100),
        }
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 5),
  }
}

function buildSeriesFromMap(countMap, slicer = (key) => key) {
  return [...countMap.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([label, value]) => ({ label: slicer(label), value }))
}

function buildTopRegionObjects(regionStats, boroughName = null, limit = 5) {
  const regions = [...regionStats.values()]
    .map((region) => {
      const sortedBoroughs = [...region.boroughCounts.entries()].sort((a, b) => b[1] - a[1])
      const dominantBorough = sortedBoroughs[0]?.[0] ?? 'Unknown'
      return {
        label: region.label,
        value: `${Math.round(region.count / 1000)}k`,
        score: region.count,
        lat: region.lat,
        lon: region.lon,
        borough: dominantBorough,
        count: region.count,
      }
    })
    .filter((region) => !boroughName || region.borough === boroughName)
    .sort((a, b) => b.count - a.count)

  const maxCount = Math.max(...regions.map((item) => item.count), 1)
  return regions.slice(0, limit).map((item) => ({
    ...item,
    score: Math.round((item.count / maxCount) * 100),
  }))
}

function buildGeohashObjects(geohashStats, boroughName = null, limit = 60) {
  const geohashes = [...geohashStats.values()]
    .map((item) => {
      const sortedBoroughs = [...item.boroughCounts.entries()].sort((a, b) => b[1] - a[1])
      const dominantBorough = sortedBoroughs[0]?.[0] ?? 'Unknown'
      return {
        label: item.label,
        value: `${Math.round(item.count).toLocaleString('en-US')}`,
        score: item.count,
        lat: round(item.latSum / item.count, 4),
        lon: round(item.lonSum / item.count, 4),
        borough: dominantBorough,
        count: item.count,
      }
    })
    .filter((item) => !boroughName || item.borough === boroughName)
    .sort((a, b) => b.count - a.count)

  const maxCount = Math.max(...geohashes.map((item) => item.count), 1)
  return geohashes.slice(0, limit).map((item) => ({
    ...item,
    score: Math.round((item.count / maxCount) * 100),
  }))
}

function buildCalendarSeries(countMap) {
  return [...countMap.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, value]) => ({ date, value }))
}

function buildWeekdayHourHeatmap(weekdayHourProfiles) {
  return WEEKDAY_LABELS.flatMap((weekday) =>
    weekdayHourProfiles.get(weekday).map((value, hour) => ({
      x: String(hour).padStart(2, '0'),
      y: weekday,
      value,
    })),
  )
}

async function main() {
  await fs.mkdir(path.dirname(outputPath), { recursive: true })

  const rawAggregates = await buildRawAggregates()
  const weatherAggregates = await buildWeatherAggregates(
    rawAggregates.hourlyDemandMap,
    rawAggregates.hourlyDemandMapByBorough,
  )
  const featureAggregates = await buildFeatureAggregates()
  const predictionAggregates = buildPredictionAggregates(rawAggregates.hourlyDemandMap, rawAggregates.juneDateKeys)

  const demandTrend = sampleSeries(buildSeriesFromMap(rawAggregates.dailyCounts, (label) => label.slice(5)), 10)
  const dailyCalendar = buildCalendarSeries(rawAggregates.dailyCounts)
  const hourlyDemand = rawAggregates.hourlyCounts.map((value, hour) => ({ label: String(hour).padStart(2, '0'), value }))
  const weekdayDemand = rawAggregates.weekdayCounts.map((value, index) => ({ label: WEEKDAY_LABELS[index], value }))
  const topRegions = buildTopRegionObjects(rawAggregates.regionStats, null, 5)
  const regionPoints = buildTopRegionObjects(rawAggregates.regionStats, null, 20)
  const geohashPoints = buildGeohashObjects(rawAggregates.geohashStats, null, 80)
  const weekdayHourHeatmap = buildWeekdayHourHeatmap(rawAggregates.weekdayHourProfiles)

  const boroughDemand = BOROUGH_NAMES.map((name) => ({ name, value: rawAggregates.boroughCounts.get(name) ?? 0 }))
  const demandTrendByBorough = Object.fromEntries(
    BOROUGH_NAMES.map((name) => [
      name,
      sampleSeries(buildSeriesFromMap(rawAggregates.dailyCountsByBorough.get(name), (label) => label.slice(5)), 10),
    ]),
  )
  const hourlyDemandByBorough = Object.fromEntries(
    BOROUGH_NAMES.map((name) => [
      name,
      rawAggregates.hourlyCountsByBorough.get(name).map((value, hour) => ({ label: String(hour).padStart(2, '0'), value })),
    ]),
  )
  const weekdayDemandByBorough = Object.fromEntries(
    BOROUGH_NAMES.map((name) => [
      name,
      rawAggregates.weekdayCountsByBorough.get(name).map((value, index) => ({ label: WEEKDAY_LABELS[index], value })),
    ]),
  )
  const topRegionsByBorough = Object.fromEntries(BOROUGH_NAMES.map((name) => [name, buildTopRegionObjects(rawAggregates.regionStats, name, 5)]))
  const weekdayProfiles = Object.fromEntries(
    WEEKDAY_LABELS.map((label) => [
      label,
      rawAggregates.weekdayHourProfiles.get(label).map((value, hour) => ({ label: String(hour).padStart(2, '0'), value })),
    ]),
  )

  const dashboardData = {
    generatedAt: new Date().toISOString(),
    overviewMetrics: [
      { label: '原始订单量', value: rawAggregates.totalOrders.toLocaleString('en-US'), change: `${rawAggregates.dailyCounts.size} 天样本`, tone: 'sky' },
      { label: '热点网格数', value: String(rawAggregates.regionStats.size), change: '来自真实坐标聚合', tone: 'teal' },
      { label: '高价值特征', value: String(featureAggregates.featureImportance.length), change: '来自分析报告 Top5', tone: 'violet' },
      { label: '测试窗口', value: '最后 7 天', change: '六月最后一周', tone: 'amber' },
    ],
    demandTrend,
    demandTrendByBorough,
    dailyCalendar,
    hourlyDemand,
    hourlyDemandByBorough,
    weekdayDemand,
    weekdayDemandByBorough,
    weekdayProfiles,
    weekdayHourHeatmap,
    topRegions,
    topRegionsByBorough,
    regionPoints,
    geohashPoints,
    boroughDemand,
    mapIntensity: rawAggregates.heatCounts,
    weatherScatter: weatherAggregates.weatherScatter,
    weatherScatterByBorough: weatherAggregates.weatherScatterByBorough,
    weatherScatterByCategory: weatherAggregates.weatherScatterByCategory,
    precipImpact: weatherAggregates.precipImpact,
    precipImpactByBorough: weatherAggregates.precipImpactByBorough,
    weatherTempBins: weatherAggregates.weatherTempBins,
    featureImportance: featureAggregates.featureImportance,
    entropyScatter: featureAggregates.entropyScatter,
    predictionMetrics: predictionAggregates.predictionMetrics,
    actualVsPredicted: predictionAggregates.actualVsPredicted,
    actualPredictedScatter: predictionAggregates.actualPredictedScatter,
    errorHistogram: predictionAggregates.errorHistogram,
    residualTrend: predictionAggregates.residualTrend,
    predictionTopErrors: predictionAggregates.predictionTopErrors,
  }

  await fs.writeFile(outputPath, `${JSON.stringify(dashboardData, null, 2)}\n`, 'utf8')
  console.log(`Dashboard data written to ${outputPath}`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
