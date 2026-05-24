import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

const webRoot = process.cwd();
const projectRoot = path.resolve(webRoot, '..');
const publicRoot = path.join(webRoot, 'public');
const assetRoot = path.join(publicRoot, 'dashboard-assets');
const dataPath = path.join(publicRoot, 'dashboard-data.json');

const monthConfigs = [
  { key: 'may14', label: '2014 年 5 月', shortLabel: '5 月', dir: 'may14', days: 31, gif: 'hourly_pickup_densitymay.gif' },
  { key: 'jun14', label: '2014 年 6 月', shortLabel: '6 月', dir: 'jun14', days: 30, gif: 'hourly_pickup_densityjune.gif' },
];

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function parseCsvLine(line) {
  const result = [];
  let current = '';
  let isQuoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const nextChar = line[index + 1];

    if (char === '"' && isQuoted && nextChar === '"') {
      current += '"';
      index += 1;
      continue;
    }

    if (char === '"') {
      isQuoted = !isQuoted;
      continue;
    }

    if (char === ',' && !isQuoted) {
      result.push(current);
      current = '';
      continue;
    }

    current += char;
  }

  result.push(current);
  return result;
}

function readSingleRowCsv(filePath) {
  const [headerLine, rowLine] = fs.readFileSync(filePath, 'utf8').trim().split(/\r?\n/);
  const headers = parseCsvLine(headerLine);
  const values = parseCsvLine(rowLine);

  return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
}

function readCsvRecords(filePath) {
  if (!fs.existsSync(filePath)) return [];

  const [headerLine, ...lines] = fs.readFileSync(filePath, 'utf8').trim().split(/\r?\n/);
  const headers = parseCsvLine(headerLine);

  return lines
    .filter((line) => line.trim())
    .map((line) => {
      const values = parseCsvLine(line);
      return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
    });
}

function readBaseCounts(filePath) {
  const [, ...rows] = fs.readFileSync(filePath, 'utf8').trim().split(/\r?\n/);

  return rows.map((row) => {
    const [base, pickups] = parseCsvLine(row);
    return { base, pickups: Number(pickups) };
  });
}

function readFeatureInsights(filePath) {
  const lines = fs.readFileSync(filePath, 'utf8').split(/\r?\n/);
  const tableRows = lines.filter((line) => line.startsWith('|') && !line.includes('---'));
  const headerIndex = tableRows.findIndex((line) => line.includes('feature') && line.includes('importance_score'));

  if (headerIndex < 0) return [];

  const headers = tableRows[headerIndex].split('|').map((item) => item.trim()).filter(Boolean);

  return tableRows.slice(headerIndex + 1, headerIndex + 6).map((line) => {
    const values = line.split('|').map((item) => item.trim()).filter(Boolean);
    const record = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));

    return {
      feature: record.feature,
      featureType: record.feature_type,
      importanceScore: Number(record.importance_score),
      pearsonCorr: Number(record.pearson_corr),
      spearmanCorr: Number(record.spearman_corr),
      entropy: Number(record.entropy),
      normalizedEntropy: Number(record.normalized_entropy),
      uniqueCount: Number(record.unique_count),
      missingRate: Number(record.missing_rate),
      sourcePath: path.relative(projectRoot, filePath),
    };
  });
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function round(value, digits = 2) {
  return Number(value.toFixed(digits));
}

function increment(map, key, amount = 1) {
  map.set(key, (map.get(key) ?? 0) + amount);
}

function sampleSeries(entries, maxPoints) {
  if (entries.length <= maxPoints) return entries;

  const sampled = [];
  const lastIndex = entries.length - 1;
  for (let index = 0; index < maxPoints; index += 1) {
    sampled.push(entries[Math.round((index / (maxPoints - 1)) * lastIndex)]);
  }

  return sampled;
}

function parseWeatherHourKey(value) {
  const text = String(value ?? '').trim();
  const [datePart, timePart] = text.split('T');
  if (!datePart || !timePart) return null;

  const hour = Number.parseInt(timePart.slice(0, 2), 10);
  if (!Number.isFinite(hour)) return null;

  return `${datePart} ${String(hour).padStart(2, '0')}:00`;
}

function getWeatherCategory(weatherType, precipitation) {
  const type = String(weatherType ?? '').toUpperCase();

  if (type.includes('FG')) return '雾天';
  if (type.includes('SN')) return '雪天';
  if (type.includes('RA') || precipitation > 0) return '雨天';
  return '晴天';
}

function buildHistogram(values, bucketCount = 6) {
  if (!values.length) return [];

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const span = Math.max(maxValue - minValue, 1);
  const bucketSize = span / bucketCount;
  const buckets = Array.from({ length: bucketCount }, (_, index) => ({
    start: minValue + bucketSize * index,
    end: minValue + bucketSize * (index + 1),
    value: 0,
  }));

  for (const value of values) {
    const bucketIndex = Math.min(Math.floor((value - minValue) / bucketSize), bucketCount - 1);
    buckets[bucketIndex].value += 1;
  }

  return buckets.map((bucket) => ({
    label: `${Math.round(bucket.start)}~${Math.round(bucket.end)}`,
    value: bucket.value,
  }));
}

function buildWeatherScatter(rows) {
  const tempBins = new Map();

  for (const row of rows) {
    const tempKey = Math.round(row.temp);
    const current = tempBins.get(tempKey) ?? [];
    current.push(row.demand);
    tempBins.set(tempKey, current);
  }

  return sampleSeries(
    [...tempBins.entries()]
      .sort((left, right) => left[0] - right[0])
      .map(([temp, values]) => ({ x: temp, y: round(mean(values)), size: 5 })),
    12,
  );
}

function buildWeatherTempBins(rows) {
  const bins = new Map();

  for (const row of rows) {
    const binStart = Math.floor(row.temp / 5) * 5;
    const label = `${binStart}~${binStart + 5}°C`;
    const current = bins.get(label) ?? [];
    current.push(row.demand);
    bins.set(label, current);
  }

  return [...bins.entries()]
    .sort((left, right) => Number.parseInt(left[0], 10) - Number.parseInt(right[0], 10))
    .map(([label, values]) => ({ label, value: Math.round(mean(values)) }));
}

function buildPrecipImpact(rows) {
  const categories = ['晴天', '雨天', '雪天', '雾天'];

  return categories.map((label) => ({
    label,
    value: Math.round(mean(rows.filter((row) => row.category === label).map((row) => row.demand))),
  }));
}

function buildWeatherAggregates(weatherCsvPath, hourlyDemandMap) {
  if (!fs.existsSync(weatherCsvPath)) {
    return { weatherScatter: [], weatherTempBins: [], precipImpact: [] };
  }

  const [headerLine, ...lines] = fs.readFileSync(weatherCsvPath, 'utf8').split(/\r?\n/);
  const headers = parseCsvLine(headerLine);
  const weatherHourly = new Map();

  for (const line of lines) {
    if (!line.trim()) continue;

    const values = parseCsvLine(line);
    const record = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
    const reportType = String(record.REPORT_TYPE ?? '').trim();
    const hourKey = parseWeatherHourKey(record.DATE);

    if (!hourKey || (reportType && !['FM-15', 'FM-16'].includes(reportType))) continue;

    const temp = Number.parseFloat(record.HourlyDryBulbTemperature);
    if (!Number.isFinite(temp)) continue;

    const rawPrecipitation = String(record.HourlyPrecipitation ?? '').trim();
    const precipitation = rawPrecipitation && rawPrecipitation !== 'T' ? Number.parseFloat(rawPrecipitation) : 0;

    weatherHourly.set(hourKey, {
      temp,
      precipitation: Number.isFinite(precipitation) ? precipitation : 0,
      weatherType: record.HourlyPresentWeatherType,
    });
  }

  const joinedRows = [];
  for (const [hourKey, demand] of hourlyDemandMap.entries()) {
    const weather = weatherHourly.get(hourKey);
    if (!weather) continue;

    joinedRows.push({
      demand,
      temp: weather.temp,
      category: getWeatherCategory(weather.weatherType, weather.precipitation),
    });
  }

  return {
    weatherScatter: buildWeatherScatter(joinedRows),
    weatherTempBins: buildWeatherTempBins(joinedRows),
    precipImpact: buildPrecipImpact(joinedRows),
  };
}

function buildPredictionAggregates(hourlyDemandMap) {
  const juneDates = [...new Set([...hourlyDemandMap.keys()].map((key) => key.slice(0, 10)).filter((key) => key.startsWith('2014-06')))].sort();
  const maxJuneDate = juneDates.at(-1);

  if (!maxJuneDate) {
    return {
      predictionMetrics: [],
      actualVsPredicted: { labels: [], actual: [], predicted: [] },
      actualPredictedScatter: [],
      errorHistogram: [],
      residualTrend: [],
      predictionTopErrors: [],
    };
  }

  const maxDate = new Date(`${maxJuneDate}T00:00:00Z`);
  const testStart = new Date(maxDate.getTime() - 6 * 24 * 60 * 60 * 1000);
  const testStartKey = testStart.toISOString().slice(0, 10);
  const predictions = [];
  const dailyActual = new Map();
  const dailyPredicted = new Map();

  for (const [hourKey, actual] of [...hourlyDemandMap.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    const dateKey = hourKey.slice(0, 10);
    if (dateKey < testStartKey || !dateKey.startsWith('2014-06')) continue;

    const previousDate = new Date(`${dateKey}T00:00:00Z`);
    previousDate.setUTCDate(previousDate.getUTCDate() - 1);
    const previousHourKey = `${previousDate.toISOString().slice(0, 10)} ${hourKey.slice(11)}`;
    const predicted = hourlyDemandMap.get(previousHourKey);
    if (predicted == null) continue;

    const error = predicted - actual;
    predictions.push({ actual, predicted, error, dateKey });
    increment(dailyActual, dateKey, actual);
    increment(dailyPredicted, dateKey, predicted);
  }

  const actualValues = predictions.map((item) => item.actual);
  const errors = predictions.map((item) => item.error);
  const mae = mean(errors.map((value) => Math.abs(value)));
  const rmse = Math.sqrt(mean(errors.map((value) => value * value)));
  const actualMean = mean(actualValues);
  const ssRes = errors.reduce((sum, value) => sum + value * value, 0);
  const ssTot = actualValues.reduce((sum, value) => sum + (value - actualMean) ** 2, 0);
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const sortedDays = [...dailyActual.keys()].sort();
  const maxDailyError = Math.max(...sortedDays.map((key) => Math.abs((dailyPredicted.get(key) ?? 0) - (dailyActual.get(key) ?? 0))), 1);

  return {
    predictionMetrics: [
      { label: 'Baseline RMSE', value: rmse.toFixed(2), change: '前一日同小时预测', tone: 'violet' },
      { label: 'Baseline MAE', value: mae.toFixed(2), change: '最后一周小时级误差', tone: 'teal' },
      { label: 'Baseline R²', value: r2.toFixed(2), change: '六月最后一周', tone: 'amber' },
      { label: '预测样本数', value: String(predictions.length), change: '小时粒度', tone: 'sky' },
    ],
    actualVsPredicted: {
      labels: sortedDays.map((dateKey) => dateKey.slice(5)),
      actual: sortedDays.map((dateKey) => Math.round(dailyActual.get(dateKey) ?? 0)),
      predicted: sortedDays.map((dateKey) => Math.round(dailyPredicted.get(dateKey) ?? 0)),
    },
    actualPredictedScatter: sortedDays.map((dateKey) => ({
      x: Math.round(dailyActual.get(dateKey) ?? 0),
      y: Math.round(dailyPredicted.get(dateKey) ?? 0),
      size: 6,
    })),
    errorHistogram: buildHistogram(errors),
    residualTrend: sortedDays.map((dateKey) => ({
      label: dateKey.slice(5),
      value: Math.round((dailyPredicted.get(dateKey) ?? 0) - (dailyActual.get(dateKey) ?? 0)),
    })),
    predictionTopErrors: sortedDays
      .map((dateKey) => {
        const actual = dailyActual.get(dateKey) ?? 0;
        const predicted = dailyPredicted.get(dateKey) ?? 0;
        const absError = Math.abs(predicted - actual);
        return { label: dateKey.slice(5), value: String(absError), score: Math.round((absError / maxDailyError) * 100) };
      })
      .sort((left, right) => right.score - left.score)
      .slice(0, 5),
  };
}

function readXgboostTopRegionPredictions(filePath) {
  if (!fs.existsSync(filePath)) {
    return {
      summary: {
        regionCount: 0,
        sampleCount: 0,
        mae: '0.00',
        rmse: '0.00',
      },
      regions: [],
    };
  }

  const [headerLine, ...lines] = fs.readFileSync(filePath, 'utf8').trim().split(/\r?\n/);
  const headers = parseCsvLine(headerLine);
  const regionMap = new Map();

  for (const line of lines) {
    if (!line.trim()) continue;

    const values = parseCsvLine(line);
    const record = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
    const region = record['地点'];
    const timestamp = record['时间'];
    const predicted = Number.parseFloat(record['预测值']);
    const actual = Number.parseFloat(record['真实值']);

    if (!region || !timestamp || !Number.isFinite(predicted) || !Number.isFinite(actual)) continue;

    const current = regionMap.get(region) ?? [];
    current.push({
      time: timestamp,
      label: `${timestamp.slice(5, 10)} ${timestamp.slice(11, 13)}时`,
      predicted: round(predicted, 2),
      actual,
      error: round(predicted - actual, 2),
    });
    regionMap.set(region, current);
  }

  const allErrors = [];
  const regions = [...regionMap.entries()]
    .map(([region, points]) => {
      const sortedPoints = points.sort((left, right) => left.time.localeCompare(right.time));
      const errors = sortedPoints.map((item) => item.predicted - item.actual);
      allErrors.push(...errors);

      const actualTotal = sortedPoints.reduce((sum, item) => sum + item.actual, 0);
      const predictedTotal = sortedPoints.reduce((sum, item) => sum + item.predicted, 0);
      const mae = mean(errors.map((value) => Math.abs(value)));
      const rmse = Math.sqrt(mean(errors.map((value) => value * value)));
      const maxActual = Math.max(...sortedPoints.map((item) => item.actual), 1);

      return {
        region,
        sampleCount: sortedPoints.length,
        timeRange: `${sortedPoints[0].time.slice(5, 16)} 至 ${sortedPoints.at(-1).time.slice(5, 16)}`,
        actualTotal: Math.round(actualTotal),
        predictedTotal: Math.round(predictedTotal),
        mae: round(mae, 2),
        rmse: round(rmse, 2),
        fitScore: Math.max(0, Math.round((1 - mae / maxActual) * 100)),
        points: sortedPoints,
      };
    })
    .sort((left, right) => left.mae - right.mae);

  const mae = mean(allErrors.map((value) => Math.abs(value)));
  const rmse = Math.sqrt(mean(allErrors.map((value) => value * value)));

  return {
    summary: {
      regionCount: regions.length,
      sampleCount: regions.reduce((sum, region) => sum + region.sampleCount, 0),
      mae: mae.toFixed(2),
      rmse: rmse.toFixed(2),
    },
    regions,
  };
}

function readStidModelData(modelRoot) {
  const metricsPath = path.join(modelRoot, 'stid_best_0.02_top30/metrics.json');
  const refineSummaryPath = path.join(modelRoot, 'stid_refine_summary.csv');
  const sweepSummaryPath = path.join(modelRoot, 'sweep_summary_with_normalized.csv');
  const trainingHistoryPath = path.join(modelRoot, 'stid_best_0.02_top30/training_history.csv');
  const predictionsPath = path.join(modelRoot, 'stid_best_0.02_top30/test_predictions.csv');

  if (!fs.existsSync(metricsPath)) {
    return {
      available: false,
      metrics: [],
      summary: { ordersCovered: 0, meanTrue: '0.00', wape: '0.00', bestName: '' },
      refineRows: [],
      sweepRanking: [],
      trainingLoss: { labels: [], train: [], validation: [] },
      actualVsPredicted: { labels: [], actual: [], predicted: [] },
      topRegionErrors: [],
    };
  }

  const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
  const refineRows = readCsvRecords(refineSummaryPath)
    .map((row) => ({
      name: row.name,
      ordersCovered: Math.round(Number(row.orders_covered)),
      meanTrue: round(Number(row.mean_true), 2),
      mae: round(Number(row.mae), 2),
      rmse: round(Number(row.rmse), 2),
      mape: round(Number(row.mape_percent), 2),
      r2: round(Number(row.r2), 4),
      wape: round(Number(row.wape_percent), 2),
    }))
    .sort((left, right) => right.r2 - left.r2);
  const bestRefine = refineRows[0];

  const sweepRanking = readCsvRecords(sweepSummaryPath)
    .map((row) => ({
      label: row.name.replace(/grid/g, '网格 ').replace(/_/g, ' '),
      value: `R² ${round(Number(row.r2), 3)} / WAPE ${round(Number(row.wape_percent), 2)}%`,
      score: Math.round(Math.max(0, Math.min(Number(row.r2), 1)) * 100),
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 6);

  const trainingRows = sampleSeries(
    readCsvRecords(trainingHistoryPath).map((row) => ({
      label: `E${row.epoch}`,
      train: round(Number(row.train_loss), 4),
      validation: round(Number(row.val_loss), 4),
    })),
    12,
  );

  const timeBuckets = new Map();
  const regionBuckets = new Map();
  for (const row of readCsvRecords(predictionsPath)) {
    const time = row.time_bin;
    const region = row.region_id;
    const actual = Number(row.y_true);
    const predicted = Number(row.y_pred);
    if (!time || !region || !Number.isFinite(actual) || !Number.isFinite(predicted)) continue;

    const timeBucket = timeBuckets.get(time) ?? { actual: 0, predicted: 0 };
    timeBucket.actual += actual;
    timeBucket.predicted += predicted;
    timeBuckets.set(time, timeBucket);

    const regionBucket = regionBuckets.get(region) ?? { actual: 0, predicted: 0, errors: [] };
    regionBucket.actual += actual;
    regionBucket.predicted += predicted;
    regionBucket.errors.push(predicted - actual);
    regionBuckets.set(region, regionBucket);
  }

  const predictionSeries = sampleSeries(
    [...timeBuckets.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([time, value]) => ({
        label: `${time.slice(5, 10)} ${time.slice(11, 13)}时`,
        actual: Math.round(value.actual),
        predicted: Math.round(value.predicted),
      })),
    14,
  );
  const maxRegionError = Math.max(
    ...[...regionBuckets.values()].map((bucket) => mean(bucket.errors.map((error) => Math.abs(error)))),
    1,
  );
  const topRegionErrors = [...regionBuckets.entries()]
    .map(([region, bucket]) => {
      const mae = mean(bucket.errors.map((error) => Math.abs(error)));
      return {
        label: region,
        value: `MAE ${round(mae, 2)} / 真实 ${Math.round(bucket.actual)}`,
        score: Math.round((mae / maxRegionError) * 100),
      };
    })
    .sort((left, right) => right.score - left.score)
    .slice(0, 8);

  return {
    available: true,
    metrics: [
      { label: 'STID MAE', value: Number(metrics.mae).toFixed(2), change: '时空图模型误差', tone: 'teal' },
      { label: 'STID RMSE', value: Number(metrics.rmse).toFixed(2), change: '测试集预测', tone: 'violet' },
      { label: 'STID R²', value: Number(metrics.r2).toFixed(4), change: '拟合优度', tone: 'amber' },
      { label: 'STID MAPE', value: `${Number(metrics.mape_percent).toFixed(2)}%`, change: '相对误差', tone: 'sky' },
    ],
    summary: {
      ordersCovered: bestRefine?.ordersCovered ?? 0,
      meanTrue: bestRefine?.meanTrue.toFixed(2) ?? '0.00',
      wape: bestRefine?.wape.toFixed(2) ?? '0.00',
      bestName: bestRefine?.name ?? '',
    },
    refineRows,
    sweepRanking,
    trainingLoss: {
      labels: trainingRows.map((row) => row.label),
      train: trainingRows.map((row) => row.train),
      validation: trainingRows.map((row) => row.validation),
    },
    actualVsPredicted: {
      labels: predictionSeries.map((row) => row.label),
      actual: predictionSeries.map((row) => row.actual),
      predicted: predictionSeries.map((row) => row.predicted),
    },
    topRegionErrors,
  };
}

async function aggregatePredictionCsv(filePath) {
  const stream = fs.createReadStream(filePath);
  const reader = readline.createInterface({ input: stream, crlfDelay: Infinity });
  const hourly = Array.from({ length: 24 }, (_, hour) => ({ hour, pickups: 0 }));
  const weekdays = new Map();
  const daily = new Map();
  const topRegions = new Map();
  const hourlyDemandMap = new Map();
  const weekdayHour = new Map();
  const geohashStats = new Map();
  const heatGrid = Array.from({ length: 5 }, () => Array.from({ length: 6 }, () => 0));
  let headers = [];

  for await (const line of reader) {
    if (!line.trim()) continue;

    if (headers.length === 0) {
      headers = parseCsvLine(line);
      continue;
    }

    const values = parseCsvLine(line);
    const record = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
    const pickups = Number(record.pickups || 0);
    const hour = Number((record.time_cat || '0').slice(0, 2));
    const dateKey = `${record.year}-${String(record.month).padStart(2, '0')}-${String(record.day).padStart(2, '0')}`;
    const hourKey = `${dateKey} ${String(hour).padStart(2, '0')}:00`;
    const regionKey = record.geohash;
    const latitude = Number(record.latitude);
    const longitude = Number(record.longitude);

    if (Number.isFinite(hour) && hourly[hour]) hourly[hour].pickups += pickups;
    weekdays.set(record.day_cat, (weekdays.get(record.day_cat) ?? 0) + pickups);
    daily.set(dateKey, (daily.get(dateKey) ?? 0) + pickups);
    topRegions.set(regionKey, (topRegions.get(regionKey) ?? 0) + pickups);
    increment(hourlyDemandMap, hourKey, pickups);
    increment(weekdayHour, `${record.day_cat}|${String(hour).padStart(2, '0')}`, pickups);

    if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
      const row = Math.min(Math.max(Math.floor(((40.82 - latitude) / (40.82 - 40.6)) * 5), 0), 4);
      const col = Math.min(Math.max(Math.floor(((longitude + 74.05) / (-73.82 + 74.05)) * 6), 0), 5);
      heatGrid[row][col] += pickups;

      const current = geohashStats.get(regionKey) ?? { geohash: regionKey, pickups: 0, latSum: 0, lonSum: 0 };
      current.pickups += pickups;
      current.latSum += latitude * pickups;
      current.lonSum += longitude * pickups;
      geohashStats.set(regionKey, current);
    }
  }

  const weekdayOrder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const weekdayLabels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
  const geohashPoints = [...geohashStats.values()]
    .filter((item) => item.pickups > 0)
    .sort((left, right) => right.pickups - left.pickups)
    .slice(0, 24)
    .map((item) => ({
      geohash: item.geohash,
      pickups: item.pickups,
      latitude: round(item.latSum / item.pickups, 4),
      longitude: round(item.lonSum / item.pickups, 4),
    }));

  return {
    hourly,
    weekdays: weekdayOrder.map((weekday, index) => ({ weekday: weekdayLabels[index], pickups: weekdays.get(weekday) ?? 0 })),
    daily: [...daily.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([day, pickups]) => ({ day, pickups })),
    topRegions: [...topRegions.entries()]
      .map(([geohash, pickups]) => ({ geohash, pickups }))
      .sort((left, right) => right.pickups - left.pickups)
      .slice(0, 8),
    weekdayHourHeatmap: weekdayOrder.flatMap((weekday, weekdayIndex) =>
      Array.from({ length: 24 }, (_, hour) => ({
        x: String(hour).padStart(2, '0'),
        y: weekdayLabels[weekdayIndex],
        value: weekdayHour.get(`${weekday}|${String(hour).padStart(2, '0')}`) ?? 0,
      })),
    ),
    geohashPoints,
    mapIntensity: heatGrid,
    hourlyDemandEntries: [...hourlyDemandMap.entries()].map(([hourKey, pickups]) => ({ hourKey, pickups })),
  };
}

function copyAsset(source, targetName) {
  const targetPath = path.join(assetRoot, targetName);
  ensureDir(path.dirname(targetPath));
  fs.copyFileSync(source, targetPath);
  return `/dashboard-assets/${targetName}`;
}

function copyOptionalAsset(relativePath, targetName) {
  const source = path.join(projectRoot, relativePath);
  if (!fs.existsSync(source)) return null;

  return copyAsset(source, targetName);
}

function buildStaticAssets() {
  const assets = {
    preprocess: {},
    models: [],
    months: {},
  };

  const preprocessAssets = [
    ['hourlyOrders', 'data_preprocess/output/hourly_orders.png'],
    ['weekdayOrders', 'data_preprocess/output/weekday_orders.png'],
    ['topRegions', 'data_preprocess/output/top_regions.png'],
    ['temperatureOrders', 'data_preprocess/output/temperature_orders.png'],
    ['precipitationOrders', 'data_preprocess/output/precipitation_orders.png'],
  ];

  for (const [key, relativePath] of preprocessAssets) {
    assets.preprocess[key] = copyAsset(path.join(projectRoot, relativePath), `preprocess/${path.basename(relativePath)}`);
  }

  const modelAssets = [
    ['LightGBM 近一周预测', 'new/models/lightgbm_actual_vs_pred_last_week.png'],
    ['LightGBM 特征重要性', 'new/models/lightgbm_feature_importance.png'],
    ['CatBoost 预测对比', 'new/models/catboost_actual_vs_pred.png'],
    ['CatBoost 特征重要性', 'new/models/catboost_feature_importance.png'],
    ['GRU 预测对比', 'new/models/gru_actual_vs_pred.png'],
    ['GRU 损失曲线', 'new/models/gru_loss_curve.png'],
  ];

  for (const [title, relativePath] of modelAssets) {
    assets.models.push({ title, src: copyAsset(path.join(projectRoot, relativePath), `models/${path.basename(relativePath)}`) });
  }

  const optionalXgboostAssets = [
    ['XGBoost 近一周预测', 'new/models/0eef56d882ebd359947ef3b62153d46b.png'],
    ['XGBoost 拟合散点', 'new/models/e339544d38c75b41e22a13007f34bbbd.png'],
    ['XGBoost 近一周预测', 'new/models/xgboost_actual_vs_pred.png'],
    ['XGBoost 近一周预测', 'new/models/xgboost_actual_vs_pred_last_week.png'],
    ['XGBoost 拟合散点', 'new/models/xgboost_actual_vs_pred_scatter.png'],
    ['XGBoost 拟合散点', 'new/models/xgboost_scatter.png'],
  ];
  const copiedXgboostAssets = new Set();

  for (const [title, relativePath] of optionalXgboostAssets) {
    const copiedPath = copyOptionalAsset(relativePath, `models/${path.basename(relativePath)}`);
    if (copiedPath && !copiedXgboostAssets.has(title)) {
      assets.models.push({ title, src: copiedPath });
      copiedXgboostAssets.add(title);
    }
  }

  for (const month of monthConfigs) {
    const monthDir = path.join(projectRoot, 'new', month.dir);

    assets.months[month.key] = {
      pickupsByHour: copyAsset(path.join(monthDir, 'pickups_by_hour.png'), `${month.key}/pickups_by_hour.png`),
      pickupsByDay: copyAsset(path.join(monthDir, 'pickups_by_day.png'), `${month.key}/pickups_by_day.png`),
      pickupsByTime: copyAsset(path.join(monthDir, 'pickups_by_time.png'), `${month.key}/pickups_by_time.png`),
      density: copyAsset(path.join(monthDir, 'pickup_density_geohash.png'), `${month.key}/pickup_density_geohash.png`),
      densityGrid: copyAsset(path.join(monthDir, 'hourly_density_grid.png'), `${month.key}/hourly_density_grid.png`),
      densityGif: copyAsset(path.join(monthDir, month.gif), `${month.key}/${month.gif}`),
    };
  }

  return assets;
}

function formatRatio(current, previous) {
  if (!previous) return 0;
  return Number((((current - previous) / previous) * 100).toFixed(2));
}

async function main() {
  ensureDir(assetRoot);

  const assets = buildStaticAssets();
  const monthData = [];

  for (const month of monthConfigs) {
    const monthDir = path.join(projectRoot, 'new', month.dir);
    const summary = readSingleRowCsv(path.join(monthDir, 'summary.csv'));
    const bases = readBaseCounts(path.join(monthDir, 'base_counts.csv'));
    const aggregates = await aggregatePredictionCsv(path.join(monthDir, 'taxi_prediction_style_aggregated.csv'));
    const totalPickups = Number(summary.total_pickups);
    const peakHour = aggregates.hourly.reduce((best, item) => (item.pickups > best.pickups ? item : best), aggregates.hourly[0]);

    monthData.push({
      key: month.key,
      label: month.label,
      shortLabel: month.shortLabel,
      sourceRows: Number(summary.source_rows),
      validRows: Number(summary.valid_rows),
      invalidRows: Number(summary.invalid_rows),
      aggregatedRows: Number(summary.aggregated_rows),
      totalPickups,
      dailyAverage: Math.round(totalPickups / month.days),
      startDatetime: summary.start_datetime,
      endDatetime: summary.end_datetime,
      uniqueGeohashes: Number(summary.unique_geohashes),
      geohashPrecision: Number(summary.geohash_precision),
      timeBinsPerDay: Number(summary.time_bins_per_day),
      minutesPerBin: Number(summary.minutes_per_bin),
      baseCounts: bases,
      peakHour,
      ...aggregates,
      assets: assets.months[month.key],
    });
  }

  const totalPickups = monthData.reduce((sum, month) => sum + month.totalPickups, 0);
  const totalRegions = Math.max(...monthData.map((month) => month.uniqueGeohashes));
  const combinedBaseMap = new Map();
  const combinedHourly = Array.from({ length: 24 }, (_, hour) => ({ hour, may14: 0, jun14: 0, total: 0 }));
  const combinedDaily = new Map();
  const combinedWeekday = new Map();
  const combinedWeekdayHour = new Map();
  const combinedHourlyDemandMap = new Map();
  const combinedGeohashMap = new Map();
  const combinedMapIntensity = Array.from({ length: 5 }, () => Array.from({ length: 6 }, () => 0));

  for (const month of monthData) {
    for (const base of month.baseCounts) {
      const current = combinedBaseMap.get(base.base) ?? { base: base.base, may14: 0, jun14: 0, total: 0 };
      current[month.key] = base.pickups;
      current.total += base.pickups;
      combinedBaseMap.set(base.base, current);
    }

    for (const item of month.hourly) {
      combinedHourly[item.hour][month.key] = item.pickups;
      combinedHourly[item.hour].total += item.pickups;
    }

    for (const item of month.daily) {
      increment(combinedDaily, item.day, item.pickups);
    }

    for (const item of month.weekdays) {
      increment(combinedWeekday, item.weekday, item.pickups);
    }

    for (const item of month.weekdayHourHeatmap) {
      increment(combinedWeekdayHour, `${item.y}|${item.x}`, item.value);
    }

    for (const item of month.hourlyDemandEntries) {
      increment(combinedHourlyDemandMap, item.hourKey, item.pickups);
    }

    for (const item of month.geohashPoints) {
      const current = combinedGeohashMap.get(item.geohash) ?? { ...item, pickups: 0 };
      current.pickups += item.pickups;
      current.latitude = item.latitude;
      current.longitude = item.longitude;
      combinedGeohashMap.set(item.geohash, current);
    }

    month.mapIntensity.forEach((row, rowIndex) => {
      row.forEach((value, colIndex) => {
        combinedMapIntensity[rowIndex][colIndex] += value;
      });
    });
  }

  const weekdayLabels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
  const weatherAggregates = buildWeatherAggregates(path.join(projectRoot, 'new/LCD_USW00094728_2014.csv'), combinedHourlyDemandMap);
  const predictionAggregates = buildPredictionAggregates(combinedHourlyDemandMap);
  const xgboostTopRegions = readXgboostTopRegionPredictions(path.join(projectRoot, 'new/top_regions_actual_vs_pred_simple.csv'));
  const stidModel = readStidModelData(path.join(projectRoot, 'new_model'));
  const featureInsights = readFeatureInsights(path.join(projectRoot, 'new/feature_analysis_report.md'));
  const maxFeatureScore = Math.max(...featureInsights.map((item) => item.importanceScore), 1);

  const dashboardData = {
    generatedAt: new Date().toISOString(),
    summary: {
      totalPickups,
      totalRegions,
      totalAggregatedRows: monthData.reduce((sum, month) => sum + month.aggregatedRows, 0),
      momGrowth: formatRatio(monthData[1].totalPickups, monthData[0].totalPickups),
      coverageRange: `${monthData[0].startDatetime.slice(0, 10)} 至 ${monthData[1].endDatetime.slice(0, 10)}`,
    },
    months: monthData,
    hourlyComparison: combinedHourly,
    demandTrend: sampleSeries(
      [...combinedDaily.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([label, value]) => ({ label: label.slice(5), value })),
      12,
    ),
    dailyCalendar: [...combinedDaily.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([date, value]) => ({ date, value })),
    weekdayDemand: weekdayLabels.map((label) => ({ label, value: combinedWeekday.get(label) ?? 0 })),
    weekdayHourHeatmap: weekdayLabels.flatMap((weekday) =>
      Array.from({ length: 24 }, (_, hour) => {
        const hourLabel = String(hour).padStart(2, '0');
        return { x: hourLabel, y: weekday, value: combinedWeekdayHour.get(`${weekday}|${hourLabel}`) ?? 0 };
      }),
    ),
    geohashPoints: [...combinedGeohashMap.values()].sort((left, right) => right.pickups - left.pickups).slice(0, 24),
    mapIntensity: combinedMapIntensity,
    baseDistribution: [...combinedBaseMap.values()]
      .map((item) => ({ ...item, share: Number(((item.total / totalPickups) * 100).toFixed(2)) }))
      .sort((left, right) => right.total - left.total),
    featureInsights,
    featureImportance: featureInsights.map((item) => ({
      label: item.feature,
      value: item.importanceScore.toFixed(3),
      score: Math.round((item.importanceScore / maxFeatureScore) * 100),
    })),
    entropyScatter: featureInsights.map((item) => ({
      x: round(item.normalizedEntropy * 100, 2),
      y: round(item.importanceScore * 800, 2),
      size: 5,
      label: item.feature,
    })),
    weatherScatter: weatherAggregates.weatherScatter,
    weatherTempBins: weatherAggregates.weatherTempBins,
    precipImpact: weatherAggregates.precipImpact,
    predictionMetrics: predictionAggregates.predictionMetrics,
    actualVsPredicted: predictionAggregates.actualVsPredicted,
    actualPredictedScatter: predictionAggregates.actualPredictedScatter,
    errorHistogram: predictionAggregates.errorHistogram,
    residualTrend: predictionAggregates.residualTrend,
    predictionTopErrors: predictionAggregates.predictionTopErrors,
    xgboostMetrics: [
      { label: 'XGBoost MAE', value: '99.83', change: '六月最后一周', tone: 'teal' },
      { label: 'XGBoost RMSE', value: '125.51', change: '六月最后一周', tone: 'violet' },
      { label: 'XGBoost R²', value: '0.9426', change: '拟合优度', tone: 'amber' },
      { label: 'Top10 区域样本', value: String(xgboostTopRegions.summary.sampleCount), change: '逐小时预测记录', tone: 'sky' },
    ],
    xgboostTopRegions,
    stidModel,
    dataSources: [
      {
        name: '月度摘要与基础指标',
        paths: ['new/may14/summary.csv', 'new/jun14/summary.csv'],
        usage: '总订单量、有效记录、聚合样本、时间范围、Geohash 覆盖。',
      },
      {
        name: '调度基地分布',
        paths: ['new/may14/base_counts.csv', 'new/jun14/base_counts.csv'],
        usage: '基地订单量、双月汇总占比。',
      },
      {
        name: '半小时 Geohash 聚合订单',
        paths: ['new/may14/taxi_prediction_style_aggregated.csv', 'new/jun14/taxi_prediction_style_aggregated.csv'],
        usage: '小时曲线、峰值时段、热点区域 Top 8。',
      },
      {
        name: '预处理与模型图片',
        paths: ['data_preprocess/output/*.png', 'new/may14/*.png|*.gif', 'new/jun14/*.png|*.gif', 'new/models/*.png'],
        usage: '天气关系图、空间热力图、模型预测/特征重要性图片展示。',
      },
      {
        name: '特征相关性报告',
        paths: ['new/feature_analysis_report.md'],
        usage: '特征名、类型、importance_score、Pearson/Spearman、熵值等表格字段。',
      },
    ],
    assets: {
      preprocess: assets.preprocess,
      models: assets.models,
    },
  };

  ensureDir(publicRoot);
  fs.writeFileSync(dataPath, `${JSON.stringify(dashboardData, null, 2)}\n`);
  console.log(`Dashboard data prepared: ${path.relative(webRoot, dataPath)}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
