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

async function aggregatePredictionCsv(filePath) {
  const stream = fs.createReadStream(filePath);
  const reader = readline.createInterface({ input: stream, crlfDelay: Infinity });
  const hourly = Array.from({ length: 24 }, (_, hour) => ({ hour, pickups: 0 }));
  const weekdays = new Map();
  const daily = new Map();
  const topRegions = new Map();
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
    const dayKey = `${record.month}-${record.day}`;
    const regionKey = record.geohash;

    if (Number.isFinite(hour) && hourly[hour]) hourly[hour].pickups += pickups;
    weekdays.set(record.day_cat, (weekdays.get(record.day_cat) ?? 0) + pickups);
    daily.set(dayKey, (daily.get(dayKey) ?? 0) + pickups);
    topRegions.set(regionKey, (topRegions.get(regionKey) ?? 0) + pickups);
  }

  const weekdayOrder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  return {
    hourly,
    weekdays: weekdayOrder.map((weekday) => ({ weekday, pickups: weekdays.get(weekday) ?? 0 })),
    daily: [...daily.entries()].map(([day, pickups]) => ({ day, pickups })),
    topRegions: [...topRegions.entries()]
      .map(([geohash, pickups]) => ({ geohash, pickups }))
      .sort((left, right) => right.pickups - left.pickups)
      .slice(0, 8),
  };
}

function copyAsset(source, targetName) {
  const targetPath = path.join(assetRoot, targetName);
  ensureDir(path.dirname(targetPath));
  fs.copyFileSync(source, targetPath);
  return `/dashboard-assets/${targetName}`;
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
  }

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
    baseDistribution: [...combinedBaseMap.values()]
      .map((item) => ({ ...item, share: Number(((item.total / totalPickups) * 100).toFixed(2)) }))
      .sort((left, right) => right.total - left.total),
    featureInsights: readFeatureInsights(path.join(projectRoot, 'new/feature_analysis_report.md')),
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
