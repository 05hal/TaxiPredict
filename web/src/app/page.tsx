"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import type { LayerGroup, Map as LeafletMap } from "leaflet";
import {
  Card,
  ConfigProvider,
  Modal,
  Progress,
  Segmented,
  Space,
  Tag,
  Typography,
} from "antd";
import dashboardJson from "../../public/dashboard-data.json";

interface HourlyPoint {
  hour: number;
  pickups: number;
}

interface SeriesPoint {
  label: string;
  value: number;
}

interface CalendarPoint {
  date: string;
  value: number;
}

interface ScatterPoint {
  x: number;
  y: number;
  size?: number;
  label?: string;
}

interface HeatmapPoint {
  x: string;
  y: string;
  value: number;
}

interface RankedPoint {
  label: string;
  value: string;
  score: number;
}

interface MetricItem {
  label: string;
  value: string;
  change: string;
  tone?: "sky" | "teal" | "violet" | "amber";
}

interface TreeModelPredictionPoint {
  time: string;
  label: string;
  predicted: number;
  actual: number;
  error: number;
}

interface TreeModelRegion {
  region: string;
  sampleCount: number;
  timeRange: string;
  actualTotal: number;
  predictedTotal: number;
  mae: number;
  rmse: number;
  fitScore: number;
  points: TreeModelPredictionPoint[];
}

interface TreeModelMapPoint {
  geohash: string;
  latitude: number;
  longitude: number;
  predicted: number;
  actual: number;
  error: number;
  absError: number;
  label: string;
}

interface TreeModelMapFrame {
  time: string;
  label: string;
  points: TreeModelMapPoint[];
}

interface TreeModelData {
  key: string;
  label: string;
  metrics: MetricItem[];
  summary: {
    regionCount: number;
    sampleCount: number;
    mae: string;
    rmse: string;
    r2: string;
    topFeature: string;
  };
  regions: TreeModelRegion[];
  featureRanking: RankedPoint[];
  mapFrames: TreeModelMapFrame[];
}

interface StidModelData {
  available: boolean;
  metrics: MetricItem[];
  summary: {
    ordersCovered: number;
    meanTrue: string;
    wape: string;
    bestName: string;
  };
  refineRows: Array<{
    name: string;
    ordersCovered: number;
    meanTrue: number;
    mae: number;
    rmse: number;
    mape: number;
    r2: number;
    wape: number;
  }>;
  sweepRanking: RankedPoint[];
  trainingLoss: {
    labels: string[];
    train: number[];
    validation: number[];
  };
  actualVsPredicted: {
    labels: string[];
    actual: number[];
    predicted: number[];
  };
  topRegionErrors: RankedPoint[];
}

interface MonthData {
  key: "may14" | "jun14";
  label: string;
  shortLabel: string;
  sourceRows: number;
  validRows: number;
  invalidRows: number;
  aggregatedRows: number;
  totalPickups: number;
  dailyAverage: number;
  startDatetime: string;
  endDatetime: string;
  uniqueGeohashes: number;
  geohashPrecision: number;
  timeBinsPerDay: number;
  minutesPerBin: number;
  peakHour: HourlyPoint;
  hourly: HourlyPoint[];
  weekdays: Array<{ weekday: string; pickups: number }>;
  daily: Array<{ day: string; pickups: number }>;
  topRegions: Array<{ geohash: string; pickups: number }>;
  weekdayHourHeatmap: HeatmapPoint[];
  geohashPoints: Array<{
    geohash: string;
    pickups: number;
    latitude: number;
    longitude: number;
  }>;
  mapIntensity: number[][];
  hourlyDemandEntries: Array<{ hourKey: string; pickups: number }>;
  assets: {
    pickupsByHour: string;
    pickupsByDay: string;
    pickupsByTime: string;
    density: string;
    densityGrid: string;
    densityGif: string;
  };
}

interface BaseDistribution {
  base: string;
  may14: number;
  jun14: number;
  total: number;
  share: number;
}

interface FeatureInsight {
  feature: string;
  featureType: string;
  importanceScore: number;
  pearsonCorr: number;
  spearmanCorr: number;
  entropy: number;
  normalizedEntropy: number;
  uniqueCount: number;
  missingRate: number;
}

interface DashboardData {
  generatedAt: string;
  summary: {
    totalPickups: number;
    totalRegions: number;
    totalAggregatedRows: number;
    momGrowth: number;
    coverageRange: string;
  };
  months: MonthData[];
  hourlyComparison: Array<{
    hour: number;
    may14: number;
    jun14: number;
    total: number;
  }>;
  demandTrend: SeriesPoint[];
  dailyCalendar: CalendarPoint[];
  weekdayDemand: SeriesPoint[];
  weekdayHourHeatmap: HeatmapPoint[];
  geohashPoints: Array<{
    geohash: string;
    pickups: number;
    latitude: number;
    longitude: number;
  }>;
  mapIntensity: number[][];
  baseDistribution: BaseDistribution[];
  featureInsights: FeatureInsight[];
  featureImportance: RankedPoint[];
  entropyScatter: ScatterPoint[];
  weatherScatter: ScatterPoint[];
  weatherTempBins: SeriesPoint[];
  precipImpact: SeriesPoint[];
  predictionMetrics: MetricItem[];
  actualVsPredicted: {
    labels: string[];
    actual: number[];
    predicted: number[];
  };
  actualPredictedScatter: ScatterPoint[];
  errorHistogram: SeriesPoint[];
  residualTrend: SeriesPoint[];
  predictionTopErrors: RankedPoint[];
  treeModels: TreeModelData[];
  xgboostMetrics: MetricItem[];
  xgboostTopRegions: {
    summary: {
      regionCount: number;
      sampleCount: number;
      mae: string;
      rmse: string;
      r2: string;
      topFeature: string;
    };
    regions: TreeModelRegion[];
  };
  stidModel: StidModelData;
  assets: {
    preprocess: Record<string, string>;
    models: Array<{ title: string; src: string }>;
  };
}

const dashboardData = dashboardJson as unknown as DashboardData;
const treeModels: TreeModelData[] = Array.isArray(dashboardData.treeModels)
  ? dashboardData.treeModels
  : [
      {
        key: "xgboost",
        label: "XGBoost",
        metrics: dashboardData.xgboostMetrics ?? [],
        summary: {
          regionCount:
            dashboardData.xgboostTopRegions?.summary?.regionCount ?? 0,
          sampleCount:
            dashboardData.xgboostTopRegions?.summary?.sampleCount ?? 0,
          mae: dashboardData.xgboostTopRegions?.summary?.mae ?? "0.00",
          rmse: dashboardData.xgboostTopRegions?.summary?.rmse ?? "0.00",
          r2: dashboardData.xgboostTopRegions?.summary?.r2 ?? "0.0000",
          topFeature:
            dashboardData.xgboostTopRegions?.summary?.topFeature ?? "--",
        },
        regions: dashboardData.xgboostTopRegions?.regions ?? [],
        featureRanking: [],
        mapFrames: [],
      },
    ];
const modelAssets = Array.isArray(dashboardData.assets?.models)
  ? dashboardData.assets.models
  : [];
const numberFormatter = new Intl.NumberFormat("zh-CN");
const icpRecord = "京ICP备2025149122号-1";
const icpRecordUrl = "https://beian.miit.gov.cn/";
const publicSecurityRecord = "京公网安备11010802046540号";
const publicSecurityRecordUrl =
  "http://www.beian.gov.cn/portal/registerSystemInfo?recordcode=11010802046540";
const icpRecordIconUrl =
  "https://gw.alicdn.com/tfs/TB1GxwdSXXXXXa.aXXXXXXXXXXX-65-70.gif";
const publicSecurityRecordIconUrl =
  "https://img.alicdn.com/tfs/TB1..50QpXXXXX7XpXXXXXXXXXX-40-40.png";
const pageItems = [
  { key: "overview", label: "运营总览" },
  { key: "operations", label: "运营结构" },
  { key: "spatial", label: "空间热力" },
  { key: "features", label: "天气特征" },
  { key: "models", label: "模型结果" },
] as const;

const featureDisplayNames: Record<string, string> = {
  day_cos: "周内周期余弦",
  day_sin: "周内周期正弦",
  is_evening_peak: "晚高峰标记",
  is_morning_peak: "早高峰标记",
  is_peak: "高峰时段标记",
  is_precip: "降水标记",
  is_rain: "雨天标记",
  longitude: "经度位置",
  precip: "降水量",
  precip_level: "降水等级",
  precip_rolling_3h: "近 3 小时降水",
  rhum: "相对湿度",
  temp_change_1h: "短时温度变化",
  time_cat: "半小时出行时段",
  time_cos: "日内周期余弦",
  time_num: "日内时间进度",
  time_sin: "日内周期强度",
  vis: "能见度",
  weather_severity: "天气影响强度",
  weather_severity_rolling_3h: "近 3 小时天气强度",
  weekend: "周末标记",
};

function formatNumber(value: number) {
  return numberFormatter.format(value);
}

function getPeakLabel(hour: number) {
  return `${String(hour).padStart(2, "0")}:00 - ${String((hour + 1) % 24).padStart(2, "0")}:00`;
}

function getFeatureDisplayName(feature: string) {
  return (
    featureDisplayNames[feature] ??
    feature.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase())
  );
}

function getModelTagColor(modelKey: string) {
  if (modelKey === "xgboost") return "gold";
  if (modelKey === "lightgbm") return "green";
  if (modelKey === "catboost") return "magenta";
  return "cyan";
}

function getPredictionModeLabel(mode: PredictionMapMode) {
  if (mode === "actual") return "真实值";
  if (mode === "error") return "误差";
  return "预测值";
}

type PredictionMapMode = "predicted" | "actual" | "error";

function SectionTitle({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mb-5">
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.34em] text-cyan-300/80">
        {eyebrow}
      </p>
      <Typography.Title
        level={2}
        className="!mb-2 !text-2xl !text-white sm:!text-3xl"
      >
        {title}
      </Typography.Title>
      <Typography.Paragraph className="!mb-0 max-w-3xl !text-sm !leading-7 !text-slate-300">
        {description}
      </Typography.Paragraph>
    </div>
  );
}

function GlassCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Card
      className={`border-white/10 bg-white/[0.08] shadow-2xl shadow-cyan-950/20 backdrop-blur ${className}`}
    >
      {children}
    </Card>
  );
}

function MetricCard({
  label,
  value,
  suffix,
  caption,
}: {
  label: string;
  value: number | string;
  suffix?: string;
  caption: string;
}) {
  return (
    <GlassCard className="h-full">
      <p className="mb-2 text-sm text-slate-300">{label}</p>
      <div className="text-3xl font-black tracking-[-0.04em] text-slate-50">
        {typeof value === "number" ? formatNumber(value) : value}
        {suffix ? (
          <span className="ml-1 text-lg text-slate-300">{suffix}</span>
        ) : null}
      </div>
      <p className="mt-3 text-xs leading-6 text-slate-400">{caption}</p>
    </GlassCard>
  );
}

function HourlyBars({
  data,
}: {
  data: Array<{ hour: number; may14: number; jun14: number; total: number }>;
}) {
  const maxValue = Math.max(...data.map((item) => item.total));

  return (
    <div className="flex h-64 items-end gap-1 rounded-3xl border border-white/10 bg-slate-950/70 p-4">
      {data.map((item) => (
        <div
          key={item.hour}
          className="group flex h-full flex-1 flex-col justify-end gap-1"
        >
          <div className="relative flex flex-1 items-end rounded-full bg-white/5">
            <div
              className="w-full rounded-full bg-gradient-to-t from-cyan-500 via-sky-400 to-amber-200 transition-all duration-300 group-hover:from-amber-300 group-hover:to-cyan-200"
              style={{
                height: `${Math.max((item.total / maxValue) * 100, 5)}%`,
              }}
            />
            <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-3 hidden w-32 -translate-x-1/2 rounded-2xl border border-white/10 bg-slate-900 px-3 py-2 text-center text-xs text-white shadow-xl group-hover:block">
              <b>{item.hour}:00</b>
              <br />
              {formatNumber(item.total)} 单
            </div>
          </div>
          {item.hour % 3 === 0 ? (
            <span className="text-center text-[10px] text-slate-500">
              {item.hour}
            </span>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function SimpleBarChart({
  data,
  valueSuffix = "单",
}: {
  data: SeriesPoint[];
  valueSuffix?: string;
}) {
  const maxValue = Math.max(...data.map((item) => item.value), 1);

  return (
    <div className="space-y-3">
      {data.map((item) => (
        <div key={item.label}>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-200">{item.label}</span>
            <span className="text-cyan-100">
              {formatNumber(Math.round(item.value))}
              {valueSuffix}
            </span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-sky-300 to-amber-200"
              style={{
                width: `${Math.max((item.value / maxValue) * 100, 4)}%`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function RankedDataList({
  data,
  activeLabel,
  onSelect,
}: {
  data: RankedPoint[];
  activeLabel?: string;
  onSelect?: (label: string) => void;
}) {
  return (
    <div className="space-y-3">
      {data.map((item, index) => {
        const isActive = activeLabel === item.label;
        const Container = onSelect ? "button" : "div";

        return (
          <Container
            key={item.label}
            type={onSelect ? "button" : undefined}
            onClick={() => onSelect?.(item.label)}
            className={`w-full rounded-3xl border p-4 text-left transition ${
              isActive
                ? "border-cyan-300/60 bg-cyan-300/15 shadow-lg shadow-cyan-500/10"
                : "border-white/10 bg-slate-950/60 hover:border-cyan-300/30 hover:bg-white/[0.08]"
            } ${onSelect ? "cursor-pointer" : ""}`}
          >
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <span
                  className={`flex h-8 w-8 items-center justify-center rounded-2xl text-xs font-black ${isActive ? "bg-cyan-200 text-slate-950" : "bg-cyan-300/15 text-cyan-100"}`}
                >
                  {index + 1}
                </span>
                <div>
                  <p className="text-sm font-bold text-white">{item.label}</p>
                  <p className="text-xs text-slate-400">{item.value}</p>
                </div>
              </div>
              <span className="text-sm font-black text-cyan-100">
                {item.score}%
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-amber-200"
                style={{ width: `${Math.max(item.score, 5)}%` }}
              />
            </div>
          </Container>
        );
      })}
    </div>
  );
}

function LineChartCard({
  title,
  subtitle,
  labels,
  primary,
  secondary,
  primaryLabel = "真实",
  secondaryLabel = "预测",
}: {
  title: string;
  subtitle: string;
  labels: string[];
  primary: number[];
  secondary?: number[];
  primaryLabel?: string;
  secondaryLabel?: string;
}) {
  const allValues = [...primary, ...(secondary ?? [])];
  const maxValue = Math.max(...allValues, 1);
  const minValue = Math.min(...allValues, 0);
  const span = Math.max(maxValue - minValue, 1);

  function toPoint(value: number, index: number) {
    const x = labels.length <= 1 ? 50 : (index / (labels.length - 1)) * 100;
    const y = 92 - ((value - minValue) / span) * 78;
    return `${x},${y}`;
  }

  const primaryPoints = primary.map(toPoint).join(" ");
  const secondaryPoints = secondary?.map(toPoint).join(" ");

  return (
    <GlassCard>
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm text-slate-300">{subtitle}</p>
          <h3 className="mt-1 text-2xl font-black text-white">{title}</h3>
        </div>
        {secondary ? (
          <div className="flex gap-2 text-xs">
            <span className="rounded-full bg-cyan-300/15 px-3 py-1 text-cyan-100">
              {primaryLabel}
            </span>
            <span className="rounded-full bg-amber-300/15 px-3 py-1 text-amber-100">
              {secondaryLabel}
            </span>
          </div>
        ) : null}
      </div>
      <div className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-4">
        <svg viewBox="0 0 100 100" className="h-64 w-full overflow-visible">
          {[20, 40, 60, 80].map((y) => (
            <line
              key={y}
              x1="0"
              x2="100"
              y1={y}
              y2={y}
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="0.5"
            />
          ))}
          {secondaryPoints ? (
            <polyline
              points={secondaryPoints}
              fill="none"
              stroke="#facc15"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ) : null}
          <polyline
            points={primaryPoints}
            fill="none"
            stroke="#22d3ee"
            strokeWidth="2.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {primary.map((value, index) => {
            const [x, y] = toPoint(value, index).split(",");
            return (
              <circle
                key={`${labels[index]}-${value}`}
                cx={x}
                cy={y}
                r="1.8"
                fill="#e0f2fe"
                stroke="#0891b2"
                strokeWidth="0.8"
              />
            );
          })}
        </svg>
        <div className="mt-2 flex justify-between text-[10px] text-slate-500">
          {labels.map((label) => (
            <span key={label}>{label}</span>
          ))}
        </div>
      </div>
    </GlassCard>
  );
}

function TreeModelRegionChart({ region }: { region: TreeModelRegion }) {
  const values = region.points.flatMap((item) => [item.actual, item.predicted]);
  const maxValue = Math.max(...values, 1);
  const minValue = Math.min(...values, 0);
  const span = Math.max(maxValue - minValue, 1);
  const labelIndexes = [
    0,
    Math.floor((region.points.length - 1) / 2),
    region.points.length - 1,
  ];

  function toPoint(value: number, index: number) {
    const x =
      region.points.length <= 1
        ? 50
        : (index / (region.points.length - 1)) * 100;
    const y = 92 - ((value - minValue) / span) * 78;
    return `${x},${y}`;
  }

  const actualPoints = region.points
    .map((item, index) => toPoint(item.actual, index))
    .join(" ");
  const predictedPoints = region.points
    .map((item, index) => toPoint(item.predicted, index))
    .join(" ");

  return (
    <div className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-4">
      <svg viewBox="0 0 100 100" className="h-72 w-full overflow-visible">
        {[20, 40, 60, 80].map((y) => (
          <line
            key={y}
            x1="0"
            x2="100"
            y1={y}
            y2={y}
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="0.5"
          />
        ))}
        <polyline
          points={predictedPoints}
          fill="none"
          stroke="#facc15"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <polyline
          points={actualPoints}
          fill="none"
          stroke="#22d3ee"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <div className="mt-2 grid grid-cols-3 text-[10px] text-slate-500">
        {labelIndexes.map((index, labelIndex) => (
          <span
            key={`${region.region}-${region.points[index]?.time}`}
            className={
              labelIndex === 1
                ? "text-center"
                : labelIndex === 2
                  ? "text-right"
                  : ""
            }
          >
            {region.points[index]?.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function TreeModelPanel({
  data,
  activeRegion,
  onChange,
}: {
  data: TreeModelData;
  activeRegion: string;
  onChange: (region: string) => void;
}) {
  const regions = Array.isArray(data.regions) ? data.regions : [];
  const metrics = Array.isArray(data.metrics) ? data.metrics : [];
  const featureRanking = Array.isArray(data.featureRanking)
    ? data.featureRanking
    : [];
  const selectedRegion =
    regions.find((region) => region.region === activeRegion) ?? regions[0];
  const rankedRegions = regions.map((region) => ({
    label: region.region,
    value: `MAE ${region.mae.toFixed(2)} / RMSE ${region.rmse.toFixed(2)}`,
    score: region.fitScore,
  }));

  if (!selectedRegion) return null;

  return (
    <GlassCard className="mb-5">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm text-slate-300">Top10 区域逐小时预测</p>
          <h3 className="mt-1 text-2xl font-black text-white">
            {data.label} 区域预测细节
          </h3>
          <p className="mt-2 text-sm text-slate-400">
            覆盖 {data.summary.regionCount} 个热点区域、
            {formatNumber(data.summary.sampleCount)} 条小时级记录，整体 MAE{" "}
            {data.summary.mae}，Top 特征为{" "}
            {getFeatureDisplayName(data.summary.topFeature)}。
          </p>
        </div>
        <Tag color={getModelTagColor(data.key)} className="rounded-full">
          R² {data.summary.r2}
        </Tag>
      </div>
      <div className="mb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <MetricCard
            key={`${data.key}-${metric.label}`}
            label={metric.label}
            value={metric.value}
            caption={metric.change}
          />
        ))}
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.25fr_0.75fr]">
        <div>
          {selectedRegion ? (
            <>
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm text-slate-300">区域拟合明细</p>
                  <h4 className="mt-1 text-xl font-black text-white">
                    {selectedRegion.region}
                  </h4>
                </div>
                <Tag color="cyan" className="rounded-full">
                  区域样本 {selectedRegion.sampleCount}
                </Tag>
              </div>
              <div className="mb-4 grid gap-3 sm:grid-cols-4">
                <MetricCard
                  label="样本数"
                  value={selectedRegion.sampleCount}
                  caption={selectedRegion.timeRange}
                />
                <MetricCard
                  label="真实总量"
                  value={selectedRegion.actualTotal}
                  caption="该区域逐小时真实订单"
                />
                <MetricCard
                  label="预测总量"
                  value={selectedRegion.predictedTotal}
                  caption="该区域逐小时预测订单"
                />
                <MetricCard
                  label="区域 MAE"
                  value={selectedRegion.mae.toFixed(2)}
                  caption={`RMSE ${selectedRegion.rmse.toFixed(2)}`}
                />
              </div>
              <div className="mb-3 flex gap-2 text-xs">
                <span className="rounded-full bg-cyan-300/15 px-3 py-1 text-cyan-100">
                  真实值
                </span>
                <span className="rounded-full bg-amber-300/15 px-3 py-1 text-amber-100">
                  预测值
                </span>
              </div>
              <TreeModelRegionChart region={selectedRegion} />
            </>
          ) : (
            <div className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-6 text-sm text-slate-400">
              暂无区域预测数据。
            </div>
          )}
        </div>
        <div className="grid gap-5">
          <div>
            <div className="mb-4 flex items-end justify-between gap-3">
              <div>
                <p className="text-sm text-slate-300">区域拟合表现</p>
                <h4 className="mt-1 text-xl font-black text-white">
                  Top10 误差排行
                </h4>
              </div>
              <span className="text-xs text-cyan-100/70">滚动查看更多</span>
            </div>
            <div className="max-h-[320px] overflow-y-auto pr-2 [scrollbar-gutter:stable]">
              <RankedDataList
                data={rankedRegions}
                activeLabel={selectedRegion.region}
                onSelect={onChange}
              />
            </div>
          </div>
          <div>
            <div className="mb-4">
              <p className="text-sm text-slate-300">特征排序</p>
              <h4 className="mt-1 text-xl font-black text-white">
                Top8 特征重要性
              </h4>
            </div>
            <RankedDataList
              data={featureRanking.map((item) => ({
                ...item,
                label: getFeatureDisplayName(item.label),
              }))}
            />
          </div>
        </div>
      </div>
    </GlassCard>
  );
}

function StidModelPanel({ data }: { data: StidModelData }) {
  if (!data.available) return null;

  return (
    <GlassCard className="mb-5">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm text-slate-300">新增时空图模型</p>
          <h3 className="mt-1 text-2xl font-black text-white">
            STID 训练预测表现
          </h3>
          <p className="mt-2 text-sm text-slate-400">
            覆盖 {formatNumber(data.summary.ordersCovered)} 单，平均真实需求{" "}
            {data.summary.meanTrue}，WAPE {data.summary.wape}%。
          </p>
        </div>
        <Tag color="purple" className="rounded-full">
          R²{" "}
          {data.metrics.find((item) => item.label === "STID R²")?.value ?? "--"}
        </Tag>
      </div>
      <div className="mb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {data.metrics.map((metric) => (
          <MetricCard
            key={metric.label}
            label={metric.label}
            value={metric.value}
            caption={metric.change}
          />
        ))}
      </div>
      <div className="mb-5 grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
        <LineChartCard
          title="STID 真实值与预测值"
          subtitle="测试集按时段聚合"
          labels={data.actualVsPredicted.labels}
          primary={data.actualVsPredicted.actual}
          secondary={data.actualVsPredicted.predicted}
        />
        <LineChartCard
          title="训练损失收敛"
          subtitle="训练集 / 验证集"
          labels={data.trainingLoss.labels}
          primary={data.trainingLoss.train}
          secondary={data.trainingLoss.validation}
          primaryLabel="训练"
          secondaryLabel="验证"
        />
      </div>
      <div className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
        <div>
          <div className="mb-4">
            <p className="text-sm text-slate-300">参数组合表现</p>
            <h4 className="mt-1 text-xl font-black text-white">配置得分排行</h4>
          </div>
          <RankedDataList data={data.sweepRanking} />
        </div>
        <div>
          <div className="mb-4 flex items-end justify-between gap-3">
            <div>
              <p className="text-sm text-slate-300">空间网格误差</p>
              <h4 className="mt-1 text-xl font-black text-white">高误差区域</h4>
            </div>
            <span className="text-xs text-cyan-100/70">
              {data.summary.bestName}
            </span>
          </div>
          <RankedDataList data={data.topRegionErrors} />
        </div>
      </div>
    </GlassCard>
  );
}

function CalendarHeatmap({ data }: { data: CalendarPoint[] }) {
  const maxValue = Math.max(...data.map((item) => item.value), 1);

  return (
    <GlassCard>
      <div className="mb-5">
        <p className="text-sm text-slate-300">按日需求节奏</p>
        <h3 className="mt-1 text-2xl font-black text-white">订单日历热力</h3>
      </div>
      <div className="grid grid-cols-7 gap-2">
        {data.map((item) => {
          const intensity = item.value / maxValue;
          return (
            <div
              key={item.date}
              className="group relative aspect-square rounded-2xl border border-white/10"
              style={{
                backgroundColor: `rgba(34, 211, 238, ${0.12 + intensity * 0.72})`,
              }}
            >
              <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-2 hidden w-28 -translate-x-1/2 rounded-2xl bg-slate-900 px-3 py-2 text-center text-xs text-white shadow-xl group-hover:block">
                <b>{item.date.slice(5)}</b>
                <br />
                {formatNumber(item.value)} 单
              </div>
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}

function HeatGrid({ data }: { data: HeatmapPoint[] }) {
  const xLabels = Array.from({ length: 24 }, (_, index) =>
    String(index).padStart(2, "0"),
  );
  const yLabels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];
  const valueMap = new Map(
    data.map((item) => [`${item.y}|${item.x}`, item.value]),
  );
  const maxValue = Math.max(...data.map((item) => item.value), 1);

  return (
    <GlassCard>
      <div className="mb-5">
        <p className="text-sm text-slate-300">小时 x 星期</p>
        <h3 className="mt-1 text-2xl font-black text-white">需求热力矩阵</h3>
      </div>
      <div className="space-y-2">
        {yLabels.map((weekday) => (
          <div
            key={weekday}
            className="grid grid-cols-[2.5rem_1fr] items-center gap-3"
          >
            <span className="text-xs text-slate-400">{weekday}</span>
            <div
              className="grid gap-1"
              style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}
            >
              {xLabels.map((hour) => {
                const value = valueMap.get(`${weekday}|${hour}`) ?? 0;
                const intensity = value / maxValue;
                return (
                  <div
                    key={`${weekday}-${hour}`}
                    title={`${weekday} ${hour}:00 ${formatNumber(value)} 单`}
                    className="h-5 rounded-md border border-white/5"
                    style={{
                      backgroundColor: `rgba(34, 211, 238, ${0.08 + intensity * 0.8})`,
                    }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </GlassCard>
  );
}

function ScatterPlot({
  data,
  xLabel,
  yLabel,
}: {
  data: ScatterPoint[];
  xLabel: string;
  yLabel: string;
}) {
  const maxX = Math.max(...data.map((item) => item.x), 1);
  const minX = Math.min(...data.map((item) => item.x), 0);
  const maxY = Math.max(...data.map((item) => item.y), 1);
  const minY = Math.min(...data.map((item) => item.y), 0);
  const xSpan = Math.max(maxX - minX, 1);
  const ySpan = Math.max(maxY - minY, 1);

  return (
    <div className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-4">
      <svg viewBox="0 0 100 100" className="h-64 w-full">
        {[20, 40, 60, 80].map((value) => (
          <g key={value}>
            <line
              x1="8"
              x2="96"
              y1={value}
              y2={value}
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="0.5"
            />
            <line
              x1={value}
              x2={value}
              y1="8"
              y2="92"
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="0.5"
            />
          </g>
        ))}
        {data.map((item, index) => {
          const x = 8 + ((item.x - minX) / xSpan) * 88;
          const y = 92 - ((item.y - minY) / ySpan) * 84;
          return (
            <circle
              key={`${item.x}-${item.y}-${index}`}
              cx={x}
              cy={y}
              r={item.size ?? 4}
              fill="rgba(34,211,238,0.72)"
              stroke="#fef3c7"
              strokeWidth="0.8"
            />
          );
        })}
      </svg>
      <div className="mt-2 flex justify-between text-xs text-slate-400">
        <span>{xLabel}</span>
        <span>{yLabel}</span>
      </div>
    </div>
  );
}

function MatrixMap({ data }: { data: number[][] }) {
  const maxValue = Math.max(...data.flat(), 1);

  return (
    <GlassCard>
      <div className="mb-5">
        <p className="text-sm text-slate-300">空间强度</p>
        <h3 className="mt-1 text-2xl font-black text-white">区域需求矩阵</h3>
      </div>
      <div className="grid grid-cols-6 gap-2">
        {data.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => {
            const intensity = value / maxValue;
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className="flex aspect-[1.25] items-end rounded-2xl border border-white/10 p-2"
                style={{
                  backgroundColor: `rgba(34, 211, 238, ${0.08 + intensity * 0.78})`,
                }}
              >
                <span className="text-[10px] font-bold text-white/80">
                  {formatNumber(value)}
                </span>
              </div>
            );
          }),
        )}
      </div>
    </GlassCard>
  );
}

function RoadDemandMap({
  points,
}: {
  points: Array<{
    geohash: string;
    pickups: number;
    latitude: number;
    longitude: number;
  }>;
}) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const hotspotLayerRef = useRef<LayerGroup | null>(null);
  const featuredPoints = useMemo(() => points.slice(0, 18), [points]);
  const maxPickups = Math.max(...points.map((item) => item.pickups), 1);

  useEffect(() => {
    let isDisposed = false;

    async function renderMap() {
      const leaflet = await import("leaflet");
      if (isDisposed || !mapContainerRef.current) return;

      if (!mapRef.current) {
        mapRef.current = leaflet
          .map(mapContainerRef.current, {
            center: [40.73061, -73.935242],
            zoom: 11,
            zoomControl: false,
            attributionControl: true,
            scrollWheelZoom: false,
          })
          .setView([40.73061, -73.935242], 11);
        leaflet.control.zoom({ position: "bottomright" }).addTo(mapRef.current);
        leaflet
          .tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution:
              '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19,
          })
          .addTo(mapRef.current);
        hotspotLayerRef.current = leaflet.layerGroup().addTo(mapRef.current);
      }

      const hotspotLayer = hotspotLayerRef.current;
      hotspotLayer?.clearLayers();
      if (!hotspotLayer) return;

      const bounds: Array<[number, number]> = [];
      for (const [index, item] of featuredPoints.entries()) {
        if (!Number.isFinite(item.latitude) || !Number.isFinite(item.longitude))
          continue;

        const intensity = item.pickups / maxPickups;
        const marker = leaflet
          .circleMarker([item.latitude, item.longitude], {
            radius: 6 + intensity * 16,
            color: index < 5 ? "#fef3c7" : "#cffafe",
            weight: 1.4,
            fillColor: index < 5 ? "#f59e0b" : "#06b6d4",
            fillOpacity: 0.72,
          })
          .bindTooltip(
            `<strong>${item.geohash}</strong><br/>${formatNumber(item.pickups)} 单<br/>${item.latitude.toFixed(4)}, ${item.longitude.toFixed(4)}`,
            { direction: "top", opacity: 0.92 },
          );

        marker.addTo(hotspotLayer);
        bounds.push([item.latitude, item.longitude]);
      }

      if (bounds.length > 1) {
        mapRef.current.fitBounds(bounds, { padding: [28, 28], maxZoom: 12 });
      }
    }

    renderMap();

    return () => {
      isDisposed = true;
    };
  }, [featuredPoints, maxPickups]);

  useEffect(
    () => () => {
      mapRef.current?.remove();
      mapRef.current = null;
    },
    [],
  );

  return (
    <GlassCard className="h-full">
      <div className="mb-5 flex items-center justify-between gap-4">
        <div>
          <p className="text-sm text-slate-300">NYC 道路需求地图</p>
          <h3 className="mt-1 text-2xl font-black text-white">
            道路底图与热点区域
          </h3>
        </div>
        <Tag color="cyan" className="rounded-full">
          Top {featuredPoints.length}
        </Tag>
      </div>
      <div className="relative overflow-hidden rounded-[2rem] border border-cyan-300/15 bg-slate-950/75 p-4">
        <div className="absolute left-4 top-4 z-10 flex flex-wrap gap-2 text-[10px]">
          <span className="rounded-full bg-cyan-300/15 px-3 py-1 text-cyan-100">
            OpenStreetMap 道路网络
          </span>
          <span className="rounded-full bg-amber-300/15 px-3 py-1 text-amber-100">
            橙色需求热点
          </span>
        </div>
        <div
          ref={mapContainerRef}
          className="taxi-road-map h-[420px] w-full rounded-[1.5rem]"
        />
        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          {featuredPoints.slice(0, 3).map((item, index) => (
            <div
              key={item.geohash}
              className="rounded-2xl border border-white/10 bg-white/[0.05] p-3"
            >
              <p className="text-xs text-slate-400">热点 #{index + 1}</p>
              <p className="mt-1 font-mono text-sm font-black text-white">
                {item.geohash}
              </p>
              <p className="text-xs text-cyan-100">
                {formatNumber(item.pickups)} 单
              </p>
            </div>
          ))}
        </div>
      </div>
    </GlassCard>
  );
}

function PredictionMapPanel({
  models,
  activeModelKey,
  onChangeModel,
  activeMode,
  onChangeMode,
  activeFrameIndex,
  onChangeFrameIndex,
}: {
  models: TreeModelData[];
  activeModelKey: string;
  onChangeModel: (modelKey: string) => void;
  activeMode: PredictionMapMode;
  onChangeMode: (mode: PredictionMapMode) => void;
  activeFrameIndex: number;
  onChangeFrameIndex: (index: number) => void;
}) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const mapLayerRef = useRef<LayerGroup | null>(null);
  const activeModel =
    models.find((item) => item.key === activeModelKey) ?? models[0];
  const frames = Array.isArray(activeModel?.mapFrames)
    ? activeModel.mapFrames
    : [];
  const safeFrameIndex = Math.min(
    Math.max(activeFrameIndex, 0),
    Math.max(frames.length - 1, 0),
  );
  const activeFrame = frames[safeFrameIndex];
  const featuredPoints = useMemo(
    () =>
      (Array.isArray(activeFrame?.points) ? activeFrame.points : []).slice(
        0,
        12,
      ),
    [activeFrame],
  );
  const maxMetricValue = useMemo(() => {
    if (!featuredPoints.length) return 1;

    return Math.max(
      ...featuredPoints.map((point) => {
        if (activeMode === "actual") return point.actual;
        if (activeMode === "error") return point.absError;
        return point.predicted;
      }),
      1,
    );
  }, [featuredPoints, activeMode]);

  useEffect(() => {
    let isDisposed = false;

    async function renderMap() {
      const leaflet = await import("leaflet");
      if (isDisposed || !mapContainerRef.current || !activeModel) return;

      if (!mapRef.current) {
        mapRef.current = leaflet
          .map(mapContainerRef.current, {
            center: [40.73061, -73.935242],
            zoom: 11,
            zoomControl: false,
            attributionControl: true,
            scrollWheelZoom: false,
          })
          .setView([40.73061, -73.935242], 11);
        leaflet.control.zoom({ position: "bottomright" }).addTo(mapRef.current);
        leaflet
          .tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution:
              '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19,
          })
          .addTo(mapRef.current);
        mapLayerRef.current = leaflet.layerGroup().addTo(mapRef.current);
      }

      const mapLayer = mapLayerRef.current;
      mapLayer?.clearLayers();
      if (!mapLayer) return;

      const bounds: Array<[number, number]> = [];
      for (const point of featuredPoints) {
        if (
          !Number.isFinite(point.latitude) ||
          !Number.isFinite(point.longitude)
        )
          continue;

        const metricValue =
          activeMode === "actual"
            ? point.actual
            : activeMode === "error"
              ? point.absError
              : point.predicted;
        const intensity = metricValue / maxMetricValue;
        const isPositiveError = point.error >= 0;
        const marker = leaflet
          .circleMarker([point.latitude, point.longitude], {
            radius: 6 + intensity * 16,
            color:
              activeMode === "error"
                ? isPositiveError
                  ? "#fecaca"
                  : "#bfdbfe"
                : "#cffafe",
            weight: 1.3,
            fillColor:
              activeMode === "error"
                ? isPositiveError
                  ? "#ef4444"
                  : "#3b82f6"
                : activeMode === "actual"
                  ? "#22d3ee"
                  : "#f59e0b",
            fillOpacity: 0.75,
          })
          .bindTooltip(
            `<strong>${point.geohash}</strong><br/>预测 ${point.predicted.toFixed(2)}<br/>真实 ${formatNumber(point.actual)}<br/>误差 ${
              point.error >= 0 ? "+" : ""
            }${point.error.toFixed(2)}`,
            { direction: "top", opacity: 0.92 },
          );

        marker.addTo(mapLayer);
        bounds.push([point.latitude, point.longitude]);
      }

      if (bounds.length > 1) {
        mapRef.current.fitBounds(bounds, { padding: [28, 28], maxZoom: 12 });
      }
    }

    renderMap();

    return () => {
      isDisposed = true;
    };
  }, [activeModel, activeMode, featuredPoints, maxMetricValue]);

  useEffect(
    () => () => {
      mapRef.current?.remove();
      mapRef.current = null;
    },
    [],
  );

  const modeDescription =
    activeMode === "error"
      ? "红色表示高估，蓝色表示低估"
      : activeMode === "actual"
        ? "圆点大小代表真实订单量"
        : "圆点大小代表预测订单量";

  return (
    <GlassCard className="mb-5">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-slate-300">Geo Prediction</p>
          <h3 className="mt-1 text-2xl font-black text-white">
            树模型预测地图
          </h3>
          <p className="mt-2 text-sm text-slate-400">
            {modeDescription}，当前时间切片为 {activeFrame?.label ?? "--"}。
          </p>
        </div>
        <Tag
          color={getModelTagColor(activeModel?.key ?? "")}
          className="rounded-full"
        >
          {activeModel?.label ?? "模型未就绪"}
        </Tag>
      </div>
      <div className="mb-5 grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-[2rem] border border-cyan-300/15 bg-slate-950/75 p-4">
          <div className="mb-4 flex flex-wrap gap-3">
            <Segmented
              size="middle"
              value={activeModel?.key}
              options={models.map((model) => ({
                label: model.label,
                value: model.key,
              }))}
              onChange={(value) => onChangeModel(String(value))}
            />
            <Segmented
              size="middle"
              value={activeMode}
              options={[
                { label: "预测值", value: "predicted" },
                { label: "真实值", value: "actual" },
                { label: "误差", value: "error" },
              ]}
              onChange={(value) => onChangeMode(value as PredictionMapMode)}
            />
          </div>
          <div
            ref={mapContainerRef}
            className="h-[420px] w-full rounded-[1.5rem]"
          />
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>{frames[0]?.label ?? "--"}</span>
              <span>{activeFrame?.label ?? "--"}</span>
              <span>{frames.at(-1)?.label ?? "--"}</span>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(frames.length - 1, 0)}
              value={safeFrameIndex}
              onChange={(event) =>
                onChangeFrameIndex(Number(event.target.value))
              }
              className="w-full accent-cyan-400"
              disabled={frames.length <= 1}
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
            <MetricCard
              label="时间切片"
              value={activeFrame?.label ?? "--"}
              caption="滑动查看逐小时分布"
            />
            <MetricCard
              label="展示点位"
              value={featuredPoints.length}
              caption="每帧最多保留 Top12 区域"
            />
            <MetricCard
              label={getPredictionModeLabel(activeMode)}
              value={maxMetricValue.toFixed(2)}
              caption={
                activeMode === "error" ? "当前帧绝对误差峰值" : "当前帧最大强度"
              }
            />
          </div>
          <div className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-4">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-300">地图点位排行</p>
                <h4 className="mt-1 text-xl font-black text-white">
                  {getPredictionModeLabel(activeMode)} Top{" "}
                  {featuredPoints.length}
                </h4>
              </div>
              <span className="text-xs text-cyan-100/70">
                {activeModel?.summary.topFeature
                  ? `Top 特征 ${getFeatureDisplayName(activeModel.summary.topFeature)}`
                  : ""}
              </span>
            </div>
            <div className="space-y-3">
              {featuredPoints.map((point, index) => {
                const metricValue =
                  activeMode === "actual"
                    ? point.actual
                    : activeMode === "error"
                      ? point.absError
                      : point.predicted;
                return (
                  <div
                    key={`${point.geohash}-${point.label}-${index}`}
                    className="rounded-2xl border border-white/10 bg-white/[0.04] p-3"
                  >
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <span className="font-mono text-sm font-black text-white">
                        {point.geohash}
                      </span>
                      <span className="rounded-full bg-cyan-300/15 px-3 py-1 text-xs font-bold text-cyan-100">
                        {metricValue.toFixed(2)}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400">
                      预测 {point.predicted.toFixed(2)} / 真实{" "}
                      {formatNumber(point.actual)} / 误差{" "}
                      {point.error >= 0 ? "+" : ""}
                      {point.error.toFixed(2)}
                    </p>
                  </div>
                );
              })}
              {!featuredPoints.length ? (
                <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 text-sm text-slate-400">
                  暂无地图切片数据。
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}

function PageNavigation({
  activePageIndex,
  onChange,
}: {
  activePageIndex: number;
  onChange: (nextIndex: number) => void;
}) {
  return (
    <div className="sticky top-0 z-20 mb-8 rounded-[2rem] border border-white/10 bg-slate-950/75 p-2 shadow-2xl shadow-slate-950/40 backdrop-blur-xl">
      <div className="grid gap-2 md:grid-cols-5">
        {pageItems.map((item, index) => {
          const isActive = activePageIndex === index + 1;

          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onChange(index + 1)}
              className={`group relative overflow-hidden rounded-[1.5rem] px-4 py-3 text-left transition duration-300 ${
                isActive
                  ? "bg-gradient-to-br from-cyan-300 via-sky-300 to-amber-200 text-slate-950 shadow-lg shadow-cyan-500/20"
                  : "bg-white/[0.06] text-slate-300 hover:bg-white/[0.12] hover:text-white"
              }`}
            >
              <span
                className={`mb-1 block text-[10px] font-black tracking-[0.28em] ${
                  isActive ? "text-slate-700" : "text-cyan-200/70"
                }`}
              >
                0{index + 1}
              </span>
              <span className="block text-sm font-bold">{item.label}</span>
              <span
                className={`absolute bottom-0 left-4 right-4 h-0.5 rounded-full transition ${
                  isActive
                    ? "bg-slate-950/50"
                    : "bg-cyan-300/0 group-hover:bg-cyan-300/50"
                }`}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}

function BaseDistributionPanel({
  data,
  maxBaseOrders,
}: {
  data: BaseDistribution[];
  maxBaseOrders: number;
}) {
  return (
    <GlassCard>
      <div className="mb-5 flex items-center justify-between gap-4">
        <div>
          <p className="text-sm text-slate-300">调度基地贡献</p>
          <h3 className="mt-1 text-2xl font-black text-white">订单占比分布</h3>
        </div>
        <Tag color="cyan" className="rounded-full">
          Top {data.length}
        </Tag>
      </div>
      <div className="space-y-4">
        {data.map((item, index) => {
          const width = Math.max((item.total / maxBaseOrders) * 100, 8);

          return (
            <div
              key={item.base}
              className="rounded-3xl border border-white/10 bg-slate-950/60 p-4"
            >
              <div className="mb-3 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-2xl bg-cyan-300/15 text-xs font-black text-cyan-100">
                    {index + 1}
                  </span>
                  <div>
                    <p className="font-mono text-base font-black text-white">
                      {item.base}
                    </p>
                    <p className="text-xs text-slate-400">
                      5 月 {formatNumber(item.may14)} / 6 月{" "}
                      {formatNumber(item.jun14)}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xl font-black text-white">{item.share}%</p>
                  <p className="text-xs text-slate-400">
                    {formatNumber(item.total)} 单
                  </p>
                </div>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-sky-300 to-amber-200"
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}

function FeatureInsightsPanel({ data }: { data: FeatureInsight[] }) {
  const maxScore = Math.max(...data.map((item) => item.importanceScore));
  const strongestFeature = data[0];

  return (
    <GlassCard>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-sm text-slate-300">关键变量强度</p>
          <h3 className="mt-1 text-2xl font-black text-white">特征影响雷达</h3>
        </div>
        <Tag color="gold" className="rounded-full">
          Top {data.length}
        </Tag>
      </div>
      <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-[2rem] border border-white/10 bg-slate-950/60 p-5">
          <div className="grid aspect-square place-items-center rounded-full border border-cyan-300/20 bg-[radial-gradient(circle,rgba(34,211,238,0.18),transparent_58%)]">
            <div className="text-center">
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-200">
                Strongest
              </p>
              <p className="mt-3 text-3xl font-black text-white">
                {strongestFeature
                  ? getFeatureDisplayName(strongestFeature.feature)
                  : "--"}
              </p>
              <p className="mt-2 font-mono text-xs text-cyan-100/80">
                {strongestFeature?.feature}
              </p>
              <p className="mt-2 text-sm text-slate-300">
                {strongestFeature?.importanceScore.toFixed(3)} 影响强度
              </p>
            </div>
          </div>
        </div>
        <div className="space-y-3">
          {data.map((item) => {
            const percent = Math.max(
              (item.importanceScore / maxScore) * 100,
              6,
            );

            return (
              <div
                key={item.feature}
                className="rounded-3xl border border-white/10 bg-white/[0.05] p-4"
              >
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-black text-cyan-100">
                      {getFeatureDisplayName(item.feature)}
                    </p>
                    <p className="text-xs text-slate-400">
                      <span className="font-mono">{item.feature}</span> ·{" "}
                      {item.featureType} · Pearson {item.pearsonCorr.toFixed(3)}{" "}
                      · Spearman {item.spearmanCorr.toFixed(3)}
                    </p>
                  </div>
                  <span className="rounded-full bg-cyan-300/15 px-3 py-1 text-xs font-bold text-cyan-100">
                    {item.importanceScore.toFixed(3)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-amber-200"
                    style={{ width: `${percent}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </GlassCard>
  );
}

function ImagePanel({
  title,
  src,
  tall = false,
  onPreview,
}: {
  title: string;
  src: string;
  tall?: boolean;
  onPreview?: (image: { title: string; src: string }) => void;
}) {
  const isPreviewable = Boolean(onPreview);

  return (
    <div className="overflow-hidden rounded-3xl border border-white/10 bg-slate-950/70">
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
        <span className="text-sm font-semibold text-white">{title}</span>
        <Tag color="cyan" className="m-0 rounded-full">
          {isPreviewable ? "点击放大" : "图表"}
        </Tag>
      </div>
      <button
        type="button"
        disabled={!isPreviewable}
        onClick={() => onPreview?.({ title, src })}
        className={`relative block w-full bg-white p-2 text-left ${tall ? "h-[380px]" : "h-[280px]"} ${
          isPreviewable
            ? "cursor-zoom-in transition duration-300 hover:bg-cyan-50"
            : "cursor-default"
        }`}
      >
        <Image
          src={src}
          alt={title}
          fill
          unoptimized
          sizes="(max-width: 768px) 100vw, 50vw"
          className="rounded-2xl object-contain p-2"
        />
      </button>
    </div>
  );
}

function ComplianceFooter() {
  return (
    <footer className="mt-12 border-t border-white/10 pb-8 pt-6 text-center text-xs leading-7 text-slate-400">
      <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
        <a
          href={icpRecordUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 transition hover:text-cyan-100"
          aria-label={`工信部备案 ${icpRecord}`}
        >
          <img
            src={icpRecordIconUrl}
            alt=""
            loading="lazy"
            className="h-4 w-4 object-contain"
            aria-hidden="true"
          />
          {icpRecord}
        </a>
        <a
          href={publicSecurityRecordUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 transition hover:text-cyan-100"
          aria-label={`公安联网备案 ${publicSecurityRecord}`}
        >
          <img
            src={publicSecurityRecordIconUrl}
            alt="公安联网备案图标"
            loading="lazy"
            className="h-4 w-4 object-contain"
          />
          {publicSecurityRecord}
        </a>
      </div>
    </footer>
  );
}

function MonthPanel({ month }: { month: MonthData }) {
  return (
    <div className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
      <GlassCard>
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="mb-1 text-sm text-slate-300">数据窗口</p>
            <h3 className="text-2xl font-black text-white">{month.label}</h3>
          </div>
          <Tag color="gold" className="rounded-full px-3 py-1">
            峰值 {getPeakLabel(month.peakHour.hour)}
          </Tag>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <MetricCard
            label="月订单量"
            value={month.totalPickups}
            caption="本月总需求规模"
          />
          <MetricCard
            label="日均订单"
            value={month.dailyAverage}
            caption="按自然日聚合后的均值"
          />
          <MetricCard
            label="空间网格"
            value={month.uniqueGeohashes}
            caption={`Geohash 精度 ${month.geohashPrecision}`}
          />
          <MetricCard
            label="半小时样本"
            value={month.aggregatedRows}
            caption={`${month.timeBinsPerDay} 个时间桶/天`}
          />
        </div>
        <div className="mt-5 rounded-3xl border border-white/10 bg-slate-950/70 p-4">
          <p className="mb-4 text-sm font-semibold text-white">
            热点区域 Top 8
          </p>
          <div className="space-y-3">
            {month.topRegions.map((region) => (
              <div key={region.geohash}>
                <div className="mb-1 flex justify-between text-xs text-slate-300">
                  <span>{region.geohash}</span>
                  <span>{formatNumber(region.pickups)} 单</span>
                </div>
                <Progress
                  percent={Math.round(
                    (region.pickups / month.topRegions[0].pickups) * 100,
                  )}
                  showInfo={false}
                  strokeColor={{ from: "#22d3ee", to: "#facc15" }}
                  railColor="rgba(255,255,255,0.08)"
                />
              </div>
            ))}
          </div>
        </div>
      </GlassCard>
      <div className="grid gap-5">
        <ImagePanel
          title={`${month.shortLabel} 上车密度热力图`}
          src={month.assets.density}
          tall
        />
        <ImagePanel
          title={`${month.shortLabel} 24 小时动态密度`}
          src={month.assets.densityGif}
        />
      </div>
    </div>
  );
}

export default function Home() {
  const [activePageIndex, setActivePageIndex] = useState(1);
  const [activeMonthKey, setActiveMonthKey] =
    useState<MonthData["key"]>("jun14");
  const [activeTreeRegions, setActiveTreeRegions] = useState<
    Record<string, string>
  >(
    Object.fromEntries(
      treeModels.map((model) => [model.key, model.regions[0]?.region ?? ""]),
    ),
  );
  const [activePredictionModelKey, setActivePredictionModelKey] = useState(
    treeModels[0]?.key ?? "xgboost",
  );
  const [activePredictionMode, setActivePredictionMode] =
    useState<PredictionMapMode>("predicted");
  const [activePredictionFrameIndex, setActivePredictionFrameIndex] =
    useState(0);
  const [previewImage, setPreviewImage] = useState<{
    title: string;
    src: string;
  } | null>(null);
  const activeMonth = useMemo(
    () =>
      dashboardData.months.find((month) => month.key === activeMonthKey) ??
      dashboardData.months[0],
    [activeMonthKey],
  );
  const activePredictionModel = useMemo(
    () =>
      treeModels.find((model) => model.key === activePredictionModelKey) ??
      treeModels[0],
    [activePredictionModelKey],
  );
  const activePage = pageItems[activePageIndex - 1].key;
  const maxBaseOrders = Math.max(
    ...dashboardData.baseDistribution.map((item) => item.total),
  );

  useEffect(() => {
    const frameCount = activePredictionModel?.mapFrames?.length ?? 0;
    setActivePredictionFrameIndex(frameCount > 0 ? frameCount - 1 : 0);
  }, [activePredictionModel]);

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: "#0891b2",
          borderRadius: 18,
          fontFamily: "var(--font-geist-sans), PingFang SC, sans-serif",
        },
        components: {
          Card: {
            colorBgContainer: "rgba(15, 23, 42, 0.72)",
            colorBorderSecondary: "rgba(255,255,255,0.1)",
          },
        },
      }}
    >
      <main className="min-h-screen overflow-hidden bg-[#061016] text-white">
        <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(34,211,238,0.28),transparent_28%),radial-gradient(circle_at_85%_0%,rgba(250,204,21,0.16),transparent_24%),linear-gradient(135deg,rgba(6,16,22,1),rgba(10,23,35,1)_48%,rgba(3,7,18,1))]" />
        <div className="pointer-events-none fixed inset-0 opacity-[0.08] [background-image:linear-gradient(rgba(255,255,255,0.7)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.7)_1px,transparent_1px)] [background-size:42px_42px]" />

        <div className="relative mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10">
          <header className="py-10">
            <Space wrap size="middle" className="mb-6">
              <Tag color="cyan" className="rounded-full px-4 py-1 text-sm">
                TaxiPredict
              </Tag>
              <Tag color="cyan" className="rounded-full px-4 py-1 text-sm">
                {dashboardData.summary.coverageRange}
              </Tag>
            </Space>
            <Typography.Title className="!mb-5 max-w-5xl !text-4xl !font-black !leading-[1.04] !tracking-[-0.05em] !text-white sm:!text-6xl">
              出租车需求预测与运营热力平台
            </Typography.Title>
            <Typography.Paragraph className="!mb-0 max-w-3xl !text-base !leading-8 !text-slate-300 sm:!text-lg">
              面向城市运力调度与需求研判，集中呈现订单趋势、空间热点、天气影响和模型预测效果。
            </Typography.Paragraph>
          </header>

          <PageNavigation
            activePageIndex={activePageIndex}
            onChange={setActivePageIndex}
          />

          {activePage === "overview" ? (
            <section className="py-6">
              <SectionTitle
                eyebrow="Overview"
                title="运营总览"
                description="快速查看平台核心指标、双月需求走势和关键峰值，帮助判断整体出行热度与运力压力。"
              />
              <div className="mb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="总订单量"
                  value={dashboardData.summary.totalPickups}
                  caption="双月累计需求规模"
                />
                <MetricCard
                  label="空间覆盖"
                  value={dashboardData.summary.totalRegions}
                  caption="服务区域覆盖强度"
                />
                <MetricCard
                  label="聚合样本"
                  value={dashboardData.summary.totalAggregatedRows}
                  caption="双月半小时网格聚合行数"
                />
                <MetricCard
                  label="环比增长"
                  value={`${dashboardData.summary.momGrowth}%`}
                  caption="6 月相对 5 月总订单量"
                />
              </div>
              <div className="grid gap-5 lg:grid-cols-[1fr_0.95fr]">
                <GlassCard>
                  <div className="mb-6 flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-300">24 小时订单负载</p>
                      <h2 className="mt-1 text-2xl font-black text-white">
                        双月时段对比
                      </h2>
                    </div>
                    <Tag color="gold" className="rounded-full">
                      高峰识别
                    </Tag>
                  </div>
                  <HourlyBars data={dashboardData.hourlyComparison} />
                </GlassCard>
                <GlassCard>
                  <h3 className="mb-5 text-lg font-bold text-white">
                    月度表现
                  </h3>
                  <div className="grid gap-4">
                    {dashboardData.months.map((month) => (
                      <div
                        key={month.key}
                        className="rounded-3xl border border-white/10 bg-white/5 p-5"
                      >
                        <div className="mb-4 flex items-center justify-between">
                          <div>
                            <p className="text-sm text-slate-300">
                              {month.label}
                            </p>
                            <p className="mt-1 text-3xl font-black text-white">
                              {formatNumber(month.totalPickups)}
                            </p>
                          </div>
                          <Tag color="cyan" className="rounded-full">
                            日均 {formatNumber(month.dailyAverage)}
                          </Tag>
                        </div>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <div className="rounded-2xl bg-slate-950/60 p-3">
                            <p className="text-xs text-slate-400">峰值时段</p>
                            <p className="mt-1 font-semibold text-cyan-100">
                              {getPeakLabel(month.peakHour.hour)}
                            </p>
                          </div>
                          <div className="rounded-2xl bg-slate-950/60 p-3">
                            <p className="text-xs text-slate-400">热点网格</p>
                            <p className="mt-1 font-semibold text-cyan-100">
                              {formatNumber(month.uniqueGeohashes)}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              </div>
              <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_0.95fr]">
                <LineChartCard
                  title="按日订单趋势"
                  subtitle="连续日期节奏"
                  labels={dashboardData.demandTrend.map((item) => item.label)}
                  primary={dashboardData.demandTrend.map((item) => item.value)}
                />
                <CalendarHeatmap data={dashboardData.dailyCalendar} />
              </div>
            </section>
          ) : null}

          {activePage === "operations" ? (
            <section className="py-6">
              <SectionTitle
                eyebrow="Operations"
                title="道路需求与运营结构"
                description="参考 NYC 道路底图需求地图的逻辑，把热点区域和小时负载放在同一页，帮助判断道路网络上的运力压力。"
              />
              <div className="grid gap-5 lg:grid-cols-[1fr_0.95fr]">
                <GlassCard>
                  <h3 className="mb-4 text-lg font-bold text-white">
                    小时级需求曲线
                  </h3>
                  <HourlyBars data={dashboardData.hourlyComparison} />
                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    {dashboardData.months.map((month) => (
                      <div
                        key={month.key}
                        className="rounded-3xl border border-white/10 bg-white/5 p-4"
                      >
                        <p className="text-sm text-slate-300">{month.label}</p>
                        <p className="mt-2 text-3xl font-black text-white">
                          {getPeakLabel(month.peakHour.hour)}
                        </p>
                        <p className="mt-1 text-sm text-cyan-200">
                          {formatNumber(month.peakHour.pickups)} 单/小时段
                        </p>
                      </div>
                    ))}
                  </div>
                </GlassCard>
                <RoadDemandMap points={dashboardData.geohashPoints} />
              </div>
              <div className="mt-5 grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
                <GlassCard>
                  <div className="mb-5">
                    <p className="text-sm text-slate-300">周内出行节奏</p>
                    <h3 className="mt-1 text-2xl font-black text-white">
                      星期需求画像
                    </h3>
                  </div>
                  <SimpleBarChart data={dashboardData.weekdayDemand} />
                </GlassCard>
                <HeatGrid data={dashboardData.weekdayHourHeatmap} />
              </div>
              <div className="mt-5">
                <BaseDistributionPanel
                  data={dashboardData.baseDistribution}
                  maxBaseOrders={maxBaseOrders}
                />
              </div>
            </section>
          ) : null}

          {activePage === "spatial" ? (
            <section className="py-6">
              <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
                <SectionTitle
                  eyebrow="Spatial Intelligence"
                  title="月度空间热力与聚合样本"
                  description="切换月份查看地理网格、热点区域、半小时聚合样本和动态密度变化。"
                />
                <Segmented
                  size="large"
                  value={activeMonthKey}
                  options={dashboardData.months.map((month) => ({
                    label: month.shortLabel,
                    value: month.key,
                  }))}
                  onChange={(value) =>
                    setActiveMonthKey(value as MonthData["key"])
                  }
                />
              </div>
              <div className="mb-5 grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
                <MatrixMap data={dashboardData.mapIntensity} />
                <GlassCard>
                  <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-sm text-slate-300">细粒度网格</p>
                      <h3 className="mt-1 text-2xl font-black text-white">
                        Geohash 热点分布
                      </h3>
                    </div>
                    <Tag color="cyan" className="rounded-full">
                      Top {dashboardData.geohashPoints.length}
                    </Tag>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    {dashboardData.geohashPoints
                      .slice(0, 10)
                      .map((item, index) => (
                        <div
                          key={item.geohash}
                          className="rounded-3xl border border-white/10 bg-slate-950/60 p-4"
                        >
                          <div className="mb-2 flex items-center justify-between">
                            <span className="font-mono text-sm font-black text-white">
                              {item.geohash}
                            </span>
                            <span className="rounded-full bg-cyan-300/15 px-3 py-1 text-xs font-bold text-cyan-100">
                              #{index + 1}
                            </span>
                          </div>
                          <p className="text-lg font-black text-white">
                            {formatNumber(item.pickups)} 单
                          </p>
                          <p className="mt-1 text-xs text-slate-400">
                            {item.latitude.toFixed(4)},{" "}
                            {item.longitude.toFixed(4)}
                          </p>
                        </div>
                      ))}
                  </div>
                </GlassCard>
              </div>
              <MonthPanel month={activeMonth} />
            </section>
          ) : null}

          {activePage === "features" ? (
            <section className="py-6">
              <SectionTitle
                eyebrow="Weather & Features"
                title="天气、时间特征与订单关系"
                description="观察天气、时间周期和订单需求之间的关联，辅助理解预测模型关注的关键变量。"
              />
              <div className="grid gap-5">
                <div className="grid gap-5 lg:grid-cols-[1fr_0.9fr]">
                  <GlassCard>
                    <div className="mb-5">
                      <p className="text-sm text-slate-300">温度与需求</p>
                      <h3 className="mt-1 text-2xl font-black text-white">
                        天气散点关系
                      </h3>
                    </div>
                    <ScatterPlot
                      data={dashboardData.weatherScatter}
                      xLabel="温度（°C）"
                      yLabel="订单量"
                    />
                  </GlassCard>
                  <GlassCard>
                    <div className="mb-5">
                      <p className="text-sm text-slate-300">天气类别</p>
                      <h3 className="mt-1 text-2xl font-black text-white">
                        平均订单影响
                      </h3>
                    </div>
                    <SimpleBarChart data={dashboardData.precipImpact} />
                  </GlassCard>
                </div>
                <GlassCard>
                  <div className="mb-5">
                    <p className="text-sm text-slate-300">温度分箱</p>
                    <h3 className="mt-1 text-2xl font-black text-white">
                      不同温度区间平均需求
                    </h3>
                  </div>
                  <SimpleBarChart data={dashboardData.weatherTempBins} />
                </GlassCard>
                <FeatureInsightsPanel data={dashboardData.featureInsights} />
                <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
                  <ImagePanel
                    title="小时订单分布"
                    src={dashboardData.assets.preprocess.hourlyOrders}
                    onPreview={setPreviewImage}
                  />
                  <ImagePanel
                    title="星期订单分布"
                    src={dashboardData.assets.preprocess.weekdayOrders}
                    onPreview={setPreviewImage}
                  />
                  <ImagePanel
                    title="温度与订单"
                    src={dashboardData.assets.preprocess.temperatureOrders}
                    onPreview={setPreviewImage}
                  />
                  <ImagePanel
                    title="降水与订单"
                    src={dashboardData.assets.preprocess.precipitationOrders}
                    onPreview={setPreviewImage}
                  />
                </div>
              </div>
            </section>
          ) : null}

          {activePage === "models" ? (
            <section className="py-6">
              <SectionTitle
                eyebrow="Model Outputs"
                title="预测模型效果看板"
                description="集中展示 XGBoost、LightGBM、CatBoost 与 STID 的结构化结果，并保留模型图片素材，方便横向比较树模型与时空模型表现。"
              />
              <PredictionMapPanel
                models={treeModels}
                activeModelKey={activePredictionModelKey}
                onChangeModel={setActivePredictionModelKey}
                activeMode={activePredictionMode}
                onChangeMode={setActivePredictionMode}
                activeFrameIndex={activePredictionFrameIndex}
                onChangeFrameIndex={setActivePredictionFrameIndex}
              />
              {treeModels.map((model) => (
                <TreeModelPanel
                  key={model.key}
                  data={model}
                  activeRegion={
                    activeTreeRegions[model.key] ??
                    model.regions[0]?.region ??
                    ""
                  }
                  onChange={(region) =>
                    setActiveTreeRegions((current) => ({
                      ...current,
                      [model.key]: region,
                    }))
                  }
                />
              ))}
              <StidModelPanel data={dashboardData.stidModel} />
              <div className="mb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {dashboardData.predictionMetrics.map((metric) => (
                  <MetricCard
                    key={metric.label}
                    label={metric.label}
                    value={metric.value}
                    caption={metric.change}
                  />
                ))}
              </div>
              <div className="mb-5 grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
                <LineChartCard
                  title="真实值与基线预测"
                  subtitle="最后一周按日汇总"
                  labels={dashboardData.actualVsPredicted.labels}
                  primary={dashboardData.actualVsPredicted.actual}
                  secondary={dashboardData.actualVsPredicted.predicted}
                />
                <GlassCard>
                  <div className="mb-5">
                    <p className="text-sm text-slate-300">真实-预测拟合</p>
                    <h3 className="mt-1 text-2xl font-black text-white">
                      拟合散点
                    </h3>
                  </div>
                  <ScatterPlot
                    data={dashboardData.actualPredictedScatter}
                    xLabel="真实订单量"
                    yLabel="预测订单量"
                  />
                </GlassCard>
              </div>
              <div className="mb-5 grid gap-5 lg:grid-cols-3">
                <GlassCard>
                  <div className="mb-5">
                    <p className="text-sm text-slate-300">误差分桶</p>
                    <h3 className="mt-1 text-2xl font-black text-white">
                      误差分布
                    </h3>
                  </div>
                  <SimpleBarChart
                    data={dashboardData.errorHistogram}
                    valueSuffix=" 个"
                  />
                </GlassCard>
                <LineChartCard
                  title="残差走势"
                  subtitle="预测值减真实值"
                  labels={dashboardData.residualTrend.map((item) => item.label)}
                  primary={dashboardData.residualTrend.map(
                    (item) => item.value,
                  )}
                />
                <GlassCard>
                  <div className="mb-5">
                    <p className="text-sm text-slate-300">误差日期</p>
                    <h3 className="mt-1 text-2xl font-black text-white">
                      偏差排行
                    </h3>
                  </div>
                  <RankedDataList data={dashboardData.predictionTopErrors} />
                </GlassCard>
              </div>
              <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                {modelAssets.map((asset) => (
                  <ImagePanel
                    key={asset.src}
                    title={asset.title}
                    src={asset.src}
                    onPreview={setPreviewImage}
                  />
                ))}
              </div>
            </section>
          ) : null}

          <ComplianceFooter />
        </div>
        <Modal
          centered
          footer={null}
          open={Boolean(previewImage)}
          title={previewImage?.title}
          width="min(1100px, 92vw)"
          onCancel={() => setPreviewImage(null)}
          styles={{
            body: { height: "72vh", padding: 0 },
          }}
        >
          {previewImage ? (
            <div className="relative h-full w-full bg-white">
              <Image
                src={previewImage.src}
                alt={previewImage.title}
                fill
                unoptimized
                sizes="92vw"
                className="object-contain p-4"
              />
            </div>
          ) : null}
        </Modal>
      </main>
    </ConfigProvider>
  );
}
