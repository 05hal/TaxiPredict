export type NavItem = {
  id: string;
  label: string;
  description: string;
  path: string;
};

export type FilterGroup = {
  label: string;
  value: string;
};

export type Metric = {
  label: string;
  value: string;
  change: string;
  tone?: "sky" | "teal" | "violet" | "amber";
};

export type SeriesPoint = {
  label: string;
  value: number;
};

export type RankedItem = {
  label: string;
  value: string;
  score: number;
  lat?: number;
  lon?: number;
  borough?: string;
  count?: number;
};

export type ScatterPoint = {
  x: number;
  y: number;
  size?: number;
};

export type CalendarPoint = {
  date: string;
  value: number;
};

export type HeatmapPoint = {
  x: string;
  y: string;
  value: number;
};

export type MapRegionDatum = {
  name: string;
  value: number;
};

export type DashboardData = {
  generatedAt: string;
  overviewMetrics: Metric[];
  demandTrend: SeriesPoint[];
  demandTrendByBorough: Record<string, SeriesPoint[]>;
  dailyCalendar: CalendarPoint[];
  hourlyDemand: SeriesPoint[];
  hourlyDemandByBorough: Record<string, SeriesPoint[]>;
  weekdayDemand: SeriesPoint[];
  weekdayDemandByBorough: Record<string, SeriesPoint[]>;
  weekdayProfiles: Record<string, SeriesPoint[]>;
  weekdayHourHeatmap: HeatmapPoint[];
  topRegions: RankedItem[];
  topRegionsByBorough: Record<string, RankedItem[]>;
  regionPoints: RankedItem[];
  geohashPoints: RankedItem[];
  boroughDemand: MapRegionDatum[];
  mapIntensity: number[][];
  weatherScatter: ScatterPoint[];
  weatherScatterByBorough: Record<string, ScatterPoint[]>;
  weatherScatterByCategory: Record<string, ScatterPoint[]>;
  precipImpact: SeriesPoint[];
  precipImpactByBorough: Record<string, SeriesPoint[]>;
  weatherTempBins: SeriesPoint[];
  featureImportance: RankedItem[];
  entropyScatter: ScatterPoint[];
  predictionMetrics: Metric[];
  actualVsPredicted: {
    actual: number[];
    predicted: number[];
    labels: string[];
  };
  actualPredictedScatter: ScatterPoint[];
  errorHistogram: SeriesPoint[];
  residualTrend: SeriesPoint[];
  predictionTopErrors: RankedItem[];
};

export const navItems: NavItem[] = [
  {
    id: "overview",
    label: "项目总览",
    description: "订单规模与实验范围",
    path: "/",
  },
  {
    id: "spatial",
    label: "时空分布",
    description: "地图热点与分时模式",
    path: "/spatial",
  },
  {
    id: "weather",
    label: "天气影响",
    description: "天气变量与需求关系",
    path: "/weather",
  },
  {
    id: "features",
    label: "特征价值",
    description: "相关性与熵值分析",
    path: "/features",
  },
  {
    id: "prediction",
    label: "预测评估",
    description: "效果、误差与解释",
    path: "/prediction",
  },
];

export const filterGroups: FilterGroup[] = [
  { label: "时间范围", value: "2014-05 至 2014-06" },
  { label: "空间粒度", value: "原始坐标 / geohash" },
  { label: "天气条件", value: "晴天、雨天、雪天、雾天" },
  { label: "样本切片", value: "工作日 / 周末 / 高峰期" },
];

export async function fetchDashboardData(): Promise<DashboardData> {
  const response = await fetch("/data/dashboard.json");

  if (!response.ok) {
    throw new Error(`读取 dashboard.json 失败: ${response.status}`);
  }

  return response.json() as Promise<DashboardData>;
}
