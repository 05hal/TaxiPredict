export type DashboardSchemaVersion = "2.0";

export type Tone = "sky" | "teal" | "violet" | "amber";

export type DimensionOption = {
  key: string;
  label: string;
};

export type MetricValueUnit = "count" | "percent" | "score" | "custom";

export type Metric = {
  id: string;
  label: string;
  value: number;
  unit?: MetricValueUnit;
  format?: "integer" | "decimal" | "percent" | "compact";
  suffix?: string;
  delta?: {
    value: number;
    unit?: "count" | "percent";
    label?: string;
  };
  tone?: Tone;
  description?: string;
};

export type SeriesPoint = {
  label: string;
  value: number;
};

export type Series1D = {
  points: SeriesPoint[];
  unit?: string;
};

export type MultiSeries = {
  labels: string[];
  series: Array<{
    key: string;
    label: string;
    values: number[];
  }>;
  unit?: string;
};

export type CalendarSeries = {
  values: Array<{
    date: string;
    value: number;
  }>;
  unit?: string;
};

export type HeatmapGrid = {
  xLabels: string[];
  yLabels: string[];
  cells: Array<{
    x: string;
    y: string;
    value: number;
  }>;
  unit?: string;
};

export type MatrixGrid = {
  rows: number;
  cols: number;
  cells: Array<{
    row: number;
    col: number;
    value: number;
  }>;
  unit?: string;
};

export type ScatterSeries = {
  points: Array<{
    x: number;
    y: number;
    size?: number;
    label?: string;
    meta?: Record<string, string | number>;
  }>;
  xAxisName?: string;
  yAxisName?: string;
};

export type RankedSeries = {
  items: Array<{
    key: string;
    label: string;
    value: number;
    score?: number;
    displayValue?: string;
    meta?: Record<string, string | number>;
  }>;
};

export type GeoPointSeries = {
  points: Array<{
    key: string;
    label: string;
    lat: number;
    lon: number;
    value: number;
    score?: number;
    group?: string;
    meta?: Record<string, string | number>;
  }>;
};

export type FeatureInsight = {
  feature: string;
  featureType?: string;
  importanceScore: number;
  pearsonCorr?: number;
  spearmanCorr?: number;
  entropy?: number;
  normalizedEntropy?: number;
  uniqueCount?: number;
  missingRate?: number;
};

export type AssetRef = {
  key: string;
  title: string;
  src: string;
  kind: "image" | "gif" | "geojson" | "json" | "other";
  tags?: string[];
};

export type PeriodSummary = {
  key: string;
  label: string;
  start: string;
  end: string;
  metrics: Metric[];
  peakHour?: {
    hour: number;
    value: number;
  };
};

export type TemporalBreakdown = {
  demandTrend?: Series1D;
  hourlyDemand?: Series1D;
  weekdayDemand?: Series1D;
};

export type SpatialBreakdown = {
  hotspots?: GeoPointSeries;
  geohashHotspots?: GeoPointSeries;
  topRegions?: RankedSeries;
};

export type WeatherBreakdown = {
  scatter?: ScatterSeries;
  precipImpact?: Series1D;
  tempBins?: Series1D;
};

export type ModelRegionDetail = {
  key: string;
  label: string;
  metrics: Metric[];
  series: MultiSeries;
};

export type ModelResult = {
  key: string;
  label: string;
  category: "baseline" | "tree" | "deep" | "hybrid" | "other";
  metrics: Metric[];
  charts?: {
    actualVsPredicted?: MultiSeries;
    fitScatter?: ScatterSeries;
    residualTrend?: Series1D;
    errorHistogram?: Series1D;
    trainingLoss?: MultiSeries;
  };
  rankings?: {
    topErrors?: RankedSeries;
    topRegions?: RankedSeries;
    sweepRanking?: RankedSeries;
  };
  regionDetail?: {
    summary: Metric[];
    items: ModelRegionDetail[];
  };
  assets?: AssetRef[];
};

// This is the canonical shape the current Next frontend should evolve toward.
// It preserves the current app's richer model/assets support and absorbs the
// legacy dashboard's reusable ideas: dimensions, breakdowns, and generic series.
export type DashboardSchema = {
  schemaVersion: DashboardSchemaVersion;
  generatedAt: string;
  meta: {
    title: string;
    description?: string;
    locale: string;
    timezone?: string;
    dateRange: {
      start: string;
      end: string;
    };
  };
  summary: {
    totalPickups: number;
    totalRegions: number;
    totalAggregatedRows: number;
    momGrowth: number;
    coverageRange: string;
  };
  dimensions: {
    periods: DimensionOption[];
    boroughs?: DimensionOption[];
    weekdays?: DimensionOption[];
    weatherCategories?: DimensionOption[];
  };
  periods: PeriodSummary[];
  temporal: {
    demandTrend: Series1D;
    dailyCalendar: CalendarSeries;
    hourlyDemand: MultiSeries;
    weekdayDemand: Series1D;
    weekdayHourHeatmap: HeatmapGrid;
    breakdowns?: {
      byBorough?: Record<string, TemporalBreakdown>;
      byWeekday?: Record<string, TemporalBreakdown>;
      byPeriod?: Record<string, TemporalBreakdown>;
    };
  };
  spatial: {
    geohashPoints: GeoPointSeries;
    mapIntensity: MatrixGrid;
    baseDistribution?: RankedSeries;
    boroughDemand?: RankedSeries;
    topRegions?: RankedSeries;
    breakdowns?: {
      byBorough?: Record<string, SpatialBreakdown>;
      byPeriod?: Record<string, SpatialBreakdown>;
    };
  };
  weather: {
    scatter: ScatterSeries;
    tempBins: Series1D;
    precipImpact: Series1D;
    breakdowns?: {
      byBorough?: Record<string, WeatherBreakdown>;
      byCategory?: Record<string, WeatherBreakdown>;
    };
  };
  features: {
    importance: RankedSeries;
    entropyScatter: ScatterSeries;
    insights?: FeatureInsight[];
  };
  models: {
    baseline?: ModelResult;
    items: ModelResult[];
  };
  assets: {
    preprocess?: AssetRef[];
    periods?: Record<string, AssetRef[]>;
    models?: AssetRef[];
  };
};
