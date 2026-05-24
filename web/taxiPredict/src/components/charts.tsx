import { useEffect, useMemo, useState } from "react";
import type { EChartsOption, SeriesOption } from "echarts";
import * as echarts from "echarts/core";
import {
  BarChart as EchartsBarChart,
  HeatmapChart as EchartsHeatmapChart,
  LineChart as EchartsLineChart,
  ScatterChart as EchartsScatterChart,
} from "echarts/charts";
import {
  BrushComponent,
  CalendarComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  ToolboxComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import ReactEChartsCoreModule from "echarts-for-react/lib/core";
import {
  CircleMarker,
  GeoJSON,
  MapContainer,
  Pane,
  TileLayer,
  Tooltip as LeafletTooltip,
} from "react-leaflet";
import type { LatLngExpression } from "leaflet";
import type { FeatureCollection, GeoJsonObject, Geometry } from "geojson";
import type {
  CalendarPoint,
  HeatmapPoint,
  MapRegionDatum,
  RankedItem,
  ScatterPoint,
  SeriesPoint,
} from "../data/dashboard";
import "leaflet/dist/leaflet.css";

const ReactEChartsCore =
  (
    ReactEChartsCoreModule as unknown as {
      default?: typeof ReactEChartsCoreModule;
    }
  ).default ?? ReactEChartsCoreModule;

echarts.use([
  GridComponent,
  TooltipComponent,
  LegendComponent,
  ToolboxComponent,
  DataZoomComponent,
  BrushComponent,
  CalendarComponent,
  VisualMapComponent,
  EchartsLineChart,
  EchartsBarChart,
  EchartsScatterChart,
  EchartsHeatmapChart,
  CanvasRenderer,
]);

type Tone = "sky" | "teal" | "violet" | "amber";

type TonePalette = {
  main: string;
  soft: string;
  deep: string;
  dim: string;
};

type ChartEventHandler = (label: string) => void;

type AxisMeta = {
  xAxisName?: string;
  yAxisName?: string;
  xLabel?: string;
  yLabel?: string;
};

type BoroughFeatureProperties = {
  BoroName?: string;
};

type BoroughFeatureCollection = FeatureCollection<
  Geometry,
  BoroughFeatureProperties
>;

const paletteByTone: Record<Tone, TonePalette> = {
  sky: {
    main: "#4f8dff",
    soft: "rgba(79, 141, 255, 0.16)",
    deep: "#1e63d8",
    dim: "#bfd4ff",
  },
  teal: {
    main: "#1da596",
    soft: "rgba(29, 165, 150, 0.16)",
    deep: "#127d71",
    dim: "#a6e5de",
  },
  violet: {
    main: "#7865ff",
    soft: "rgba(120, 101, 255, 0.16)",
    deep: "#5843e0",
    dim: "#d2ccff",
  },
  amber: {
    main: "#e9a13c",
    soft: "rgba(233, 161, 60, 0.18)",
    deep: "#bf7722",
    dim: "#f4d9ac",
  },
};

const axisStyle = {
  axisLine: { lineStyle: { color: "rgba(20, 32, 51, 0.16)" } },
  axisTick: { show: false },
  axisLabel: { color: "#728094", fontSize: 11 },
  splitLine: { lineStyle: { color: "rgba(20, 32, 51, 0.08)" } },
};

const nycCenter: LatLngExpression = [40.73061, -73.935242];

function buildToolbox(enableZoom = false) {
  return {
    show: true,
    top: 0,
    right: 0,
    feature: {
      saveAsImage: { title: "导出图片", pixelRatio: 2 },
      restore: { title: "重置图表" },
      ...(enableZoom
        ? {
            dataZoom: {
              title: {
                zoom: "区域缩放",
                back: "缩放还原",
              },
            },
          }
        : {}),
    },
    iconStyle: {
      borderColor: "#7a8797",
    },
    emphasis: {
      iconStyle: {
        borderColor: "#142033",
      },
    },
  };
}

function buildTooltip() {
  return {
    trigger: "axis" as const,
    axisPointer: {
      type: "line" as const,
      lineStyle: { color: "rgba(20, 32, 51, 0.22)" },
      label: { backgroundColor: "#142033" },
    },
    backgroundColor: "rgba(11, 17, 28, 0.94)",
    borderWidth: 0,
    textStyle: {
      color: "#f4f7fb",
      fontSize: 12,
    },
  };
}

function buildBarColors(
  data: SeriesPoint[],
  palette: TonePalette,
  selectedLabel?: string,
) {
  return data.map((point) => {
    if (!selectedLabel) {
      return palette.main;
    }
    return point.label === selectedLabel ? palette.deep : palette.dim;
  });
}

function ChartFrame({
  option,
  height = 260,
  onClick,
}: {
  option: EChartsOption;
  height?: number;
  onClick?: ChartEventHandler;
}) {
  return (
    <div className="chart-box chart-box--echart">
      <ReactEChartsCore
        echarts={echarts}
        option={option}
        style={{ width: "100%", height: `${height}px` }}
        notMerge
        lazyUpdate
        onEvents={
          onClick
            ? {
                click: (params: { name?: string }) => {
                  if (params?.name) {
                    onClick(params.name);
                  }
                },
              }
            : undefined
        }
      />
    </div>
  );
}

export function LineChart({
  data,
  lines = [data.map((point) => point.value)],
  tone = "sky",
  xAxisName,
  yAxisName,
}: {
  data: SeriesPoint[];
  lines?: number[][];
  tone?: Tone;
} & AxisMeta) {
  const palette = paletteByTone[tone];

  const option = useMemo<EChartsOption>(() => {
    const labels = data.map((point) => point.label);
    const legendData = lines.length > 1 ? ["真实值", "预测值"] : ["订单量"];
    const series: SeriesOption[] = lines.map((line, index) => {
      const isPrimary = index === 0;
      const color = isPrimary ? palette.main : palette.deep;
      return {
        type: "line",
        name: legendData[index] ?? `序列 ${index + 1}`,
        data: line,
        smooth: true,
        symbol: "circle",
        symbolSize: 8,
        showSymbol: line.length <= 12,
        lineStyle: { width: isPrimary ? 3 : 2.5, color },
        itemStyle: { color, borderColor: "#ffffff", borderWidth: 2 },
        areaStyle:
          isPrimary && lines.length === 1 ? { color: palette.soft } : undefined,
        emphasis: { focus: "series" },
      };
    });

    return {
      animationDuration: 600,
      color: [palette.main, palette.deep],
      tooltip: buildTooltip(),
      legend:
        lines.length > 1
          ? {
              data: legendData,
              top: 2,
              left: 0,
              textStyle: { color: "#728094", fontSize: 12 },
            }
          : undefined,
      toolbox: buildToolbox(labels.length > 8),
      grid: {
        left: 24,
        right: 24,
        top: lines.length > 1 ? 56 : 28,
        bottom: labels.length > 8 ? 56 : 30,
      },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: labels,
        name: xAxisName,
        ...axisStyle,
      },
      yAxis: { type: "value", name: yAxisName, ...axisStyle },
      dataZoom:
        labels.length > 8
          ? [
              { type: "inside", start: 0, end: 100 },
              {
                type: "slider",
                height: 16,
                bottom: 8,
                borderColor: "transparent",
              },
            ]
          : undefined,
      series,
    };
  }, [data, lines, palette, xAxisName, yAxisName]);

  return <ChartFrame option={option} height={260} />;
}

export function BarChart({
  data,
  tone = "sky",
  selectedLabel,
  onSelect,
  xAxisName,
  yAxisName,
}: {
  data: SeriesPoint[];
  tone?: Tone;
  selectedLabel?: string;
  onSelect?: ChartEventHandler;
} & AxisMeta) {
  const palette = paletteByTone[tone];

  const option = useMemo<EChartsOption>(
    () => ({
      animationDuration: 500,
      tooltip: buildTooltip(),
      toolbox: buildToolbox(data.length > 8),
      grid: { left: 24, right: 24, top: 28, bottom: data.length > 8 ? 56 : 40 },
      xAxis: {
        type: "category",
        data: data.map((point) => point.label),
        name: xAxisName,
        ...axisStyle,
        axisLabel: {
          ...axisStyle.axisLabel,
          rotate: data.length > 8 ? 30 : 0,
        },
      },
      yAxis: { type: "value", name: yAxisName, ...axisStyle },
      dataZoom:
        data.length > 8
          ? [
              { type: "inside", start: 0, end: 100 },
              {
                type: "slider",
                height: 16,
                bottom: 8,
                borderColor: "transparent",
              },
            ]
          : undefined,
      series: [
        {
          type: "bar",
          data: data.map((point) => ({
            value: point.value,
            itemStyle: {
              color: point.label === selectedLabel ? palette.deep : undefined,
            },
          })),
          barMaxWidth: 34,
          itemStyle: {
            color: (params: { dataIndex: number }) =>
              buildBarColors(data, palette, selectedLabel)[params.dataIndex],
            borderRadius: [10, 10, 3, 3],
            shadowBlur: 12,
            shadowColor: palette.soft,
          },
          emphasis: { focus: "series", itemStyle: { color: palette.deep } },
        },
      ],
    }),
    [data, palette, selectedLabel, xAxisName, yAxisName],
  );

  return <ChartFrame option={option} height={250} onClick={onSelect} />;
}

export function RankedList({
  items,
  tone = "sky",
  selectedLabel,
  onSelect,
}: {
  items: RankedItem[];
  tone?: Tone;
  selectedLabel?: string;
  onSelect?: ChartEventHandler;
}) {
  const palette = paletteByTone[tone];

  const option = useMemo<EChartsOption>(
    () => ({
      animationDuration: 500,
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "rgba(11, 17, 28, 0.94)",
        borderWidth: 0,
        textStyle: { color: "#f4f7fb", fontSize: 12 },
        formatter: (params: unknown) => {
          const current = (Array.isArray(params) ? params[0] : params) as {
            name: string;
            value: number;
            dataIndex: number;
          };
          const item = items[current.dataIndex];
          return `${current.name}<br/>数值 ${item.value}<br/>强度 ${current.value}`;
        },
      },
      toolbox: buildToolbox(false),
      grid: { left: 88, right: 24, top: 24, bottom: 18 },
      xAxis: { type: "value", max: 100, ...axisStyle, splitNumber: 4 },
      yAxis: {
        type: "category",
        inverse: true,
        data: items.map((item) => item.label),
        ...axisStyle,
      },
      series: [
        {
          type: "bar",
          data: items.map((item) => item.score),
          barWidth: 14,
          showBackground: true,
          backgroundStyle: {
            color: "rgba(15, 23, 42, 0.06)",
            borderRadius: 999,
          },
          label: {
            show: true,
            position: "right",
            color: "#728094",
            formatter: (params: { dataIndex: number }) =>
              items[params.dataIndex]?.value ?? "",
          },
          itemStyle: {
            color: (params: { dataIndex: number }) => {
              const item = items[params.dataIndex];
              if (!selectedLabel) {
                return palette.main;
              }
              return item.label === selectedLabel ? palette.deep : palette.dim;
            },
            borderRadius: 999,
          },
          emphasis: { focus: "series", itemStyle: { color: palette.deep } },
        },
      ],
    }),
    [items, palette, selectedLabel],
  );

  return <ChartFrame option={option} height={250} onClick={onSelect} />;
}

export function ScatterChart({
  points,
  tone = "teal",
  xMaxOverride,
  yMaxOverride,
  xAxisName,
  yAxisName,
  xLabel = "X",
  yLabel = "Y",
}: {
  points: ScatterPoint[];
  tone?: Tone;
  xMaxOverride?: number;
  yMaxOverride?: number;
} & AxisMeta & {
    xMaxOverride?: number;
    yMaxOverride?: number;
  }) {
  const palette = paletteByTone[tone];
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  const xMin = Math.min(...xValues, 0);
  const xMax = xMaxOverride ?? Math.max(...xValues, 100);
  const yMin = Math.min(...yValues, 0);
  const yMax = yMaxOverride ?? Math.max(...yValues, 100);

  const option = useMemo<EChartsOption>(
    () => ({
      animationDuration: 500,
      tooltip: {
        trigger: "item",
        backgroundColor: "rgba(11, 17, 28, 0.94)",
        borderWidth: 0,
        textStyle: { color: "#f4f7fb", fontSize: 12 },
        formatter: (params: unknown) => {
          const [x, y, size] = (params as { value: [number, number, number] })
            .value;
          return `${xLabel}: ${x}<br/>${yLabel}: ${y}<br/>气泡大小: ${size}`;
        },
      },
      toolbox: buildToolbox(true),
      brush: {
        toolbox: ["rect", "polygon", "clear"],
        xAxisIndex: "all",
        yAxisIndex: "all",
      },
      grid: { left: 32, right: 24, top: 28, bottom: 56 },
      xAxis: {
        type: "value",
        min: xMin,
        max: xMax,
        name: xAxisName,
        ...axisStyle,
        nameGap: 16,
      },
      yAxis: {
        type: "value",
        min: yMin,
        max: yMax,
        name: yAxisName,
        ...axisStyle,
        scale: true,
      },
      dataZoom: [
        { type: "inside", xAxisIndex: 0, yAxisIndex: 0 },
        {
          type: "slider",
          xAxisIndex: 0,
          height: 16,
          bottom: 8,
          borderColor: "transparent",
        },
      ],
      series: [
        {
          type: "scatter",
          data: points.map((point) => [point.x, point.y, point.size ?? 5]),
          symbolSize: (value: number[]) => Math.max(10, (value[2] ?? 5) * 3),
          itemStyle: {
            color: palette.main,
            shadowBlur: 18,
            shadowColor: palette.soft,
            opacity: 0.84,
          },
          emphasis: {
            focus: "series",
            itemStyle: {
              color: palette.deep,
              borderColor: "#ffffff",
              borderWidth: 2,
            },
          },
        },
      ],
    }),
    [
      palette,
      points,
      xAxisName,
      xLabel,
      xMax,
      xMin,
      yAxisName,
      yLabel,
      yMax,
      yMin,
    ],
  );

  return <ChartFrame option={option} height={260} />;
}

export function CalendarHeatmapChart({
  data,
  tone = "sky",
}: {
  data: CalendarPoint[];
  tone?: Tone;
}) {
  const palette = paletteByTone[tone];
  const values = data.map((item) => item.value);
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 1);
  const option = useMemo<EChartsOption>(() => {
    const range = data.length
      ? [data[0].date, data[data.length - 1].date]
      : ["2014-05-01", "2014-06-30"];

    return {
      animationDuration: 500,
      tooltip: {
        trigger: "item",
        backgroundColor: "rgba(11, 17, 28, 0.94)",
        borderWidth: 0,
        textStyle: { color: "#f4f7fb", fontSize: 12 },
        formatter: (params: unknown) => {
          const value = (params as { value: [string, number] }).value;
          return `${value[0]}<br/>订单量 ${value[1].toLocaleString("en-US")}`;
        },
      },
      visualMap: {
        min: minValue,
        max: maxValue,
        calculable: true,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        textStyle: { color: "#728094" },
        inRange: {
          color: ["#eef5ff", palette.dim, palette.main, palette.deep],
        },
      },
      calendar: {
        top: 18,
        left: 24,
        right: 24,
        cellSize: ["auto", 18],
        range,
        itemStyle: {
          borderWidth: 2,
          borderColor: "#f8fbff",
          borderRadius: 6,
        },
        splitLine: { lineStyle: { color: "rgba(20, 32, 51, 0.05)" } },
        yearLabel: { show: false },
        dayLabel: { color: "#728094", firstDay: 1 },
        monthLabel: { color: "#142033" },
      },
      series: [
        {
          type: "heatmap",
          coordinateSystem: "calendar",
          data: data.map((item) => [item.date, item.value]),
        },
      ],
    };
  }, [data, maxValue, minValue, palette.deep, palette.dim, palette.main]);

  return <ChartFrame option={option} height={240} />;
}

export function HeatGridChart({
  data,
  xLabels,
  yLabels,
  tone = "sky",
}: {
  data: HeatmapPoint[];
  xLabels: string[];
  yLabels: string[];
  tone?: Tone;
}) {
  const palette = paletteByTone[tone];
  const values = data.map((item) => item.value);
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 1);

  const option = useMemo<EChartsOption>(
    () => ({
      animationDuration: 500,
      tooltip: {
        trigger: "item",
        backgroundColor: "rgba(11, 17, 28, 0.94)",
        borderWidth: 0,
        textStyle: { color: "#f4f7fb", fontSize: 12 },
        formatter: (params: unknown) => {
          const value = (params as { value: [number, number, number] }).value;
          return `${yLabels[value[1]]} ${xLabels[value[0]]}:00<br/>订单量 ${value[2].toLocaleString("en-US")}`;
        },
      },
      grid: { left: 48, right: 24, top: 24, bottom: 42 },
      xAxis: {
        type: "category",
        data: xLabels,
        name: "小时",
        ...axisStyle,
      },
      yAxis: {
        type: "category",
        data: yLabels,
        name: "星期",
        ...axisStyle,
      },
      visualMap: {
        min: minValue,
        max: maxValue,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        textStyle: { color: "#728094" },
        inRange: {
          color: ["#eef5ff", palette.dim, palette.main, palette.deep],
        },
      },
      series: [
        {
          type: "heatmap",
          data: data.map((item) => [
            xLabels.indexOf(item.x),
            yLabels.indexOf(item.y),
            item.value,
          ]),
          label: {
            show: false,
          },
          emphasis: {
            itemStyle: {
              borderColor: "#ffffff",
              borderWidth: 1,
            },
          },
        },
      ],
    }),
    [
      data,
      maxValue,
      minValue,
      palette.deep,
      palette.dim,
      palette.main,
      xLabels,
      yLabels,
    ],
  );

  return <ChartFrame option={option} height={280} />;
}

export function GeoPointMapChart({
  points,
  selectedBorough,
}: {
  points: RankedItem[];
  selectedBorough?: string;
}) {
  const maxCount = Math.max(...points.map((item) => item.count ?? 0), 1);

  return (
    <div className="map-shell">
      <div className="map-shell__legend">
        <span className="map-shell__legend-chip">
          蓝色底图：OpenStreetMap 道路网络
        </span>
        <span className="map-shell__legend-chip map-shell__legend-chip--hotspot">
          橙色点：geohash 聚合热点
        </span>
      </div>
      <div className="map-shell__canvas">
        <MapContainer
          center={nycCenter}
          zoom={11}
          scrollWheelZoom
          className="leaflet-map"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <Pane name="geohash-hotspots" style={{ zIndex: 650 }} />
          {points.map((item) => {
            const ratio = (item.count ?? 0) / maxCount;
            return (
              <CircleMarker
                key={`${item.label}-${item.lat}-${item.lon}`}
                center={[item.lat ?? 0, item.lon ?? 0]}
                radius={5 + ratio * 10}
                pathOptions={{
                  pane: "geohash-hotspots",
                  color: "rgba(255,255,255,0.92)",
                  weight: 1.2,
                  fillColor:
                    selectedBorough && item.borough === selectedBorough
                      ? "#4f8dff"
                      : "#f78c2b",
                  fillOpacity: 0.7,
                }}
              >
                <LeafletTooltip direction="top" offset={[0, -8]}>
                  <div className="map-tooltip">
                    <strong>{item.label}</strong>
                    <span>geohash 聚合单元</span>
                    <span>
                      订单量 {(item.count ?? 0).toLocaleString("en-US")}
                    </span>
                    <span>所属区域 {item.borough ?? "Unknown"}</span>
                  </div>
                </LeafletTooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}

export function MapChart({
  data,
  hotspots,
  selectedBorough,
  onSelectBorough,
}: {
  data: MapRegionDatum[];
  hotspots: RankedItem[];
  selectedBorough?: string;
  onSelectBorough?: ChartEventHandler;
}) {
  const [boroughGeoJson, setBoroughGeoJson] =
    useState<BoroughFeatureCollection | null>(null);
  const boroughData = useMemo(
    () => new Map(data.map((item) => [item.name, item.value])),
    [data],
  );
  const maxDemand = Math.max(...data.map((item) => item.value), 1);
  const maxHotspotCount = Math.max(
    ...hotspots.map((item) => item.count ?? 0),
    1,
  );

  useEffect(() => {
    let active = true;

    fetch("/data/nyc-boroughs.geojson")
      .then((response) => response.json())
      .then((geoJson: GeoJsonObject) => {
        if (active) {
          setBoroughGeoJson(geoJson as BoroughFeatureCollection);
        }
      })
      .catch(() => {
        if (active) {
          setBoroughGeoJson(null);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const styleByBorough = (boroughName: string) => {
    const demand = boroughData.get(boroughName) ?? 0;
    const intensity = demand / maxDemand;
    const isSelected = selectedBorough === boroughName;
    return {
      fillColor: isSelected
        ? "#4f8dff"
        : `rgba(79, 141, 255, ${0.18 + intensity * 0.52})`,
      fillOpacity: 0.72,
      color: isSelected ? "#163e9d" : "#ffffff",
      weight: isSelected ? 3 : 2,
      opacity: 0.95,
      dashArray: isSelected ? undefined : "4 3",
    };
  };

  return (
    <div className="map-shell">
      <div className="map-shell__legend">
        <span className="map-shell__legend-chip">
          蓝色区块：borough 总订单量
        </span>
        <span className="map-shell__legend-chip map-shell__legend-chip--hotspot">
          橙色亮点：热点区域中心点
        </span>
      </div>
      <div className="map-shell__canvas">
        {!boroughGeoJson ? (
          <div className="chart-loading">正在加载道路底图与区域边界...</div>
        ) : (
          <MapContainer
            center={nycCenter}
            zoom={11}
            scrollWheelZoom
            className="leaflet-map"
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <GeoJSON
              key={selectedBorough ?? "all"}
              data={boroughGeoJson}
              style={(feature) =>
                styleByBorough(feature?.properties?.BoroName ?? "")
              }
              onEachFeature={(feature, layer) => {
                const boroughName = feature.properties?.BoroName ?? "Unknown";
                const demand = boroughData.get(boroughName) ?? 0;
                layer.bindTooltip(
                  `${boroughName}<br/>订单量 ${demand.toLocaleString("en-US")}`,
                );
                layer.on({
                  click: () => onSelectBorough?.(boroughName),
                });
              }}
            />
            <Pane name="hotspots" style={{ zIndex: 650 }} />
            {hotspots.map((item) => {
              const ratio = (item.count ?? 0) / maxHotspotCount;
              const radius = 8 + ratio * 18;
              return (
                <CircleMarker
                  key={`${item.label}-${item.lat}-${item.lon}`}
                  center={[item.lat ?? 0, item.lon ?? 0]}
                  radius={radius}
                  pathOptions={{
                    pane: "hotspots",
                    color: "rgba(255,255,255,0.86)",
                    weight: 1.5,
                    fillColor: "#f78c2b",
                    fillOpacity: 0.66,
                  }}
                >
                  <LeafletTooltip direction="top" offset={[0, -8]}>
                    <div className="map-tooltip">
                      <strong>{item.label}</strong>
                      <span>热点中心点</span>
                      <span>
                        订单量 {(item.count ?? 0).toLocaleString("en-US")}
                      </span>
                      <span>所属区域 {item.borough ?? "Unknown"}</span>
                    </div>
                  </LeafletTooltip>
                </CircleMarker>
              );
            })}
          </MapContainer>
        )}
      </div>
    </div>
  );
}
