"use client";

import { useMemo, useState } from "react";
import Image from "next/image";
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
  topRegions: Array<{ geohash: string; pickups: number }>;
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
  hourlyComparison: Array<{ hour: number; may14: number; jun14: number; total: number }>;
  baseDistribution: BaseDistribution[];
  featureInsights: FeatureInsight[];
  assets: {
    preprocess: Record<string, string>;
    models: Array<{ title: string; src: string }>;
  };
}

const dashboardData = dashboardJson as DashboardData;
const numberFormatter = new Intl.NumberFormat("zh-CN");
const pageItems = [
  { key: "overview", label: "运营总览" },
  { key: "operations", label: "运营结构" },
  { key: "spatial", label: "空间热力" },
  { key: "features", label: "天气特征" },
  { key: "models", label: "模型结果" },
] as const;

function formatNumber(value: number) {
  return numberFormatter.format(value);
}

function getPeakLabel(hour: number) {
  return `${String(hour).padStart(2, "0")}:00 - ${String((hour + 1) % 24).padStart(2, "0")}:00`;
}

function SectionTitle({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <div className="mb-5">
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.34em] text-cyan-300/80">{eyebrow}</p>
      <Typography.Title level={2} className="!mb-2 !text-2xl !text-white sm:!text-3xl">
        {title}
      </Typography.Title>
      <Typography.Paragraph className="!mb-0 max-w-3xl !text-sm !leading-7 !text-slate-300">
        {description}
      </Typography.Paragraph>
    </div>
  );
}

function GlassCard({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <Card className={`border-white/10 bg-white/[0.08] shadow-2xl shadow-cyan-950/20 backdrop-blur ${className}`}>
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
        {suffix ? <span className="ml-1 text-lg text-slate-300">{suffix}</span> : null}
      </div>
      <p className="mt-3 text-xs leading-6 text-slate-400">{caption}</p>
    </GlassCard>
  );
}

function HourlyBars({ data }: { data: Array<{ hour: number; may14: number; jun14: number; total: number }> }) {
  const maxValue = Math.max(...data.map((item) => item.total));

  return (
    <div className="flex h-64 items-end gap-1 rounded-3xl border border-white/10 bg-slate-950/70 p-4">
      {data.map((item) => (
        <div key={item.hour} className="group flex h-full flex-1 flex-col justify-end gap-1">
          <div className="relative flex flex-1 items-end rounded-full bg-white/5">
            <div
              className="w-full rounded-full bg-gradient-to-t from-cyan-500 via-sky-400 to-amber-200 transition-all duration-300 group-hover:from-amber-300 group-hover:to-cyan-200"
              style={{ height: `${Math.max((item.total / maxValue) * 100, 5)}%` }}
            />
            <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-3 hidden w-32 -translate-x-1/2 rounded-2xl border border-white/10 bg-slate-900 px-3 py-2 text-center text-xs text-white shadow-xl group-hover:block">
              <b>{item.hour}:00</b>
              <br />
              {formatNumber(item.total)} 单
            </div>
          </div>
          {item.hour % 3 === 0 ? <span className="text-center text-[10px] text-slate-500">{item.hour}</span> : null}
        </div>
      ))}
    </div>
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
                  isActive ? "bg-slate-950/50" : "bg-cyan-300/0 group-hover:bg-cyan-300/50"
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
            <div key={item.base} className="rounded-3xl border border-white/10 bg-slate-950/60 p-4">
              <div className="mb-3 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-2xl bg-cyan-300/15 text-xs font-black text-cyan-100">
                    {index + 1}
                  </span>
                  <div>
                    <p className="font-mono text-base font-black text-white">{item.base}</p>
                    <p className="text-xs text-slate-400">5 月 {formatNumber(item.may14)} / 6 月 {formatNumber(item.jun14)}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xl font-black text-white">{item.share}%</p>
                  <p className="text-xs text-slate-400">{formatNumber(item.total)} 单</p>
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
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-200">Strongest</p>
              <p className="mt-3 font-mono text-3xl font-black text-white">{data[0]?.feature}</p>
              <p className="mt-2 text-sm text-slate-300">{data[0]?.importanceScore.toFixed(3)} 影响强度</p>
            </div>
          </div>
        </div>
        <div className="space-y-3">
          {data.map((item) => {
            const percent = Math.max((item.importanceScore / maxScore) * 100, 6);

            return (
              <div key={item.feature} className="rounded-3xl border border-white/10 bg-white/[0.05] p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <p className="font-mono text-sm font-black text-cyan-100">{item.feature}</p>
                    <p className="text-xs text-slate-400">
                      {item.featureType} · Pearson {item.pearsonCorr.toFixed(3)} · Spearman {item.spearmanCorr.toFixed(3)}
                    </p>
                  </div>
                  <span className="rounded-full bg-cyan-300/15 px-3 py-1 text-xs font-bold text-cyan-100">
                    {item.importanceScore.toFixed(3)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white/10">
                  <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-amber-200" style={{ width: `${percent}%` }} />
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
          isPreviewable ? "cursor-zoom-in transition duration-300 hover:bg-cyan-50" : "cursor-default"
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
          <MetricCard label="月订单量" value={month.totalPickups} caption="本月总需求规模" />
          <MetricCard label="日均订单" value={month.dailyAverage} caption="按自然日聚合后的均值" />
          <MetricCard label="空间网格" value={month.uniqueGeohashes} caption={`Geohash 精度 ${month.geohashPrecision}`} />
          <MetricCard label="半小时样本" value={month.aggregatedRows} caption={`${month.timeBinsPerDay} 个时间桶/天`} />
        </div>
        <div className="mt-5 rounded-3xl border border-white/10 bg-slate-950/70 p-4">
          <p className="mb-4 text-sm font-semibold text-white">热点区域 Top 8</p>
          <div className="space-y-3">
            {month.topRegions.map((region) => (
              <div key={region.geohash}>
                <div className="mb-1 flex justify-between text-xs text-slate-300">
                  <span>{region.geohash}</span>
                  <span>{formatNumber(region.pickups)} 单</span>
                </div>
                <Progress
                  percent={Math.round((region.pickups / month.topRegions[0].pickups) * 100)}
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
        <ImagePanel title={`${month.shortLabel} 上车密度热力图`} src={month.assets.density} tall />
        <ImagePanel title={`${month.shortLabel} 24 小时动态密度`} src={month.assets.densityGif} />
      </div>
    </div>
  );
}

export default function Home() {
  const [activePageIndex, setActivePageIndex] = useState(1);
  const [activeMonthKey, setActiveMonthKey] = useState<MonthData["key"]>("jun14");
  const [previewImage, setPreviewImage] = useState<{ title: string; src: string } | null>(null);
  const activeMonth = useMemo(
    () => dashboardData.months.find((month) => month.key === activeMonthKey) ?? dashboardData.months[0],
    [activeMonthKey],
  );
  const activePage = pageItems[activePageIndex - 1].key;
  const maxBaseOrders = Math.max(...dashboardData.baseDistribution.map((item) => item.total));

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

          <PageNavigation activePageIndex={activePageIndex} onChange={setActivePageIndex} />

          {activePage === "overview" ? (
            <section className="py-6">
              <SectionTitle
                eyebrow="Overview"
                title="运营总览"
                description="快速查看平台核心指标、双月需求走势和关键峰值，帮助判断整体出行热度与运力压力。"
              />
              <div className="mb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard label="总订单量" value={dashboardData.summary.totalPickups} caption="双月累计需求规模" />
                <MetricCard label="空间覆盖" value={dashboardData.summary.totalRegions} caption="服务区域覆盖强度" />
                <MetricCard label="聚合样本" value={dashboardData.summary.totalAggregatedRows} caption="双月半小时网格聚合行数" />
                <MetricCard label="环比增长" value={`${dashboardData.summary.momGrowth}%`} caption="6 月相对 5 月总订单量" />
              </div>
              <div className="grid gap-5 lg:grid-cols-[1fr_0.95fr]">
                <GlassCard>
                  <div className="mb-6 flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-300">24 小时订单负载</p>
                      <h2 className="mt-1 text-2xl font-black text-white">双月时段对比</h2>
                    </div>
                    <Tag color="gold" className="rounded-full">
                      高峰识别
                    </Tag>
                  </div>
                  <HourlyBars data={dashboardData.hourlyComparison} />
                </GlassCard>
                <GlassCard>
                  <h3 className="mb-5 text-lg font-bold text-white">月度表现</h3>
                  <div className="grid gap-4">
                    {dashboardData.months.map((month) => (
                      <div key={month.key} className="rounded-3xl border border-white/10 bg-white/5 p-5">
                        <div className="mb-4 flex items-center justify-between">
                          <div>
                            <p className="text-sm text-slate-300">{month.label}</p>
                            <p className="mt-1 text-3xl font-black text-white">{formatNumber(month.totalPickups)}</p>
                          </div>
                          <Tag color="cyan" className="rounded-full">
                            日均 {formatNumber(month.dailyAverage)}
                          </Tag>
                        </div>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <div className="rounded-2xl bg-slate-950/60 p-3">
                            <p className="text-xs text-slate-400">峰值时段</p>
                            <p className="mt-1 font-semibold text-cyan-100">{getPeakLabel(month.peakHour.hour)}</p>
                          </div>
                          <div className="rounded-2xl bg-slate-950/60 p-3">
                            <p className="text-xs text-slate-400">热点网格</p>
                            <p className="mt-1 font-semibold text-cyan-100">{formatNumber(month.uniqueGeohashes)}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              </div>
            </section>
          ) : null}

          {activePage === "operations" ? (
            <section className="py-6">
              <SectionTitle
                eyebrow="Operations"
                title="订单结构与调度基地"
                description="对比小时需求、峰值时段和调度基地占比，用于判断运力投放的时段和基地优先级。"
              />
              <div className="grid gap-5 lg:grid-cols-[1fr_0.95fr]">
                <GlassCard>
                  <h3 className="mb-4 text-lg font-bold text-white">小时级需求曲线</h3>
                  <HourlyBars data={dashboardData.hourlyComparison} />
                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    {dashboardData.months.map((month) => (
                      <div key={month.key} className="rounded-3xl border border-white/10 bg-white/5 p-4">
                        <p className="text-sm text-slate-300">{month.label}</p>
                        <p className="mt-2 text-3xl font-black text-white">{getPeakLabel(month.peakHour.hour)}</p>
                        <p className="mt-1 text-sm text-cyan-200">{formatNumber(month.peakHour.pickups)} 单/小时段</p>
                      </div>
                    ))}
                  </div>
                </GlassCard>
                <BaseDistributionPanel data={dashboardData.baseDistribution} maxBaseOrders={maxBaseOrders} />
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
                  options={dashboardData.months.map((month) => ({ label: month.shortLabel, value: month.key }))}
                  onChange={(value) => setActiveMonthKey(value as MonthData["key"])}
                />
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
                <FeatureInsightsPanel data={dashboardData.featureInsights} />
                <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
                  <ImagePanel title="小时订单分布" src={dashboardData.assets.preprocess.hourlyOrders} onPreview={setPreviewImage} />
                  <ImagePanel title="星期订单分布" src={dashboardData.assets.preprocess.weekdayOrders} onPreview={setPreviewImage} />
                  <ImagePanel title="温度与订单" src={dashboardData.assets.preprocess.temperatureOrders} onPreview={setPreviewImage} />
                  <ImagePanel title="降水与订单" src={dashboardData.assets.preprocess.precipitationOrders} onPreview={setPreviewImage} />
                </div>
              </div>
            </section>
          ) : null}

          {activePage === "models" ? (
            <section className="py-6">
              <SectionTitle
                eyebrow="Model Outputs"
                title="预测模型效果看板"
                description="集中展示 LightGBM、CatBoost 与 GRU 的训练和预测效果，方便横向比较传统树模型和序列模型表现。"
              />
              <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                {dashboardData.assets.models.map((asset) => (
                  <ImagePanel key={asset.src} title={asset.title} src={asset.src} onPreview={setPreviewImage} />
                ))}
              </div>
            </section>
          ) : null}
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
