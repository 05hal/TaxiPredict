import { useEffect, useMemo, useState } from "react";
import { FilterBar } from "./FilterBar";
import { MetricGrid } from "./MetricGrid";
import { SectionCard } from "./SectionCard";
import { SidebarNav } from "./SidebarNav";
import {
  BarChart,
  LineChart,
  MapChart,
  RankedList,
  ScatterChart,
} from "./charts";
import {
  type DashboardData,
  fetchDashboardData,
  filterGroups,
  navItems,
} from "../data/dashboard";

type ActiveFilter =
  | { type: "borough"; value: string }
  | { type: "weekday"; value: string }
  | { type: "weather"; value: string }
  | null;

function formatFilterLabel(activeFilter: ActiveFilter) {
  if (!activeFilter) {
    return "当前展示全量数据。点击地图、星期柱或天气柱可联动筛选。";
  }

  if (activeFilter.type === "borough") {
    return `当前筛选：${activeFilter.value}，时空、天气和热点区域已联动更新。`;
  }

  if (activeFilter.type === "weekday") {
    return `当前筛选：${activeFilter.value}，小时分布已切换到该星期画像。`;
  }

  return `当前筛选：${activeFilter.value}，天气散点已切换到该天气类别。`;
}

export function DashboardShell() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(
    null,
  );
  const [errorMessage, setErrorMessage] = useState("");
  const [activeFilter, setActiveFilter] = useState<ActiveFilter>(null);

  useEffect(() => {
    let active = true;

    fetchDashboardData()
      .then((data) => {
        if (active) {
          setDashboardData(data);
        }
      })
      .catch((error: Error) => {
        if (active) {
          setErrorMessage(error.message);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const activeFilterText = useMemo(
    () => formatFilterLabel(activeFilter),
    [activeFilter],
  );

  if (errorMessage) {
    return (
      <div className="app-shell">
        <SidebarNav items={navItems} />
        <main className="dashboard-main">
          <section className="state-panel">
            <p className="state-panel__eyebrow">数据加载失败</p>
            <h2>无法读取真实面板数据</h2>
            <p>{errorMessage}</p>
            <p>
              请先执行 `npm run prepare:data` 生成
              `public/data/dashboard.json`。
            </p>
          </section>
        </main>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="app-shell">
        <SidebarNav items={navItems} />
        <main className="dashboard-main">
          <section className="state-panel">
            <p className="state-panel__eyebrow">正在加载</p>
            <h2>正在读取真实 CSV 聚合结果</h2>
            <p>
              页面会直接消费由真实 Uber
              订单、天气数据和特征分析报告生成的统计结果。
            </p>
          </section>
        </main>
      </div>
    );
  }

  const selectedBorough =
    activeFilter?.type === "borough" ? activeFilter.value : null;
  const selectedWeekday =
    activeFilter?.type === "weekday" ? activeFilter.value : null;
  const selectedWeather =
    activeFilter?.type === "weather" ? activeFilter.value : null;

  const displayedDemandTrend = selectedBorough
    ? (dashboardData.demandTrendByBorough[selectedBorough] ??
      dashboardData.demandTrend)
    : dashboardData.demandTrend;
  const displayedHourlyDemand = selectedBorough
    ? (dashboardData.hourlyDemandByBorough[selectedBorough] ??
      dashboardData.hourlyDemand)
    : selectedWeekday
      ? (dashboardData.weekdayProfiles[selectedWeekday] ??
        dashboardData.hourlyDemand)
      : dashboardData.hourlyDemand;
  const displayedWeekdayDemand = selectedBorough
    ? (dashboardData.weekdayDemandByBorough[selectedBorough] ??
      dashboardData.weekdayDemand)
    : dashboardData.weekdayDemand;
  const displayedTopRegions = selectedBorough
    ? (dashboardData.topRegionsByBorough[selectedBorough] ??
      dashboardData.topRegions)
    : dashboardData.topRegions;
  const displayedWeatherScatter = selectedBorough
    ? (dashboardData.weatherScatterByBorough[selectedBorough] ??
      dashboardData.weatherScatter)
    : selectedWeather
      ? (dashboardData.weatherScatterByCategory[selectedWeather] ??
        dashboardData.weatherScatter)
      : dashboardData.weatherScatter;
  const displayedPrecipImpact = selectedBorough
    ? (dashboardData.precipImpactByBorough[selectedBorough] ??
      dashboardData.precipImpact)
    : dashboardData.precipImpact;
  const mapHotspots = selectedBorough
    ? (dashboardData.topRegionsByBorough[selectedBorough] ??
      dashboardData.regionPoints)
    : dashboardData.regionPoints;

  const actualVsPredictedSeries = dashboardData.actualVsPredicted.labels.map(
    (label, index) => ({
      label,
      value: dashboardData.actualVsPredicted.actual[index] ?? 0,
    }),
  );

  function toggleBoroughFilter(label: string) {
    setActiveFilter((current) => {
      if (current?.type === "borough" && current.value === label) {
        return null;
      }
      return { type: "borough", value: label };
    });
  }

  function toggleWeekdayFilter(label: string) {
    setActiveFilter((current) => {
      if (current?.type === "weekday" && current.value === label) {
        return null;
      }
      return { type: "weekday", value: label };
    });
  }

  function toggleWeatherFilter(label: string) {
    setActiveFilter((current) => {
      if (current?.type === "weather" && current.value === label) {
        return null;
      }
      return { type: "weather", value: label };
    });
  }

  function handleTopRegionSelect(label: string) {
    const matched = displayedTopRegions.find((item) => item.label === label);
    if (matched?.borough) {
      toggleBoroughFilter(matched.borough);
    }
  }

  return (
    <div className="app-shell">
      <SidebarNav items={navItems} />

      <main className="dashboard-main">
        <FilterBar filters={filterGroups} />

        <section className="linked-filter-panel">
          <div>
            <p className="linked-filter-panel__eyebrow">点击联动筛选</p>
            <h3>地图、柱状图和排行榜现在支持联动</h3>
            <p>{activeFilterText}</p>
          </div>
          <div className="linked-filter-panel__actions">
            {activeFilter ? (
              <button
                type="button"
                className="linked-filter-chip linked-filter-chip--active"
                onClick={() => setActiveFilter(null)}
              >
                清空筛选
              </button>
            ) : null}
            <span className="linked-filter-chip">地图点选 borough</span>
            <span className="linked-filter-chip">星期柱切小时画像</span>
            <span className="linked-filter-chip">天气柱切散点</span>
          </div>
        </section>

        <SectionCard
          title="项目总览"
          subtitle="用总览页交代数据规模、分析范围与主要实验切片，适合作为首页首屏。"
          action={<a href="#prediction">查看模型结果</a>}
          className="section-card--hero"
        >
          <MetricGrid items={dashboardData.overviewMetrics} />
          <div className="panel-grid panel-grid--hero">
            <article className="chart-panel">
              <div className="chart-panel__head">
                <div>
                  <h3>订单趋势</h3>
                  <p>
                    {selectedBorough
                      ? `${selectedBorough} 的按天订单趋势。`
                      : "按天汇总真实订单量，取样展示整个时间范围。"}
                  </p>
                </div>
                <span>折线图</span>
              </div>
              <LineChart data={displayedDemandTrend} tone="sky" />
            </article>

            <article className="chart-panel chart-panel--summary">
              <div className="chart-panel__head">
                <div>
                  <h3>真实数据状态</h3>
                  <p>
                    当前页面图表全部来自真实 CSV/ZIP 聚合后的 `dashboard.json`。
                  </p>
                </div>
              </div>
              <ul className="summary-list">
                <li>
                  地图已升级为 NYC borough 真地图，支持点击 borough 联动。
                </li>
                <li>
                  页面不再依赖手写 mock 常量，替换数据只需重跑预处理脚本。
                </li>
                <li>
                  数据生成时间：
                  {new Date(dashboardData.generatedAt).toLocaleString("zh-CN")}
                </li>
              </ul>
            </article>
          </div>
        </SectionCard>

        <div id="spatial" className="section-grid">
          <SectionCard
            title="订单时空分析"
            subtitle="点击地图或热点区域后，相关图表会按 borough 联动切换。"
          >
            <div className="panel-grid panel-grid--two-columns">
              <article className="chart-panel chart-panel--tall chart-panel--map">
                <div className="chart-panel__head">
                  <div>
                    <h3>NYC 道路底图需求地图</h3>
                    <p>
                      基于道路底图展示 borough
                      订单总量，橙色亮点表示热点区域中心点。
                    </p>
                  </div>
                  <span>道路底图</span>
                </div>
                <MapChart
                  data={dashboardData.boroughDemand}
                  hotspots={mapHotspots}
                  selectedBorough={selectedBorough ?? undefined}
                  onSelectBorough={toggleBoroughFilter}
                />
              </article>

              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>热门区域 Top5</h3>
                    <p>
                      {selectedBorough
                        ? `${selectedBorough} 内的热点区域。`
                        : "点击排行也会回写 borough 筛选。"}
                    </p>
                  </div>
                  <span>排行图</span>
                </div>
                <RankedList
                  items={displayedTopRegions}
                  tone="violet"
                  selectedLabel={displayedTopRegions[0]?.label}
                  onSelect={handleTopRegionSelect}
                />
              </article>
            </div>

            <div className="panel-grid panel-grid--three-columns">
              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>小时分布</h3>
                    <p>
                      {selectedWeekday
                        ? `${selectedWeekday} 的 24 小时需求画像。`
                        : selectedBorough
                          ? `${selectedBorough} 的 24 小时需求分布。`
                          : "由真实订单时间直接按小时汇总。"}
                    </p>
                  </div>
                </div>
                <BarChart data={displayedHourlyDemand} tone="sky" />
              </article>

              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>星期模式</h3>
                    <p>点击某一天可联动到对应小时画像。</p>
                  </div>
                </div>
                <BarChart
                  data={displayedWeekdayDemand}
                  tone="amber"
                  selectedLabel={selectedWeekday ?? undefined}
                  onSelect={toggleWeekdayFilter}
                />
              </article>

              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>区域钻取入口</h3>
                    <p>
                      当前已接入 borough 级联动，后续可继续细化到 geohash
                      或街区。
                    </p>
                  </div>
                </div>
                <div className="drilldown-card">
                  <span>当前联动路径</span>
                  <strong>地图点击 到 borough 视图切换</strong>
                  <p>同步刷新趋势、小时分布、星期分布、天气对比和热点排行。</p>
                </div>
              </article>
            </div>
          </SectionCard>
        </div>

        <div id="weather" className="section-grid">
          <SectionCard
            title="天气影响分析"
            subtitle="点击天气类别后，散点图会切换到对应类别；若已选择 borough，则优先展示 borough 维度。"
          >
            <div className="panel-grid panel-grid--two-columns">
              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>温度与订单关系</h3>
                    <p>
                      {selectedBorough
                        ? `${selectedBorough} 的温度-需求散点。`
                        : selectedWeather
                          ? `${selectedWeather} 条件下的温度-需求散点。`
                          : "按小时将真实天气与真实需求对齐后得到温度-需求散点。"}
                    </p>
                  </div>
                  <span>散点图</span>
                </div>
                <ScatterChart points={displayedWeatherScatter} tone="teal" />
              </article>

              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>天气类别影响</h3>
                    <p>点击柱体可把散点图切到指定天气类别。</p>
                  </div>
                  <span>柱状图</span>
                </div>
                <BarChart
                  data={displayedPrecipImpact}
                  tone="teal"
                  selectedLabel={selectedWeather ?? undefined}
                  onSelect={toggleWeatherFilter}
                />
              </article>
            </div>
          </SectionCard>
        </div>

        <div id="features" className="section-grid">
          <SectionCard
            title="特征价值分析"
            subtitle="这里直接读取作者产出的特征分析报告，用于辅助判断哪些变量更值得保留。"
          >
            <div className="panel-grid panel-grid--two-columns">
              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>相关性 Top 特征</h3>
                    <p>
                      来自 `feature_analysis_report.md` 的 `importance_score`。
                    </p>
                  </div>
                  <span>条形排行</span>
                </div>
                <RankedList
                  items={dashboardData.featureImportance}
                  tone="sky"
                />
              </article>

              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>熵值与相关性分布</h3>
                    <p>横轴为归一化熵，纵轴近似映射特征重要性。</p>
                  </div>
                  <span>散点图</span>
                </div>
                <ScatterChart
                  points={dashboardData.entropyScatter}
                  tone="violet"
                />
              </article>
            </div>
          </SectionCard>
        </div>

        <div id="prediction" className="section-grid">
          <SectionCard
            title="预测评估与解释"
            subtitle="预测页维持全局基线结果，用于跟前面的筛选分析分工展示。"
          >
            <MetricGrid items={dashboardData.predictionMetrics} />
            <div className="panel-grid panel-grid--two-columns">
              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>真实值 vs 预测值</h3>
                    <p>
                      使用六月最后一周，按天汇总真实需求和前一日同小时基线预测。
                    </p>
                  </div>
                  <span>双折线</span>
                </div>
                <LineChart
                  data={actualVsPredictedSeries}
                  lines={[
                    dashboardData.actualVsPredicted.actual,
                    dashboardData.actualVsPredicted.predicted,
                  ]}
                  tone="amber"
                />
              </article>

              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>误差分布</h3>
                    <p>展示最后一周小时级基线预测误差的分桶统计。</p>
                  </div>
                  <span>直方图</span>
                </div>
                <BarChart data={dashboardData.errorHistogram} tone="violet" />
              </article>
            </div>

            <div className="panel-grid panel-grid--two-columns">
              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>误差最大日期</h3>
                    <p>按天聚合后，显示基线预测偏差最明显的日期。</p>
                  </div>
                  <span>误差排行</span>
                </div>
                <RankedList
                  items={dashboardData.predictionTopErrors}
                  tone="amber"
                />
              </article>

              <article className="chart-panel">
                <div className="chart-panel__head">
                  <div>
                    <h3>后续接入建议</h3>
                    <p>
                      地图和联动筛选已经接通，下一步适合继续下钻到更细粒度空间单元。
                    </p>
                  </div>
                </div>
                <ul className="summary-list">
                  <li>
                    可继续把 borough 下钻到 geohash、街区或 POI 周边热区。
                  </li>
                  <li>
                    如果拿到真实模型输出文件，可把预测页也做成 borough
                    联动版本。
                  </li>
                  <li>后续可增加图例联动、刷选高亮和多筛选组合。</li>
                </ul>
              </article>
            </div>
          </SectionCard>
        </div>
      </main>
    </div>
  );
}
