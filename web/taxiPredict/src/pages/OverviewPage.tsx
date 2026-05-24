import { MetricGrid } from '../components/MetricGrid'
import { SectionCard } from '../components/SectionCard'
import { CalendarHeatmapChart, LineChart } from '../components/charts'
import { useDashboardPageContext } from './useDashboardPageContext'

export function OverviewPage() {
  const { dashboardData, displayedDemandTrend, selectedBorough } = useDashboardPageContext()

  return (
    <SectionCard
      title="项目总览"
      subtitle="总览页补充了真实数据口径、订单日历热力图和按天趋势，便于先快速建立全局认知。"
      action={undefined}
      className="section-card--hero"
    >
      <MetricGrid items={dashboardData.overviewMetrics} />
      <div className="panel-grid panel-grid--hero">
        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>订单趋势</h3>
              <p>{selectedBorough ? `${selectedBorough} 的按天订单趋势。` : '按天汇总真实订单量，取样展示整个时间范围。'}</p>
            </div>
            <span>折线图</span>
          </div>
          <LineChart data={displayedDemandTrend} tone="sky" xAxisName="日期" yAxisName="订单量" />
          <p className="chart-note">口径：原始订单按自然日聚合，当前图只做可视化抽样，不改变底层统计结果。</p>
        </article>

        <article className="chart-panel chart-panel--summary">
          <div className="chart-panel__head">
            <div>
              <h3>真实数据状态</h3>
              <p>当前页面图表全部来自真实 CSV/ZIP 聚合后的 `dashboard.json`。</p>
            </div>
          </div>
          <ul className="summary-list">
            <li>地图已升级为 NYC 道路底图，并支持 borough 联动与 geohash 热点展示。</li>
            <li>天气页补充了温度分箱视图，预测页补充了残差走势和拟合散点。</li>
            <li>数据生成时间：{new Date(dashboardData.generatedAt).toLocaleString('zh-CN')}</li>
          </ul>
        </article>
      </div>

      <div className="panel-grid panel-grid--two-columns">
        <article className="chart-panel chart-panel--wide">
          <div className="chart-panel__head">
            <div>
              <h3>订单日历热力图</h3>
              <p>按日期查看 2014-05 到 2014-06 的日订单量，方便发现周末节奏和异常日期。</p>
            </div>
            <span>日历热力</span>
          </div>
          <CalendarHeatmapChart data={dashboardData.dailyCalendar} tone="violet" />
          <p className="chart-note">口径：横向按日展示，不受 borough 筛选影响，用来保留全局节奏参考。</p>
        </article>

        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>阅读建议</h3>
              <p>建议先从时间节奏入手，再进入地图和天气页看空间与外部因素。</p>
            </div>
          </div>
          <ul className="summary-list">
            <li>先看趋势和日历热力，识别整体波动与高需求日期。</li>
            <li>再到时空页查看 borough 分布、geohash 热点和小时 x 星期矩阵。</li>
            <li>最后去天气页和预测页解释需求变化与误差来源。</li>
          </ul>
        </article>
      </div>
    </SectionCard>
  )
}
