import { SectionCard } from '../components/SectionCard'
import { BarChart, ScatterChart } from '../components/charts'
import { useDashboardPageContext } from './useDashboardPageContext'

export function WeatherPage() {
  const {
    dashboardData,
    selectedBorough,
    selectedWeather,
    displayedWeatherScatter,
    displayedPrecipImpact,
    toggleWeatherFilter,
  } = useDashboardPageContext()

  return (
    <SectionCard title="天气影响分析" subtitle="这一页补充了温度分箱视图，并明确标出了温度和订单的展示上限。">
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
                    : '按小时将真实天气与真实需求对齐后得到温度-需求散点。'}
              </p>
            </div>
            <span>散点图</span>
          </div>
          <ScatterChart
            points={displayedWeatherScatter}
            tone="teal"
            xMaxOverride={50}
            yMaxOverride={4000}
            xAxisName="温度（°C）"
            yAxisName="订单量"
            xLabel="温度（°C）"
            yLabel="订单量"
          />
          <p className="chart-note">展示上限：温度到 50°C，订单量到 4000；超出范围的异常点不改变底层统计，只在视觉上裁切。</p>
        </article>

        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>温度分箱后平均订单</h3>
              <p>把温度按 5°C 分箱后取平均订单量，阅读上比原始散点更稳定。</p>
            </div>
            <span>分箱柱图</span>
          </div>
          <BarChart data={dashboardData.weatherTempBins} tone="teal" xAxisName="温度区间" yAxisName="平均订单量" />
        </article>
      </div>

      <div className="panel-grid panel-grid--two-columns">
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
            xAxisName="天气类别"
            yAxisName="平均订单量"
          />
        </article>

        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>阅读建议</h3>
              <p>天气页建议优先看分箱结果，再回到散点图查看具体波动形态。</p>
            </div>
          </div>
          <ul className="summary-list">
            <li>散点图更适合看温度和需求是否近似线性关系。</li>
            <li>分箱柱图更适合对外展示，因为噪声更小、趋势更稳。</li>
            <li>类别柱图适合看雨天、雪天、雾天对平均订单量的整体影响。</li>
          </ul>
        </article>
      </div>
    </SectionCard>
  )
}
