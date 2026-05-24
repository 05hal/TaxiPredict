import { MetricGrid } from '../components/MetricGrid'
import { SectionCard } from '../components/SectionCard'
import { BarChart, LineChart, RankedList, ScatterChart } from '../components/charts'
import { useDashboardPageContext } from './useDashboardPageContext'

export function PredictionPage() {
  const { dashboardData, actualVsPredictedSeries } = useDashboardPageContext()

  return (
    <SectionCard title="预测评估与解释" subtitle="预测页补充了拟合散点和残差走势，用于更直观看基线预测的稳定性。">
      <MetricGrid items={dashboardData.predictionMetrics} />
      <div className="panel-grid panel-grid--two-columns">
        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>真实值 vs 预测值</h3>
              <p>使用六月最后一周，按天汇总真实需求和前一日同小时基线预测。</p>
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
            xAxisName="日期"
            yAxisName="订单量"
          />
        </article>

        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>拟合散点</h3>
              <p>横轴是真实值，纵轴是预测值，离对角线越近说明拟合越稳定。</p>
            </div>
            <span>散点图</span>
          </div>
          <ScatterChart
            points={dashboardData.actualPredictedScatter}
            tone="amber"
            xAxisName="真实订单量"
            yAxisName="预测订单量"
            xLabel="真实订单量"
            yLabel="预测订单量"
          />
        </article>
      </div>

      <div className="panel-grid panel-grid--two-columns">
        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>误差分布</h3>
              <p>展示最后一周小时级基线预测误差的分桶统计。</p>
            </div>
            <span>直方图</span>
          </div>
          <BarChart data={dashboardData.errorHistogram} tone="violet" xAxisName="误差区间" yAxisName="样本数" />
        </article>

        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>残差走势</h3>
              <p>按天汇总预测值减真实值，正值代表高估，负值代表低估。</p>
            </div>
            <span>折线图</span>
          </div>
          <LineChart data={dashboardData.residualTrend} tone="violet" xAxisName="日期" yAxisName="残差" />
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
          <RankedList items={dashboardData.predictionTopErrors} tone="amber" />
        </article>

        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>模型说明</h3>
              <p>当前仍然是基线预测页，适合用于展示误差分布和接入真实模型输出前的对照基线。</p>
            </div>
          </div>
          <ul className="summary-list">
            <li>双折线回答“整体跟踪趋势如何”。</li>
            <li>拟合散点回答“预测值和真实值的贴合程度如何”。</li>
            <li>残差走势回答“哪些天系统性高估或低估”。</li>
          </ul>
        </article>
      </div>
    </SectionCard>
  )
}
