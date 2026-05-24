import { SectionCard } from '../components/SectionCard'
import { RankedList, ScatterChart } from '../components/charts'
import { useDashboardPageContext } from './useDashboardPageContext'

export function FeaturesPage() {
  const { dashboardData } = useDashboardPageContext()

  return (
    <SectionCard title="特征价值分析" subtitle="这里直接读取作者产出的特征分析报告，并补充更清晰的坐标轴语义。">
      <div className="panel-grid panel-grid--two-columns">
        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>相关性 Top 特征</h3>
              <p>来自 `feature_analysis_report.md` 的 `importance_score`。</p>
            </div>
            <span>条形排行</span>
          </div>
          <RankedList items={dashboardData.featureImportance} tone="sky" />
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
            xAxisName="归一化熵（%）"
            yAxisName="特征重要性映射值"
            xLabel="归一化熵（%）"
            yLabel="特征重要性映射值"
          />
        </article>
      </div>
    </SectionCard>
  )
}
