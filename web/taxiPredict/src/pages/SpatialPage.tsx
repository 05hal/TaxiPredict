import { SectionCard } from '../components/SectionCard'
import { BarChart, GeoPointMapChart, HeatGridChart, MapChart, RankedList } from '../components/charts'
import { useDashboardPageContext } from './useDashboardPageContext'

export function SpatialPage() {
  const {
    dashboardData,
    selectedBorough,
    selectedWeekday,
    displayedHourlyDemand,
    displayedWeekdayDemand,
    displayedTopRegions,
    mapHotspots,
    toggleBoroughFilter,
    toggleWeekdayFilter,
    handleTopRegionSelect,
  } = useDashboardPageContext()

  const displayedGeohashPoints = selectedBorough
    ? dashboardData.geohashPoints.filter((item) => item.borough === selectedBorough)
    : dashboardData.geohashPoints

  return (
    <SectionCard title="订单时空分析" subtitle="这一页补充了 geohash 热点和小时 x 星期热力矩阵，空间层级比之前更细。">
      <div className="panel-grid panel-grid--two-columns">
        <article className="chart-panel chart-panel--tall chart-panel--map">
          <div className="chart-panel__head">
            <div>
              <h3>NYC 道路底图需求地图</h3>
              <p>基于道路底图展示 borough 订单总量，橙色亮点表示热点区域中心点。</p>
            </div>
            <span>道路底图</span>
          </div>
          <MapChart
            data={dashboardData.boroughDemand}
            hotspots={mapHotspots}
            selectedBorough={selectedBorough ?? undefined}
            onSelectBorough={toggleBoroughFilter}
          />
          <p className="chart-note">口径：蓝色填色按 borough 汇总，橙色热点来自真实坐标聚合后的区域中心点。</p>
        </article>

        <article className="chart-panel chart-panel--tall chart-panel--map">
          <div className="chart-panel__head">
            <div>
              <h3>Geohash 热点分布</h3>
              <p>{selectedBorough ? `${selectedBorough} 内的 geohash 聚合热点。` : '将原始订单编码为 geohash 后展示更细粒度热点。'}</p>
            </div>
            <span>geohash</span>
          </div>
          <GeoPointMapChart points={displayedGeohashPoints} selectedBorough={selectedBorough ?? undefined} />
          <p className="chart-note">口径：当前使用 geohash 6 位编码聚合，点越大表示该 geohash 单元累计订单越多。</p>
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
                    : '由真实订单时间直接按小时汇总。'}
              </p>
            </div>
          </div>
          <BarChart data={displayedHourlyDemand} tone="sky" xAxisName="小时" yAxisName="订单量" />
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
            xAxisName="星期"
            yAxisName="订单量"
          />
        </article>

        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>热门区域 Top5</h3>
              <p>{selectedBorough ? `${selectedBorough} 内的热点区域。` : '点击排行也会回写 borough 筛选。'}</p>
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

      <div className="panel-grid panel-grid--two-columns">
        <article className="chart-panel chart-panel--wide">
          <div className="chart-panel__head">
            <div>
              <h3>小时 x 星期热力图</h3>
              <p>一眼看清工作日和周末在不同小时段的需求差异，是比单独柱图更强的节奏视图。</p>
            </div>
            <span>矩阵热力</span>
          </div>
          <HeatGridChart
            data={dashboardData.weekdayHourHeatmap}
            xLabels={Array.from({ length: 24 }, (_, index) => String(index).padStart(2, '0'))}
            yLabels={['周一', '周二', '周三', '周四', '周五', '周六', '周日']}
            tone="sky"
          />
          <p className="chart-note">口径：来自原始订单的小时与星期二维聚合，不受 borough 筛选影响，用于展示全市节奏。</p>
        </article>

        <article className="chart-panel">
          <div className="chart-panel__head">
            <div>
              <h3>区域钻取说明</h3>
              <p>这轮已经从 borough 级补到 geohash 热点，后续如果需要还能继续做街区级细化。</p>
            </div>
          </div>
          <ul className="summary-list">
            <li>当前地图支持 borough 级点击联动，geohash 地图负责展示更细空间热点。</li>
            <li>工作日筛选仍然作用于小时柱图，适合和热力矩阵配合解读。</li>
            <li>如果后续增加时间滑块，可让 geohash 热点按小时演化。</li>
          </ul>
        </article>
      </div>
    </SectionCard>
  )
}
