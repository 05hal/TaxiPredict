import { Outlet } from 'react-router-dom'
import { FilterBar } from '../FilterBar'
import { SidebarNav } from '../SidebarNav'
import { filterGroups, navItems } from '../../data/dashboard'
import { useDashboardState } from '../../hooks/useDashboardState'

export function DashboardLayout() {
  const { dashboardData, errorMessage, viewModel } = useDashboardState()

  if (errorMessage) {
    return (
      <div className="app-shell">
        <SidebarNav items={navItems} />
        <main className="dashboard-main">
          <section className="state-panel">
            <p className="state-panel__eyebrow">数据加载失败</p>
            <h2>无法读取真实面板数据</h2>
            <p>{errorMessage}</p>
            <p>请先执行 `npm run prepare:data` 生成 `public/data/dashboard.json`。</p>
          </section>
        </main>
      </div>
    )
  }

  if (!dashboardData || !viewModel) {
    return (
      <div className="app-shell">
        <SidebarNav items={navItems} />
        <main className="dashboard-main">
          <section className="state-panel">
            <p className="state-panel__eyebrow">正在加载</p>
            <h2>正在读取真实 CSV 聚合结果</h2>
            <p>页面会直接消费由真实 Uber 订单、天气数据和特征分析报告生成的统计结果。</p>
          </section>
        </main>
      </div>
    )
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
            <p>{viewModel.activeFilterText}</p>
          </div>
          <div className="linked-filter-panel__actions">
            {viewModel.activeFilter ? (
              <button
                type="button"
                className="linked-filter-chip linked-filter-chip--active"
                onClick={viewModel.clearFilter}
              >
                清空筛选
              </button>
            ) : null}
            <span className="linked-filter-chip">地图点选 borough</span>
            <span className="linked-filter-chip">geohash 细粒度热点</span>
            <span className="linked-filter-chip">星期柱切小时画像</span>
            <span className="linked-filter-chip">天气柱切散点</span>
          </div>
        </section>
        <Outlet context={viewModel} />
      </main>
    </div>
  )
}
