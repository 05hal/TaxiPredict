import { NavLink } from 'react-router-dom'
import type { NavItem } from '../data/dashboard'

type SidebarNavProps = {
  items: NavItem[]
}

export function SidebarNav({ items }: SidebarNavProps) {
  return (
    <aside className="sidebar">
      <div>
        <p className="sidebar__kicker">TaxiPredict</p>
        <h1>NYC Taxi demand cockpit</h1>
        <p className="sidebar__lead">
          用一套 React 面板同时覆盖订单洞察、天气影响、特征诊断与模型评估。
        </p>
      </div>

      <nav className="sidebar__nav" aria-label="页面分区导航">
        {items.map((item, index) => (
          <NavLink
            key={item.id}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              `sidebar__nav-item${isActive ? ' sidebar__nav-item--active' : ''}`
            }
          >
            <span className="sidebar__nav-index">0{index + 1}</span>
            <span>
              <strong>{item.label}</strong>
              <small>{item.description}</small>
            </span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar__footer">
        <p>当前代码为前端骨架，后续只需把 `data/dashboard.ts` 替换成真实 API 或 CSV 解析结果。</p>
      </div>
    </aside>
  )
}
