import { useOutletContext } from 'react-router-dom'
import type { DashboardViewModel } from '../hooks/useDashboardState'

export function useDashboardPageContext() {
  return useOutletContext<DashboardViewModel>()
}
