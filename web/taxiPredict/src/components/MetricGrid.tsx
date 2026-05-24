import type { Metric } from '../data/dashboard'

type MetricGridProps = {
  items: Metric[]
}

export function MetricGrid({ items }: MetricGridProps) {
  return (
    <div className="metric-grid">
      {items.map((item) => (
        <article key={item.label} className={`metric-card metric-card--${item.tone ?? 'sky'}`}>
          <p>{item.label}</p>
          <strong>{item.value}</strong>
          <span>{item.change}</span>
        </article>
      ))}
    </div>
  )
}
