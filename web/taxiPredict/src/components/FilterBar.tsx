import type { FilterGroup } from '../data/dashboard'

type FilterBarProps = {
  filters: FilterGroup[]
}

export function FilterBar({ filters }: FilterBarProps) {
  return (
    <div className="filter-bar">
      {filters.map((filter) => (
        <button key={filter.label} type="button" className="filter-pill">
          <span>{filter.label}</span>
          <strong>{filter.value}</strong>
        </button>
      ))}
    </div>
  )
}
