import type { PropsWithChildren, ReactNode } from 'react'

type SectionCardProps = PropsWithChildren<{
  title: string
  subtitle: string
  action?: ReactNode
  className?: string
}>

export function SectionCard({ title, subtitle, action, className, children }: SectionCardProps) {
  return (
    <section className={`section-card ${className ?? ''}`.trim()}>
      <header className="section-card__header">
        <div>
          <p className="section-card__eyebrow">Dashboard module</p>
          <h2>{title}</h2>
          <p className="section-card__subtitle">{subtitle}</p>
        </div>
        {action ? <div className="section-card__action">{action}</div> : null}
      </header>
      {children}
    </section>
  )
}
