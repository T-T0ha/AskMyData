/** The automated data-quality report (§5.4) — a per-table score derived from
 *  the persisted semantic layer, not a fresh analysis pass. Completeness and
 *  uniqueness/consistency/validity are the four dimensions the Report names;
 *  the before/after comparison only ever covers what an archived build can
 *  honestly support (see `app/export/quality.py`), so a table's `previous`
 *  entry may be missing consistency/uniqueness even when its current score
 *  has them.
 */

import { useEffect } from 'react'
import { EmptyState, Panel, Tag } from './ui'

const DIMENSIONS = [
  { key: 'completeness', label: 'Completeness' },
  { key: 'uniqueness', label: 'Uniqueness' },
  { key: 'consistency', label: 'Consistency' },
  { key: 'validity', label: 'Validity' },
]

function ScoreBar({ label, value }) {
  if (value == null) return null
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 shrink-0" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </span>
      <div
        className="h-1.5 flex-1 overflow-hidden rounded-full"
        style={{ background: 'var(--surface-3)' }}
      >
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.max(0, Math.min(value, 100))}%`, background: 'var(--series-1)' }}
        />
      </div>
      <span className="w-8 shrink-0 text-right tabular" style={{ color: 'var(--text-muted)' }}>
        {Math.round(value)}
      </span>
    </div>
  )
}

function PreviousBuildNote({ previous }) {
  if (!previous) return null
  const parts = [
    previous.completeness != null && `completeness ${Math.round(previous.completeness)}`,
    previous.validity != null && `validity ${Math.round(previous.validity)}`,
  ].filter(Boolean)
  if (!parts.length) return null
  return (
    <p className="mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
      Previous build — {parts.join(', ')}
    </p>
  )
}

function TableScoreCard({ table }) {
  return (
    <Panel
      title={table.name}
      subtitle={table.summary}
      actions={<Tag title="Average of the four dimensions below">{`Overall ${Math.round(table.overall)}`}</Tag>}
    >
      <div className="space-y-2">
        {DIMENSIONS.map(({ key, label }) => (
          <ScoreBar key={key} label={label} value={table[key]} />
        ))}
      </div>
      <PreviousBuildNote previous={table.previous} />
    </Panel>
  )
}

export function QualityReportPanel({ exported, report, busy, onRefresh }) {
  useEffect(() => {
    if (!exported) return
    onRefresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exported])

  if (!exported) {
    return (
      <Panel
        title="Data quality"
        subtitle="A per-table score derived from the semantic layer, not a fresh analysis pass."
      >
        <EmptyState>Build the semantic layer first — there is nothing to score yet.</EmptyState>
      </Panel>
    )
  }

  if (!report) {
    return (
      <Panel title="Data quality">
        <EmptyState>{busy ? 'Scoring…' : 'No report yet.'}</EmptyState>
      </Panel>
    )
  }

  return (
    <div className="space-y-4">
      <Panel
        title="Data quality"
        subtitle="Completeness, uniqueness, consistency and validity — each already computed during enrichment, only read and scored here."
        actions={<Tag title="Average across every table">{`Overall ${Math.round(report.overall)}`}</Tag>}
      />
      {report.tables.length === 0 ? (
        <EmptyState>No tables to score yet.</EmptyState>
      ) : (
        report.tables.map((table) => <TableScoreCard key={table.name} table={table} />)
      )}
    </div>
  )
}
