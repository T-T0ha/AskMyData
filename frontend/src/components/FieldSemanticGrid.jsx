/** Phase 1 view — the Field Semantic View.
 *
 *  One row per column: detected type, taxonomy label, nullability, value
 *  distribution. Both detections are editable — a dropdown change is an
 *  override that is stored beside the detected value, never on top of it, so
 *  the system's accuracy stays measurable after correction.
 *
 *  Confidence is shown as a labelled status badge rather than colour alone.
 */

import { useMemo, useState } from 'react'
import { Distribution, DistributionStats, NullBar } from './Distribution'
import { EmptyState, Panel, StatusBadge, Tag } from './ui'

const CONFIDENCE_STATUS = (score) => (score >= 0.85 ? 'good' : score >= 0.6 ? 'warning' : 'serious')
const CONFIDENCE_LABEL = (score) => (score >= 0.85 ? 'high' : score >= 0.6 ? 'medium' : 'low')

function Confidence({ score, evidence }) {
  return (
    <span title={evidence?.join(' · ')}>
      <StatusBadge status={CONFIDENCE_STATUS(score)}>
        {CONFIDENCE_LABEL(score)} · {Math.round(score * 100)}%
      </StatusBadge>
    </span>
  )
}

function ColumnRow({ column, vocabulary, onOverride, busy }) {
  const [open, setOpen] = useState(false)
  const overridden = column.user_column_type || column.user_taxonomy_label

  return (
    <>
      <tr style={{ background: overridden ? 'var(--series-1-wash)' : undefined }}>
        <td className="border-b px-3 py-2 align-top" style={{ borderColor: 'var(--gridline)' }}>
          <button
            type="button"
            className="focus-ring text-left text-sm font-medium"
            onClick={() => setOpen((value) => !value)}
            title="Show samples and statistics"
          >
            {column.name}
          </button>
          {column.original_name && column.original_name !== column.name && (
            <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              was “{column.original_name}”
            </p>
          )}
        </td>

        <td className="border-b px-3 py-2 align-top" style={{ borderColor: 'var(--gridline)' }}>
          <select
            className="w-full text-xs"
            disabled={busy}
            value={column.effective_type}
            onChange={(event) => onOverride(column.id, { column_type: event.target.value })}
          >
            {vocabulary.column_types.map((type) => (
              <option key={type.value} value={type.value}>
                {type.value}
                {type.from_paper ? '' : ' *'}
              </option>
            ))}
          </select>
          <div className="mt-1">
            <Confidence score={column.type_confidence} evidence={column.type_evidence} />
          </div>
        </td>

        <td className="border-b px-3 py-2 align-top" style={{ borderColor: 'var(--gridline)' }}>
          <select
            className="w-full text-xs"
            disabled={busy}
            value={column.effective_label}
            onChange={(event) => onOverride(column.id, { taxonomy_label: event.target.value })}
            style={
              column.effective_label === 'unknown'
                ? { borderColor: 'var(--status-serious)' }
                : undefined
            }
          >
            {vocabulary.taxonomy_labels.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>
          <div className="mt-1 flex flex-wrap items-center gap-1">
            <Confidence score={column.taxonomy_confidence} />
            <Tag
              muted
              title={
                column.taxonomy_source === 'rule'
                  ? `matched rule ${column.taxonomy_rule}`
                  : column.taxonomy_source === 'claude'
                    ? 'labelled by Claude (no rule matched)'
                    : 'no rule matched and no model label was available'
              }
            >
              {column.taxonomy_source}
            </Tag>
            {column.is_additive && <Tag title="Values can be meaningfully summed">Σ additive</Tag>}
          </div>
        </td>

        <td className="border-b px-3 py-2 align-top" style={{ borderColor: 'var(--gridline)' }}>
          <NullBar ratio={column.null_ratio} />
          <p className="tabular mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            {column.unique_count.toLocaleString()} distinct
          </p>
        </td>

        <td
          className="w-64 border-b px-3 py-2 align-top"
          style={{ borderColor: 'var(--gridline)' }}
        >
          <Distribution distribution={column.distribution} />
        </td>
      </tr>

      {open && (
        <tr>
          <td
            colSpan={5}
            className="border-b px-3 py-3"
            style={{ borderColor: 'var(--gridline)', background: 'var(--surface-2)' }}
          >
            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <p className="mb-1 text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Why this type
                </p>
                <ul className="space-y-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  {(column.type_evidence ?? []).map((line) => (
                    <li key={line}>· {line}</li>
                  ))}
                  {column.user_column_type && (
                    <li style={{ color: 'var(--series-1)' }}>
                      · you changed this from “{column.column_type}”
                    </li>
                  )}
                </ul>
              </div>
              <div>
                <p className="mb-1 text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Sample values
                </p>
                <ul className="space-y-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  {(column.sample_values ?? []).map((value, index) => (
                    <li key={index} className="tabular truncate">
                      {String(value)}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="mb-1 text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                  Distribution
                </p>
                <DistributionStats distribution={column.distribution} />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export function FieldSemanticGrid({ semantics, vocabulary, onOverride, busy }) {
  const [filter, setFilter] = useState('all')

  const tables = useMemo(() => {
    if (filter === 'unknown') {
      return semantics.tables
        .map((table) => ({
          ...table,
          columns: table.columns.filter((column) => column.effective_label === 'unknown'),
        }))
        .filter((table) => table.columns.length)
    }
    if (filter === 'low') {
      return semantics.tables
        .map((table) => ({
          ...table,
          columns: table.columns.filter(
            (column) => Math.min(column.type_confidence, column.taxonomy_confidence) < 0.85,
          ),
        }))
        .filter((table) => table.columns.length)
    }
    return semantics.tables
  }, [semantics, filter])

  if (!semantics?.tables?.length) {
    return <EmptyState>Run the semantic analysis to see detected types and labels.</EmptyState>
  }

  return (
    <div className="space-y-4">
      <Panel
        title="Field semantics"
        subtitle={`${semantics.total_columns} columns · ${semantics.unknown_count} unlabelled · ${semantics.validated_count} corrected by you`}
        actions={
          <div className="flex items-center gap-1 text-xs">
            {[
              ['all', 'All'],
              ['low', 'Needs review'],
              ['unknown', 'Unlabelled'],
            ].map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setFilter(key)}
                className="focus-ring rounded px-2 py-1"
                style={{
                  background: filter === key ? 'var(--series-1)' : 'var(--surface-3)',
                  color: filter === key ? '#fff' : 'var(--text-secondary)',
                }}
              >
                {label}
              </button>
            ))}
          </div>
        }
      >
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Types marked <span className="font-medium">*</span> (currency, boolean, identifier) extend
          the six types defined in SemTabla. Click a column name for its evidence, samples and
          statistics. Changing a dropdown records a correction without discarding what was detected.
        </p>
      </Panel>

      {tables.map((table) => (
        <section key={table.table} className="panel overflow-hidden">
          <header className="border-b px-4 py-2" style={{ borderColor: 'var(--border)' }}>
            <h3 className="text-sm font-semibold">{table.table}</h3>
          </header>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  {['Column', 'Data type', 'Taxonomy', 'Nulls / distinct', 'Value distribution'].map(
                    (heading) => (
                      <th
                        key={heading}
                        className="border-b px-3 py-2 text-left font-medium"
                        style={{ borderColor: 'var(--gridline)' }}
                      >
                        {heading}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {table.columns.map((column) => (
                  <ColumnRow
                    key={column.id}
                    column={column}
                    vocabulary={vocabulary}
                    onOverride={onOverride}
                    busy={busy}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  )
}
