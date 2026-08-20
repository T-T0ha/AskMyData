/** Phase 0 view — data quality triage.
 *
 *  Shown before any semantic work so the user knows up front which sheets will
 *  just work and which need a decision. Each sheet also exposes what the
 *  structural repair actually did (banner rows skipped, header rows flattened,
 *  columns retyped), because a silent repair is not a trustworthy one.
 */

import { useState } from 'react'
import { Button, DataTable, EmptyState, Panel, StatusBadge, Tag } from './ui'

export const TRIAGE_META = {
  clean: { status: 'good', label: 'Clean', blurb: 'Ready to use as-is.' },
  fixable: { status: 'warning', label: 'Fixable', blurb: 'Issues the cleaning plan can resolve.' },
  needs_attention: {
    status: 'serious',
    label: 'Needs attention',
    blurb: 'Needs a decision from you.',
  },
  structural_issues: {
    status: 'critical',
    label: 'Structural issues',
    blurb: 'The sheet itself is broken.',
  },
}

const SEVERITY_STATUS = { info: 'neutral', warning: 'warning', error: 'critical' }

function HeaderRepair({ header }) {
  if (!header) return null
  const facts = []
  if (header.banner_rows?.length) facts.push(`skipped ${header.banner_rows.length} title row(s)`)
  if (header.is_multi_row) facts.push(`flattened header rows ${header.rows.join(' + ')}`)
  if (header.merged_ranges) facts.push(`${header.merged_ranges} merged cell range(s) expanded`)
  if (!facts.length) facts.push('single-row header, no repair needed')

  return (
    <div className="mt-3">
      <p className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
        Structural repair
      </p>
      <ul className="mt-1 space-y-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        {facts.map((fact) => (
          <li key={fact}>· {fact}</li>
        ))}
        <li>
          · header confidence{' '}
          <span className="tabular">{Math.round((header.confidence ?? 0) * 100)}%</span>
        </li>
      </ul>
      {header.original_labels?.some((parts) => parts.length > 1) && (
        <div className="mt-2 overflow-x-auto">
          <table className="border-collapse text-[11px]">
            <tbody>
              {header.columns.map((column, index) => {
                const parts = header.original_labels[index] ?? []
                if (parts.length < 2) return null
                return (
                  <tr key={column}>
                    <td className="pr-2" style={{ color: 'var(--text-muted)' }}>
                      {parts.join(' ▸ ')}
                    </td>
                    <td style={{ color: 'var(--text-secondary)' }}>→ {column}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function CleaningLog({ reports }) {
  const changed = (reports ?? []).filter((report) => report.changed || report.notes?.length)
  if (!changed.length) return null
  return (
    <div className="mt-3">
      <p className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
        Value cleaning
      </p>
      <ul className="mt-1 space-y-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        {changed.map((report) => (
          <li key={report.column}>
            · <span style={{ color: 'var(--text-secondary)' }}>{report.column}</span>{' '}
            {report.retyped_to && <>→ {report.retyped_to} </>}
            {report.detected_currency_symbol && <>(currency {report.detected_currency_symbol}) </>}
            {report.nulls_normalized > 0 && <>· {report.nulls_normalized} null marker(s) normalised </>}
            {report.percent_converted > 0 && <>· {report.percent_converted} percent(s) converted </>}
            {report.coerced_to_null > 0 && <>· {report.coerced_to_null} unparseable value(s) </>}
            {report.notes?.map((note) => (
              <span key={note} className="block pl-3">
                {note}
              </span>
            ))}
          </li>
        ))}
      </ul>
    </div>
  )
}

const SOURCE_LABELS = {
  excel: 'Excel sheet',
  csv: 'CSV file',
  sqlite: 'SQLite table',
  postgresql: 'PostgreSQL table',
  mysql: 'MySQL table',
  mariadb: 'MariaDB table',
  sql_dump: 'SQL dump table',
}

/** Keys the source database declared about itself.
 *
 *  Worth its own block rather than a line in the repair log: a declared key is
 *  stronger evidence than anything Phase 3 can detect from value overlap, and
 *  the user should see that the platform already knows it.
 */
function DeclaredSchema({ schema }) {
  const primaryKey = schema?.primary_key ?? []
  const foreignKeys = schema?.foreign_keys ?? []
  if (!primaryKey.length && !foreignKeys.length) return null

  return (
    <div className="mt-3">
      <p className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
        Declared by the source
      </p>
      <ul className="mt-1 space-y-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        {primaryKey.length > 0 && <li>· primary key: {primaryKey.join(', ')}</li>}
        {foreignKeys.map((foreignKey) => (
          <li key={foreignKey.columns.join(',')}>
            · foreign key: {foreignKey.columns.join(', ')} → {foreignKey.references_table}.
            {foreignKey.references_columns.join(', ')}
          </li>
        ))}
      </ul>
    </div>
  )
}

const KEY_SOURCE_LABELS = {
  declared: 'declared by the source database',
  detected: 'detected during ingestion',
  confirmed: 'confirmed by you',
}

/** What identifies a row — and, when nothing obvious does, the question.
 *
 *  A table with no key cannot be referenced by another table, so this is not
 *  a detail hidden behind "what was repaired?": it is the one Phase 0 answer
 *  that decides whether the exported database can have relationships at all.
 */
function RowIdentity({ sheet, onDecideKey, busy }) {
  const [selected, setSelected] = useState(0)
  const keys = sheet.key_analysis
  if (!keys || !Object.keys(keys).length) return null
  const candidates = keys.candidates ?? []

  const decided = keys.primary_key?.length > 0
  const asking = keys.needs_confirmation && candidates.length > 0

  return (
    <div
      className="mt-3 rounded-md border px-3 py-2"
      style={{
        borderColor: asking ? 'var(--status-warning)' : 'var(--border)',
        background: 'var(--surface-2)',
      }}
    >
      <p className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
        Row identity
      </p>

      {decided && (
        <p className="mt-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          <span className="font-medium">{keys.primary_key.join(' + ')}</span> identifies a row —{' '}
          {KEY_SOURCE_LABELS[keys.source] ?? keys.source}.
        </p>
      )}

      {asking && (
        <>
          <p className="mt-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
            No single column identifies a row here. These columns do it together — confirm the
            combination that means one row in your business, or add a numbered row_id instead.
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <select
              className="text-xs"
              value={selected}
              disabled={busy}
              aria-label="Candidate key"
              onChange={(event) => setSelected(Number(event.target.value))}
            >
              {candidates.map((candidate, index) => (
                <option key={candidate.label} value={index}>
                  {candidate.label}
                </option>
              ))}
            </select>
            <Button
              variant="primary"
              disabled={busy}
              onClick={() => onDecideKey?.(sheet.table_name, candidates[selected]?.columns ?? [])}
            >
              Use as key
            </Button>
            <Button disabled={busy} onClick={() => onDecideKey?.(sheet.table_name, [])}>
              None of these
            </Button>
          </div>
          <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            {candidates[selected]?.evidence}
          </p>
        </>
      )}

      {!decided && !asking && keys.needs_synthetic_key && (
        <p className="mt-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Nothing identifies a row. The cleaning plan will offer to add a numbered row_id.
        </p>
      )}

      {keys.notes?.length > 0 && (
        <ul className="mt-1 space-y-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {keys.notes.map((note) => (
            <li key={note}>· {note}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

function SheetCard({ sheet, onPreview, preview, onDecideKey, busy }) {
  const [open, setOpen] = useState(false)
  const meta = TRIAGE_META[sheet.triage] ?? TRIAGE_META.clean
  const sourceLabel = SOURCE_LABELS[sheet.source_kind] ?? sheet.source_kind

  return (
    <article className="panel p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold">{sheet.table_name}</h3>
          <p className="mt-0.5 truncate text-xs" style={{ color: 'var(--text-muted)' }}>
            from “{sheet.source_name}” · {sheet.source_file}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Tag muted title="where this table came from">
            {sourceLabel}
          </Tag>
          <StatusBadge status={meta.status} title={meta.blurb}>
            {meta.label}
          </StatusBadge>
        </div>
      </div>

      <p className="mt-2 tabular text-xs" style={{ color: 'var(--text-secondary)' }}>
        {sheet.row_count.toLocaleString()} rows × {sheet.column_count} columns
      </p>

      {sheet.skipped && (
        <p className="mt-2 text-xs" style={{ color: 'var(--status-critical)' }}>
          Skipped: {sheet.skip_reason}
        </p>
      )}

      {sheet.issues?.length > 0 && (
        <ul className="mt-3 space-y-1.5">
          {sheet.issues.map((issue) => (
            <li key={issue.code} className="flex items-start gap-2 text-xs">
              <span className="mt-0.5 shrink-0">
                <StatusBadge status={SEVERITY_STATUS[issue.severity]}>{issue.severity}</StatusBadge>
              </span>
              <span style={{ color: 'var(--text-secondary)' }}>
                {issue.message}
                {issue.columns?.length > 0 && (
                  <span className="ml-1 flex flex-wrap gap-1 pt-1">
                    {issue.columns.map((column) => (
                      <Tag key={column} muted>
                        {column}
                      </Tag>
                    ))}
                  </span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}

      {!sheet.skipped && <RowIdentity sheet={sheet} onDecideKey={onDecideKey} busy={busy} />}

      <div className="mt-3 flex gap-3 text-xs">
        <button
          type="button"
          className="focus-ring underline"
          style={{ color: 'var(--series-1)' }}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? 'Hide repair details' : 'What was repaired?'}
        </button>
        {!sheet.skipped && (
          <button
            type="button"
            className="focus-ring underline"
            style={{ color: 'var(--series-1)' }}
            onClick={() => onPreview(sheet.table_name)}
          >
            Preview data
          </button>
        )}
      </div>

      {open && (
        <>
          {sheet.source_kind === 'excel' || sheet.source_kind === 'csv' ? (
            <HeaderRepair header={sheet.header} />
          ) : (
            <DeclaredSchema schema={sheet.native_schema} />
          )}
          <CleaningLog reports={sheet.clean_reports} />
        </>
      )}

      {preview?.table === sheet.table_name && (
        <div className="mt-3 rounded border p-2" style={{ borderColor: 'var(--border)' }}>
          <DataTable columns={preview.columns} rows={preview.rows} />
        </div>
      )}
    </article>
  )
}

export function TriageBoard({ triage, onPreview, preview, onDecideKey, busy }) {
  if (!triage?.sheets?.length) {
    return <EmptyState>Add a file or a database connection to see its data quality triage.</EmptyState>
  }

  const counts = triage.summary?.triage_counts ?? {}
  return (
    <div className="space-y-4">
      <Panel
        title="Data quality triage"
        subtitle={`${triage.summary.sheet_count} table(s), ${triage.summary.total_rows.toLocaleString()} rows ingested`}
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {Object.entries(TRIAGE_META).map(([key, meta]) => (
            <div
              key={key}
              className="rounded-md border px-3 py-2"
              style={{ borderColor: 'var(--border)', background: 'var(--surface-2)' }}
            >
              <p className="tabular text-2xl font-semibold">{counts[key] ?? 0}</p>
              <div className="mt-1">
                <StatusBadge status={meta.status}>{meta.label}</StatusBadge>
              </div>
              <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                {meta.blurb}
              </p>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        {triage.sheets.map((sheet) => (
          <SheetCard
            key={sheet.id}
            sheet={sheet}
            onPreview={onPreview}
            preview={preview}
            onDecideKey={onDecideKey}
            busy={busy}
          />
        ))}
      </div>
    </div>
  )
}
