/** Semantic layer — the last stage, and the one that produces something the
 *  user can take away.
 *
 *  Two audiences read this screen. Somebody who wants the database wants to
 *  know it worked and where it is; somebody who is about to trust it wants to
 *  know what the platform could *not* assert — a reference with orphan rows, a
 *  key that cleaning broke, a column renamed to fit SQL. The summary answers
 *  the first in one line, and everything below it answers the second, because
 *  a silent downgrade is the one outcome this project exists to avoid.
 */

import { useState } from 'react'
import { Button, EmptyState, Panel, StatusBadge, Tag } from './ui'

function Stat({ label, value, title }) {
  return (
    <div
      className="rounded border px-3 py-2"
      style={{ borderColor: 'var(--border)' }}
      title={title}
    >
      <div className="tabular text-lg font-semibold">{value}</div>
      <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        {label}
      </div>
    </div>
  )
}

function download(name, body, type) {
  const url = URL.createObjectURL(new Blob([body], { type }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function ColumnRow({ column }) {
  return (
    <tr>
      <td className="py-1 pr-3 font-mono text-[11px]">
        {column.name}
        {column.renamed && (
          <span className="ml-1" style={{ color: 'var(--text-muted)' }} title="renamed for SQL">
            ← {column.source_name}
          </span>
        )}
      </td>
      <td className="py-1 pr-3 font-mono text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {column.sql_type}
        {!column.nullable && ' NOT NULL'}
      </td>
      <td className="py-1 pr-3 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {column.taxonomy_label.replaceAll('_', ' ')}
      </td>
      <td className="py-1 text-[11px]">
        {column.is_primary_key && <Tag title="primary key">key</Tag>}
        {column.is_foreign_key && (
          <Tag title={`references ${column.references_table}.${column.references_column}`}>
            → {column.references_table}
          </Tag>
        )}
        {column.is_additive && <Tag muted title="can be summed">Σ</Tag>}
      </td>
    </tr>
  )
}

function TableCard({ table }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="rounded border p-3" style={{ borderColor: 'var(--border)' }}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-3 text-left"
      >
        <span className="font-mono text-sm font-semibold">{table.name}</span>
        <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {table.table_type !== 'unknown' && `${table.table_type.replaceAll('_', ' ')} · `}
          {table.rows_written?.toLocaleString()} rows · {table.columns.length} columns
          {table.primary_key.length > 0 && ` · key ${table.primary_key.join(' + ')}`}
        </span>
      </button>
      {table.description && (
        <p className="mt-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          {table.description}
        </p>
      )}
      {table.notes?.length > 0 && (
        <ul className="mt-1 text-[11px]" style={{ color: 'var(--warn)' }}>
          {table.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
      {open && (
        <table className="mt-2 w-full">
          <tbody>
            {table.columns.map((column) => (
              <ColumnRow key={column.name} column={column} />
            ))}
          </tbody>
        </table>
      )}
    </li>
  )
}

export function ExportPanel({
  state,
  busy,
  onExport,
  onDownloadBundle,
  onDownloadDdl,
  onDownloadDocs,
}) {
  const [showDdl, setShowDdl] = useState(false)

  if (!state?.exported) {
    return (
      <Panel
        title="Semantic layer"
        subtitle="Build the clean database: one table per sheet, the keys and references you confirmed, and a sem_metadata table describing every column."
      >
        <EmptyState>
          <span className="mr-2">Nothing exported yet.</span>
          <Button variant="primary" disabled={busy} onClick={onExport}>
            Build the database
          </Button>
        </EmptyState>
      </Panel>
    )
  }

  const report = state.report ?? {}
  const warnings = report.warnings ?? []
  const unenforced = (report.foreign_keys ?? []).filter((fk) => !fk.enforced)
  const cleaning = report.cleaning

  return (
    <div className="space-y-4">
      <Panel
        title="Semantic layer"
        subtitle={`Written to ${state.target} · ${state.dialect} · version ${
          state.semantic_version ?? 1
        } · ${new Date(state.exported_at).toLocaleString()}`}
        actions={
          <div className="flex items-center gap-2">
            <Button onClick={onDownloadDocs} title="the schema, written out for a person to read">
              Download docs
            </Button>
            <Button onClick={onDownloadDdl}>Download SQL</Button>
            <Button onClick={onDownloadBundle}>Download JSON</Button>
            <Button variant="primary" disabled={busy} onClick={onExport}>
              Rebuild
            </Button>
          </div>
        }
      >
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
          <Stat label="tables" value={state.table_count} />
          <Stat label="rows" value={state.row_count.toLocaleString()} />
          <Stat label="columns described" value={state.column_count} title="rows in sem_metadata" />
          <Stat
            label="enforced references"
            value={report.enforced_foreign_keys ?? 0}
            title="foreign key constraints the database checks"
          />
          <Stat
            label="embedded"
            value={state.embedded_count}
            title={
              state.vector_index
                ? 'descriptions embedded, with an HNSW index for nearest-neighbour search'
                : 'descriptions embedded; no vector index on this database'
            }
          />
        </div>

        {cleaning && !cleaning.complete && (
          <p className="mt-3 text-xs" style={{ color: 'var(--warn)' }}>
            The cleaning plan for this dataset is still open ({cleaning.completed_steps} of{' '}
            {cleaning.step_count} steps). What was exported is the tables as they are now —
            finish the plan and rebuild to include the rest.
          </p>
        )}
      </Panel>

      {(warnings.length > 0 || unenforced.length > 0) && (
        <Panel
          title="What was not asserted"
          subtitle="Everything the export could not state as a constraint, and why"
        >
          <ul className="space-y-1.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
            {warnings.map((warning) => (
              <li key={warning} className="flex gap-2">
                <StatusBadge status="warning">note</StatusBadge>
                <span>{warning}</span>
              </li>
            ))}
          </ul>
          {unenforced.length > 0 && (
            <p className="mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
              A reference that is not enforced is still recorded in sem_metadata — questions can
              still join on it, and the database will not police it.
            </p>
          )}
        </Panel>
      )}

      <Panel title="What was built" subtitle="Click a table to see its columns">
        <ul className="space-y-2">
          {(report.tables ?? []).map((table) => (
            <TableCard key={table.name} table={table} />
          ))}
        </ul>
      </Panel>

      {(report.widened_types?.length > 0 || report.renamed?.length > 0) && (
        <Panel title="Changes the database required">
          {report.renamed?.length > 0 && (
            <div className="mb-3">
              <h3 className="mb-1 text-xs font-semibold">Renamed to be valid SQL</h3>
              <ul className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                {report.renamed.map((entry) => (
                  <li key={`${entry.kind}-${entry.table ?? ''}-${entry.from}`} className="font-mono">
                    {entry.table ? `${entry.table}.` : ''}
                    {entry.from} → {entry.to}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {report.widened_types?.length > 0 && (
            <div>
              <h3 className="mb-1 text-xs font-semibold">
                Types widened so no value was rounded or cut
              </h3>
              <ul className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                {report.widened_types.map((entry) => (
                  <li key={`${entry.table}.${entry.column}`}>
                    <span className="font-mono">
                      {entry.table}.{entry.column}
                    </span>{' '}
                    — {entry.note}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Panel>
      )}

      <Panel
        title="The script"
        actions={
          <Button onClick={() => setShowDdl((value) => !value)}>
            {showDdl ? 'Hide' : 'Show'}
          </Button>
        }
      >
        {showDdl ? (
          <pre
            className="overflow-x-auto rounded p-3 text-[11px] leading-relaxed"
            style={{ background: 'var(--surface-3)' }}
          >
            {report.ddl}
          </pre>
        ) : (
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            The exact CREATE TABLE statements, as PostgreSQL spells them.
          </p>
        )}
      </Panel>
    </div>
  )
}

export { download }
