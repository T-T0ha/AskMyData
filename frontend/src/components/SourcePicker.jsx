/** Phase 0 view — choosing a source.
 *
 *  The proposal accepts five kinds of input, which split into two gestures:
 *  a file you hand over (Excel, CSV, SQLite, SQL dump) and a database you
 *  point at. Connecting is deliberately two steps — inspect, then choose —
 *  because a real database has more tables than anyone wants ingested, and
 *  the table list with row counts is the only way to pick well.
 */

import { useRef, useState } from 'react'
import { Button, Panel, StatusBadge, Tag } from './ui'

const FILE_KIND_FALLBACK = [
  { kind: 'excel', label: 'Excel workbook', extensions: ['.xlsx', '.xlsm'] },
  { kind: 'csv', label: 'CSV / TSV file', extensions: ['.csv', '.tsv'] },
  { kind: 'sqlite', label: 'SQLite database', extensions: ['.db', '.sqlite'] },
  { kind: 'sql_dump', label: 'SQL dump', extensions: ['.sql'] },
]

function formatRows(count) {
  if (count === null || count === undefined) return 'unknown size'
  return `${count.toLocaleString()} row${count === 1 ? '' : 's'}`
}

function FileTab({ vocabulary, session, busy, onUpload }) {
  const fileInput = useRef(null)
  const kinds = vocabulary?.accepted_file_kinds ?? FILE_KIND_FALLBACK

  return (
    <>
      <div
        className="rounded-md border border-dashed p-8 text-center"
        style={{ borderColor: 'var(--baseline)' }}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          onUpload(Array.from(event.dataTransfer.files ?? []))
        }}
      >
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Drop file(s) here, or
        </p>
        {/* Multiple, because "one CSV per table" is how a set of exports
            arrives — dropping five of them should not silently ingest one. */}
        <input
          ref={fileInput}
          type="file"
          multiple
          accept={vocabulary?.accepted_extensions?.join(',')}
          className="hidden"
          onChange={(event) => onUpload(Array.from(event.target.files ?? []))}
        />
        <div className="mt-3">
          <Button variant="primary" disabled={busy} onClick={() => fileInput.current?.click()}>
            Choose file(s)
          </Button>
        </div>
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          {kinds.map((kind) => (
            <Tag key={kind.kind} title={kind.extensions.join(' ')}>
              {kind.label}
            </Tag>
          ))}
        </div>
        <p className="mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Your data never leaves this machine. Only column names, detected types and at most five
          sample values per column are ever sent to the language model.
        </p>
      </div>

      {session?.sheets?.length > 0 && (
        <p className="mt-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
          {session.sheets.length} table(s) already ingested from{' '}
          {(session.source_files ?? []).join(', ')}. Add another source to this session at any time.
        </p>
      )}
    </>
  )
}

function DatabaseTab({ vocabulary, busy, onInspect, onConnect }) {
  const [url, setUrl] = useState('')
  const [inspection, setInspection] = useState(null)
  const [selected, setSelected] = useState([])

  const backends = vocabulary?.database_backends ?? ['postgresql', 'mysql', 'sqlite']

  const inspect = async () => {
    const result = await onInspect(url)
    if (!result) return
    setInspection(result)
    setSelected(result.tables.map((table) => table.source_name))
  }

  const toggle = (name) =>
    setSelected((current) =>
      current.includes(name) ? current.filter((item) => item !== name) : [...current, name],
    )

  return (
    <div className="space-y-3">
      <label className="block text-xs">
        <span style={{ color: 'var(--text-secondary)' }}>Connection string</span>
        <input
          className="mt-1 w-full font-mono text-xs"
          placeholder="postgresql://user:password@host:5432/database"
          value={url}
          spellCheck={false}
          onChange={(event) => {
            setUrl(event.target.value)
            setInspection(null)
          }}
        />
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={busy || !url.trim()} onClick={inspect}>
          Inspect
        </Button>
        <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Supported: {backends.join(', ')}. Nothing is read until you choose the tables, and the
          connection is used read-only — no credentials are stored.
        </span>
      </div>

      {inspection && (
        <div className="rounded-md border" style={{ borderColor: 'var(--border)' }}>
          <div
            className="flex flex-wrap items-center justify-between gap-2 border-b px-3 py-2"
            style={{ borderColor: 'var(--border)' }}
          >
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <StatusBadge status="good">{inspection.dialect}</StatusBadge>
              <span className="font-mono text-[11px]" style={{ color: 'var(--text-muted)' }}>
                {inspection.display_url}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={() => setSelected(inspection.tables.map((t) => t.source_name))}>
                Select all
              </Button>
              <Button
                variant="primary"
                disabled={busy || !selected.length}
                onClick={() => onConnect(url, selected)}
              >
                Ingest {selected.length} table{selected.length === 1 ? '' : 's'}
              </Button>
            </div>
          </div>

          {inspection.tables.length === 0 ? (
            <p className="px-3 py-4 text-sm" style={{ color: 'var(--text-muted)' }}>
              This database has no readable tables.
            </p>
          ) : (
            <ul className="max-h-80 divide-y overflow-y-auto" style={{ borderColor: 'var(--border)' }}>
              {inspection.tables.map((table) => (
                <li key={`${table.schema ?? ''}.${table.source_name}`} className="px-3 py-2">
                  <label className="flex cursor-pointer items-start gap-2">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={selected.includes(table.source_name)}
                      onChange={() => toggle(table.source_name)}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium">{table.name}</span>
                        {table.schema && <Tag muted>{table.schema}</Tag>}
                        <span className="text-[11px] tabular" style={{ color: 'var(--text-muted)' }}>
                          {formatRows(table.row_count)} · {table.columns.length} columns
                        </span>
                      </span>
                      <span
                        className="mt-0.5 block text-[11px]"
                        style={{ color: 'var(--text-secondary)' }}
                      >
                        {table.primary_key.length > 0 && `key: ${table.primary_key.join(', ')}`}
                        {table.primary_key.length > 0 && table.foreign_keys.length > 0 && ' · '}
                        {table.foreign_keys.length > 0 &&
                          `${table.foreign_keys.length} declared relationship${
                            table.foreign_keys.length === 1 ? '' : 's'
                          }`}
                        {table.primary_key.length === 0 &&
                          table.foreign_keys.length === 0 &&
                          'no declared keys'}
                      </span>
                    </span>
                  </label>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

export function SourcePicker({ vocabulary, session, busy, onUpload, onInspect, onConnect }) {
  const [tab, setTab] = useState('file')

  return (
    <Panel
      title="Add your data"
      subtitle="Excel workbooks, CSVs, SQLite files and SQL dumps, or a direct connection to a PostgreSQL or MySQL database."
      actions={
        <div className="flex gap-1">
          {[
            { key: 'file', label: 'Upload a file' },
            { key: 'database', label: 'Connect a database' },
          ].map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => setTab(item.key)}
              className="focus-ring rounded-md px-2.5 py-1 text-xs font-medium"
              style={{
                background: tab === item.key ? 'var(--surface-3)' : 'transparent',
                color: tab === item.key ? 'var(--text-primary)' : 'var(--text-secondary)',
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      }
    >
      {tab === 'file' ? (
        <FileTab vocabulary={vocabulary} session={session} busy={busy} onUpload={onUpload} />
      ) : (
        <DatabaseTab
          vocabulary={vocabulary}
          busy={busy}
          onInspect={onInspect}
          onConnect={onConnect}
        />
      )}
    </Panel>
  )
}
