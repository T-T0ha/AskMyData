/** The pinned dashboard — Phase 6.
 *
 *  A card stores a question and the SQL that answered it, never the answer
 *  itself: every render here is a live re-execution the backend just did
 *  (`GET /dashboard/cards`), so "last refreshed" is always "this request",
 *  not a snapshot slowly going stale.  The chart renderer is the same one
 *  "Ask questions" uses (`./charts`) — a card looks exactly like the answer
 *  it was pinned from.
 *
 *  Layout is react-grid-layout over a 12-column grid.  A card with no stored
 *  position yet (never dragged or resized) gets a deterministic three-across
 *  default computed from its list index, so the grid is never empty-looking
 *  on first pin; dragging or resizing persists a real position via
 *  `onLayoutChange`, fired only on drop/resize-stop rather than every
 *  intermediate frame.
 */

import { useEffect, useRef, useState } from 'react'
import GridLayout, { useContainerWidth } from 'react-grid-layout'
import 'react-grid-layout/css/styles.css'
import 'react-resizable/css/styles.css'
import { Chart, toCsv } from './charts'
import { Button, DataTable, EmptyState, Panel } from './ui'

const COLS = 12
const ROW_HEIGHT = 90
const DEFAULT_W = 4
const DEFAULT_H = 4
//: Live re-poll while the dashboard tab is open — "results always current",
//: not only on load.  Long enough that a dozen cards re-executing does not
//: hammer the database; short enough to feel live.
const AUTO_REFRESH_MS = 60_000

function defaultLayout(card, index) {
  const stored = card.layout || {}
  if (stored.w) return { i: card.id, x: stored.x ?? 0, y: stored.y ?? 0, w: stored.w, h: stored.h ?? DEFAULT_H }
  const perRow = Math.floor(COLS / DEFAULT_W)
  return {
    i: card.id,
    x: (index % perRow) * DEFAULT_W,
    y: Math.floor(index / perRow) * DEFAULT_H,
    w: DEFAULT_W,
    h: DEFAULT_H,
  }
}

function relativeTime(iso) {
  if (!iso) return ''
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000))
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ago`
}

function DashboardCard({ card, onRename, onUnpin }) {
  const [view, setView] = useState('chart')
  const [showSql, setShowSql] = useState(false)
  const [editingTitle, setEditingTitle] = useState(false)
  const [title, setTitle] = useState(card.title)

  useEffect(() => setTitle(card.title), [card.title])

  const commitTitle = () => {
    setEditingTitle(false)
    const trimmed = title.trim()
    if (trimmed && trimmed !== card.title) onRename(card.id, trimmed)
    else setTitle(card.title)
  }

  const copyData = () => {
    if (!card.ok) return
    navigator.clipboard?.writeText(toCsv(card.columns, card.rows)).catch(() => {})
  }

  return (
    <div className="panel flex h-full flex-col overflow-hidden">
      <header
        className="card-drag-handle flex cursor-grab items-start justify-between gap-2 border-b px-3 py-2 active:cursor-grabbing"
        style={{ borderColor: 'var(--border)' }}
      >
        <div className="min-w-0 flex-1">
          {editingTitle ? (
            <input
              type="text"
              autoFocus
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              onBlur={commitTitle}
              onKeyDown={(event) => {
                if (event.key === 'Enter') commitTitle()
                if (event.key === 'Escape') {
                  setTitle(card.title)
                  setEditingTitle(false)
                }
              }}
              className="focus-ring w-full rounded border px-1.5 py-0.5 text-xs font-semibold"
              style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
              onMouseDown={(event) => event.stopPropagation()}
            />
          ) : (
            <h3
              className="truncate text-xs font-semibold"
              title="Double-click to rename"
              onDoubleClick={() => setEditingTitle(true)}
            >
              {card.title}
            </h3>
          )}
          <p className="mt-0.5 truncate text-[10px]" style={{ color: 'var(--text-muted)' }}>
            {card.ok
              ? `${card.row_count.toLocaleString()} row${card.row_count === 1 ? '' : 's'} · refreshed ${relativeTime(card.refreshed_at)}`
              : 'failed to refresh'}
            {card.date_filtered && ' · date filtered'}
          </p>
        </div>
        <button
          type="button"
          title="Remove from dashboard"
          onClick={() => onUnpin(card.id)}
          onMouseDown={(event) => event.stopPropagation()}
          className="focus-ring shrink-0 rounded px-1 text-xs"
          style={{ color: 'var(--text-muted)' }}
        >
          ✕
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-auto px-3 py-2" onMouseDown={(event) => event.stopPropagation()}>
        {!card.ok ? (
          <p className="text-xs" style={{ color: 'var(--status-critical)' }}>
            {card.error}
          </p>
        ) : card.rows.length === 0 ? (
          <EmptyState>No rows.</EmptyState>
        ) : view === 'table' ? (
          <DataTable columns={card.columns} rows={card.rows} />
        ) : (
          <Chart columns={card.columns} rows={card.rows} visualization={card.visualization} />
        )}
        {showSql && (
          <pre
            className="mt-2 overflow-x-auto rounded p-2 text-[10px] leading-relaxed"
            style={{ background: 'var(--surface-3)' }}
          >
            {card.sql}
          </pre>
        )}
      </div>

      {card.ok && (
        <footer
          className="flex items-center gap-1 border-t px-2 py-1.5"
          style={{ borderColor: 'var(--border)' }}
          onMouseDown={(event) => event.stopPropagation()}
        >
          {card.visualization?.chart !== 'table' && (
            <button
              type="button"
              onClick={() => setView((v) => (v === 'chart' ? 'table' : 'chart'))}
              className="focus-ring rounded px-1.5 py-0.5 text-[10px]"
              style={{ color: 'var(--text-secondary)' }}
            >
              {view === 'chart' ? 'Table' : 'Chart'}
            </button>
          )}
          <button
            type="button"
            onClick={copyData}
            className="focus-ring rounded px-1.5 py-0.5 text-[10px]"
            style={{ color: 'var(--text-secondary)' }}
          >
            Copy
          </button>
          <button
            type="button"
            onClick={() => setShowSql((v) => !v)}
            className="focus-ring rounded px-1.5 py-0.5 text-[10px]"
            style={{ color: 'var(--text-secondary)' }}
          >
            {showSql ? 'Hide SQL' : 'SQL'}
          </button>
        </footer>
      )}
    </div>
  )
}

function HistorySidebar({ history, onRerun, busy }) {
  return (
    <Panel title="Question history" subtitle="Last 20 questions this session — click to ask again.">
      {!history?.length ? (
        <EmptyState>Nothing asked yet.</EmptyState>
      ) : (
        <ul className="space-y-1">
          {history.map((entry) => (
            <li key={entry.id}>
              <button
                type="button"
                disabled={busy}
                onClick={() => onRerun(entry.question)}
                className="focus-ring flex w-full items-start gap-1.5 rounded px-1.5 py-1 text-left text-xs hover:bg-[var(--surface-3)] disabled:cursor-not-allowed disabled:opacity-50"
              >
                <span
                  aria-hidden="true"
                  className="mt-0.5 shrink-0"
                  style={{ color: entry.ok ? 'var(--status-good)' : 'var(--status-critical)' }}
                >
                  {entry.ok ? '✓' : '✕'}
                </span>
                <span className="min-w-0 flex-1 truncate" style={{ color: 'var(--text-secondary)' }}>
                  {entry.question}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}

export function DashboardPanel({
  exported,
  cards,
  history,
  busy,
  onRefresh,
  onRename,
  onUnpin,
  onLayoutChange,
  onRerun,
}) {
  // `measureBeforeMount` holds the grid unmounted until the real container
  // width is known — without it, the grid's first paint uses the hook's
  // 1280px placeholder default and (at least in this library version) some
  // items' pixel positions never get recomputed once the real, narrower
  // width (this column sits beside the history sidebar) arrives a tick
  // later, leaving them stranded over the sidebar.
  const { width, containerRef, mounted } = useContainerWidth({ measureBeforeMount: true })
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const rangeRef = useRef({ start: '', end: '' })
  rangeRef.current = { start, end }

  useEffect(() => {
    if (!exported) return
    onRefresh(rangeRef.current)
    const timer = setInterval(() => onRefresh(rangeRef.current), AUTO_REFRESH_MS)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exported])

  const applyRange = () => onRefresh({ start, end })
  const clearRange = () => {
    setStart('')
    setEnd('')
    onRefresh({ start: '', end: '' })
  }

  if (!exported) {
    return (
      <Panel title="Dashboard" subtitle="Pinned questions, always re-run against the live database.">
        <EmptyState>Build the semantic layer first — there is nothing to pin yet.</EmptyState>
      </Panel>
    )
  }

  const layout = cards.map((card, index) => defaultLayout(card, index))

  const settleLayout = (_layout, _oldItem, newItem) => {
    if (!newItem) return
    onLayoutChange(newItem.i, { x: newItem.x, y: newItem.y, w: newItem.w, h: newItem.h })
  }

  return (
    <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1fr_260px]">
      <div className="space-y-4">
        <Panel
          title="Dashboard"
          subtitle="Every card re-runs its query against the live database on load and every minute."
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="date"
                value={start}
                onChange={(event) => setStart(event.target.value)}
                className="focus-ring rounded-md border px-2 py-1 text-xs"
                style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
                aria-label="Filter from date"
              />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                –
              </span>
              <input
                type="date"
                value={end}
                onChange={(event) => setEnd(event.target.value)}
                className="focus-ring rounded-md border px-2 py-1 text-xs"
                style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
                aria-label="Filter to date"
              />
              <Button onClick={applyRange} disabled={busy}>
                Apply
              </Button>
              {(start || end) && <Button onClick={clearRange}>Clear</Button>}
              <Button onClick={() => window.print()}>Export to PDF</Button>
            </div>
          }
        >
          {/* This wrapper stays mounted regardless of card count — it is what
              `containerRef` measures, and a ref that only exists once cards
              have loaded is a ref the width-measuring effect never gets a
              second chance to attach to. */}
          <div ref={containerRef}>
            {cards.length === 0 ? (
              <EmptyState>
                Nothing pinned yet — ask a question and use "Pin to dashboard" on an answer you
                want to keep.
              </EmptyState>
            ) : (
              mounted && (
                <GridLayout
                  layout={layout}
                  width={width}
                  gridConfig={{ cols: COLS, rowHeight: ROW_HEIGHT, margin: [12, 12] }}
                  dragConfig={{ handle: '.card-drag-handle' }}
                  onDragStop={settleLayout}
                  onResizeStop={settleLayout}
                >
                  {cards.map((card) => (
                    <div key={card.id}>
                      <DashboardCard card={card} onRename={onRename} onUnpin={onUnpin} />
                    </div>
                  ))}
                </GridLayout>
              )
            )}
          </div>
        </Panel>
      </div>
      <div className="no-print">
        <HistorySidebar history={history} onRerun={onRerun} busy={busy} />
      </div>
    </div>
  )
}
