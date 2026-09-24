/** The account's dataset history — every session it has ever created, in one
 *  place to jump back into, the way a chat product lists past conversations.
 *  Before this existed the only way back to an older dataset was to still
 *  have its id in this browser's local storage. */

import { useState } from 'react'
import { Button } from './ui'

function formatDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function sessionSubtitle(session) {
  const parts = [formatDate(session.created_at)]
  const stats = session.stats ?? {}
  if (stats.sheet_count) {
    parts.push(`${stats.sheet_count} table${stats.sheet_count === 1 ? '' : 's'}`)
  }
  if (stats.total_rows) {
    parts.push(`${stats.total_rows.toLocaleString()} rows`)
  }
  return parts.filter(Boolean).join(' · ')
}

export function SessionSwitcher({ sessions, activeId, busy, onOpen, onSelect, onDelete }) {
  const [open, setOpen] = useState(false)
  const [pendingDelete, setPendingDelete] = useState(null)

  const toggle = () => {
    const next = !open
    setOpen(next)
    setPendingDelete(null)
    if (next) onOpen?.()
  }

  return (
    <div className="relative">
      <Button variant="ghost" onClick={toggle} aria-expanded={open} disabled={busy}>
        Your databases{sessions?.length ? ` (${sessions.length})` : ''} ▾
      </Button>

      {open && (
        <div
          className="absolute left-0 z-20 mt-1 max-h-96 w-80 overflow-y-auto rounded-md border shadow-lg"
          style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
        >
          {!sessions?.length ? (
            <p className="p-4 text-xs" style={{ color: 'var(--text-muted)' }}>
              Nothing yet — upload a file or connect a database to get started.
            </p>
          ) : (
            <ul>
              {sessions.map((session) => {
                const active = session.id === activeId
                return (
                  <li
                    key={session.id}
                    className="group flex items-start gap-2 border-b px-3 py-2.5 last:border-b-0"
                    style={{
                      borderColor: 'var(--border)',
                      background: active ? 'var(--surface-3)' : undefined,
                    }}
                  >
                    <button
                      type="button"
                      className="focus-ring min-w-0 flex-1 text-left"
                      onClick={() => {
                        setOpen(false)
                        if (!active) onSelect(session.id)
                      }}
                    >
                      <p className="truncate text-sm font-medium">
                        {session.name}
                        {active && (
                          <span className="ml-1.5 text-[11px] font-normal" style={{ color: 'var(--text-muted)' }}>
                            (open)
                          </span>
                        )}
                      </p>
                      <p className="mt-0.5 truncate text-[11px]" style={{ color: 'var(--text-muted)' }}>
                        {sessionSubtitle(session)}
                      </p>
                    </button>

                    {pendingDelete === session.id ? (
                      <div className="flex shrink-0 items-center gap-1.5 pt-0.5">
                        <button
                          type="button"
                          className="focus-ring text-[11px] font-medium"
                          style={{ color: 'var(--status-critical)' }}
                          onClick={() => {
                            setPendingDelete(null)
                            onDelete(session.id)
                          }}
                        >
                          Delete
                        </button>
                        <button
                          type="button"
                          className="focus-ring text-[11px]"
                          style={{ color: 'var(--text-muted)' }}
                          onClick={() => setPendingDelete(null)}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        title="Delete this dataset"
                        aria-label={`Delete ${session.name}`}
                        className="focus-ring shrink-0 rounded px-1.5 py-0.5 text-xs opacity-0 transition group-hover:opacity-100"
                        style={{ color: 'var(--text-muted)' }}
                        onClick={() => setPendingDelete(session.id)}
                      >
                        ✕
                      </button>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
