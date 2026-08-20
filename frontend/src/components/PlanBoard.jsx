/** Phase 2 view — co-planning.
 *
 *  The Cocoa pattern: the agent proposes, the human edits. Cards can be
 *  reordered by dragging, re-parameterised inline, deleted, or added from
 *  scratch — the plan that runs is the plan the user approved, not the one
 *  that was proposed. Nothing executes until "Run this plan" is pressed.
 */

import { useEffect, useState } from 'react'
import { Button, Panel, StatusBadge, Tag } from './ui'

const STEP_FIELDS = {
  rename_column: [
    { key: 'column', label: 'Column' },
    { key: 'new_name', label: 'New name' },
  ],
  drop_column: [{ key: 'column', label: 'Column' }],
  standardize_format: [
    { key: 'column', label: 'Column' },
    { key: 'format', label: 'Format', options: ['upper', 'lower', 'title', 'strip', 'date'] },
    { key: 'date_format', label: 'Date format' },
  ],
  standardize_casing: [
    { key: 'column', label: 'Column' },
    { key: 'casing', label: 'Match casing', options: ['lower', 'upper', 'title'] },
  ],
  strip_whitespace: [
    { key: 'column', label: 'Column' },
    { key: 'collapse_internal', label: 'Also collapse repeated inner spaces', options: ['false', 'true'] },
  ],
  merge_sheets: [
    { key: 'left_table', label: 'Left table' },
    { key: 'left_key', label: 'Left key' },
    { key: 'right_table', label: 'Right table' },
    { key: 'right_key', label: 'Right key' },
    { key: 'how', label: 'Join', options: ['inner', 'left', 'right', 'outer'] },
    { key: 'result_table', label: 'Result table' },
  ],
  split_column: [
    { key: 'column', label: 'Column' },
    { key: 'delimiter', label: 'Delimiter' },
    { key: 'into', label: 'New columns (comma separated)' },
  ],
  deduplicate: [{ key: 'subset', label: 'Only these columns (comma separated, blank = all)' }],
  type_cast: [
    { key: 'column', label: 'Column' },
    { key: 'to', label: 'Convert to', options: ['numeric', 'datetime', 'string', 'boolean'] },
  ],
  add_synthetic_key: [{ key: 'column', label: 'New key column' }],
}

const STEP_TITLES = {
  rename_column: 'Rename column',
  drop_column: 'Drop column',
  standardize_format: 'Standardise format',
  standardize_casing: 'Merge capitalisation variants',
  strip_whitespace: 'Trim stray spaces',
  merge_sheets: 'Merge sheets',
  split_column: 'Split column',
  deduplicate: 'Remove duplicate rows',
  type_cast: 'Convert data type',
  add_synthetic_key: 'Add row id',
}

function toInput(value) {
  if (value === null || value === undefined) return ''
  if (Array.isArray(value)) return value.join(', ')
  return String(value)
}

function fromInput(key, raw) {
  if (key === 'into' || key === 'subset') {
    const parts = raw
      .split(',')
      .map((part) => part.trim())
      .filter(Boolean)
    return parts.length ? parts : null
  }
  if (raw === '') return null
  // Send a real boolean: the string "false" is truthy on the Python side.
  if (key === 'collapse_internal') return raw === 'true'
  if (key === 'value' && raw !== '' && !Number.isNaN(Number(raw))) return Number(raw)
  return raw
}

function StepCard({ step, index, total, editable, onChange, onDelete, onMove, dragHandlers }) {
  const [open, setOpen] = useState(false)
  const fields = STEP_FIELDS[step.type] ?? []

  return (
    <li
      draggable={editable}
      {...(editable ? dragHandlers : {})}
      className="panel p-3"
      style={{ cursor: editable ? 'grab' : 'default' }}
    >
      <div className="flex items-start gap-3">
        <span
          className="tabular mt-0.5 w-6 shrink-0 text-center text-xs"
          style={{ color: 'var(--text-muted)' }}
          aria-hidden="true"
        >
          {editable ? '⠿' : index + 1}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="text-sm font-semibold">{STEP_TITLES[step.type] ?? step.type}</h4>
            <Tag>{step.table}</Tag>
            <Tag muted title={step.origin === 'agent' ? 'proposed by Claude' : `added by ${step.origin}`}>
              {step.origin}
            </Tag>
            {step.status !== 'pending' && (
              <StatusBadge
                status={
                  step.status === 'approved'
                    ? 'good'
                    : step.status === 'failed'
                      ? 'critical'
                      : step.status === 'reverted'
                        ? 'serious'
                        : 'neutral'
                }
              >
                {step.status}
              </StatusBadge>
            )}
          </div>

          <p className="mt-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
            {step.description}
          </p>

          {step.rationale && (
            <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
              ↳ {step.rationale}
            </p>
          )}

          {open && (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {fields.map((field) => (
                <label key={field.key} className="text-[11px]">
                  <span style={{ color: 'var(--text-muted)' }}>{field.label}</span>
                  {field.options ? (
                    <select
                      className="mt-0.5 w-full text-xs"
                      disabled={!editable}
                      value={toInput(step.params[field.key])}
                      onChange={(event) =>
                        onChange({ ...step.params, [field.key]: event.target.value })
                      }
                    >
                      <option value="">—</option>
                      {field.options.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      className="mt-0.5 w-full text-xs"
                      disabled={!editable}
                      value={toInput(step.params[field.key])}
                      onChange={(event) =>
                        onChange({
                          ...step.params,
                          [field.key]: fromInput(field.key, event.target.value),
                        })
                      }
                    />
                  )}
                </label>
              ))}
            </div>
          )}
        </div>

        <div className="flex shrink-0 flex-col items-end gap-1">
          <button
            type="button"
            className="focus-ring text-[11px] underline"
            style={{ color: 'var(--series-1)' }}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? 'hide' : 'edit'}
          </button>
          {editable && (
            <>
              <div className="flex gap-1">
                <button
                  type="button"
                  aria-label="Move step up"
                  disabled={index === 0}
                  onClick={() => onMove(index, index - 1)}
                  className="focus-ring rounded px-1 text-xs disabled:opacity-30"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  ↑
                </button>
                <button
                  type="button"
                  aria-label="Move step down"
                  disabled={index === total - 1}
                  onClick={() => onMove(index, index + 1)}
                  className="focus-ring rounded px-1 text-xs disabled:opacity-30"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  ↓
                </button>
              </div>
              <button
                type="button"
                className="focus-ring text-[11px] underline"
                style={{ color: 'var(--status-critical)' }}
                onClick={onDelete}
              >
                remove
              </button>
            </>
          )}
        </div>
      </div>
    </li>
  )
}

function AddStep({ tables, stepTypes, onAdd }) {
  const [type, setType] = useState('drop_column')
  const [table, setTable] = useState(tables[0] ?? '')

  useEffect(() => {
    if (!table && tables.length) setTable(tables[0])
  }, [tables, table])

  return (
    <div
      className="flex flex-wrap items-end gap-2 rounded-md border border-dashed p-3"
      style={{ borderColor: 'var(--baseline)' }}
    >
      <label className="text-[11px]">
        <span style={{ color: 'var(--text-muted)' }}>Add a step</span>
        <select className="mt-0.5 block text-xs" value={type} onChange={(e) => setType(e.target.value)}>
          {stepTypes.map((option) => (
            <option key={option} value={option}>
              {STEP_TITLES[option] ?? option}
            </option>
          ))}
        </select>
      </label>
      <label className="text-[11px]">
        <span style={{ color: 'var(--text-muted)' }}>on table</span>
        <select className="mt-0.5 block text-xs" value={table} onChange={(e) => setTable(e.target.value)}>
          {tables.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
      <Button
        onClick={() =>
          onAdd({
            id: `user-${Math.random().toString(36).slice(2, 10)}`,
            type,
            table,
            description: 'Added by you.',
            params: {},
            origin: 'user',
            status: 'pending',
          })
        }
      >
        Add
      </Button>
    </div>
  )
}

export function PlanBoard({ plan, planSource, rejections, tables, stepTypes, onSubmit, busy }) {
  const [draft, setDraft] = useState(plan)
  const [dragIndex, setDragIndex] = useState(null)

  useEffect(() => setDraft(plan), [plan])

  const move = (from, to) => {
    if (to < 0 || to >= draft.length) return
    const next = [...draft]
    const [item] = next.splice(from, 1)
    next.splice(to, 0, item)
    setDraft(next)
  }

  return (
    <Panel
      title="Review the cleaning plan"
      subtitle={
        planSource === 'claude'
          ? ''
          : ''
      }
      actions={
        <div className="flex gap-2">
          <Button disabled={busy} onClick={() => onSubmit('cancel', null)}>
            Cancel
          </Button>
          <Button variant="primary" disabled={busy || !draft.length} onClick={() => onSubmit('confirm', draft)}>
            Run this plan ({draft.length})
          </Button>
        </div>
      }
    >
      {rejections?.length > 0 && (
        <div
          className="mb-3 rounded-md border px-3 py-2 text-xs"
          style={{ borderColor: 'var(--status-warning)' }}
        >
          <p className="font-medium">Some proposed steps were rejected before you saw them:</p>
          <ul className="mt-1 space-y-0.5" style={{ color: 'var(--text-secondary)' }}>
            {rejections.map((reason) => (
              <li key={reason}>· {reason}</li>
            ))}
          </ul>
        </div>
      )}

      <ol className="space-y-2">
        {draft.map((step, index) => (
          <StepCard
            key={step.id}
            step={step}
            index={index}
            total={draft.length}
            editable
            onChange={(params) =>
              setDraft(draft.map((item, i) => (i === index ? { ...item, params } : item)))
            }
            onDelete={() => setDraft(draft.filter((_, i) => i !== index))}
            onMove={move}
            dragHandlers={{
              onDragStart: () => setDragIndex(index),
              onDragOver: (event) => event.preventDefault(),
              onDrop: () => {
                if (dragIndex !== null && dragIndex !== index) move(dragIndex, index)
                setDragIndex(null)
              },
              onDragEnd: () => setDragIndex(null),
            }}
          />
        ))}
      </ol>

      {draft.length === 0 && (
        <p className="py-4 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
          The plan is empty — nothing will be changed.
        </p>
      )}

      <div className="mt-3">
        <AddStep tables={tables} stepTypes={stepTypes} onAdd={(step) => setDraft([...draft, step])} />
      </div>

      <p className="mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        No step on this list can write a value into an empty cell. Missing values stay missing —
        they arrive in the database as NULL, so totals and averages are computed from the rows you
        actually have, not from filled-in guesses.
      </p>
    </Panel>
  )
}

export { STEP_FIELDS, STEP_TITLES, toInput, fromInput }
