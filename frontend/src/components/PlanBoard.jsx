/** Phase 2 view — co-planning.
 *
 *  The Cocoa pattern: the agent proposes, the human edits. The proposal is
 *  grouped by what it does, in plain words, so a person who has never heard of
 *  a "type cast" can still tell what is being offered and why. Steps that can
 *  only remove nothing (trim stray spaces, drop a column with no values in it)
 *  are ticked; steps that change how a value is written are offered but left
 *  unticked. Each can be edited, removed, or added from scratch — the plan that
 *  runs is the plan the user approved, not the one that was proposed. Nothing
 *  executes until "Run" is pressed.
 */

import { useEffect, useState } from 'react'
import { Button, Panel, Tag } from './ui'

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
  rename_column: 'Rename a column',
  drop_column: 'Remove an empty column',
  standardize_format: 'Write values in one format',
  standardize_casing: 'Make capitalisation consistent',
  strip_whitespace: 'Trim extra spaces',
  merge_sheets: 'Combine two tables',
  split_column: 'Split a column in two',
  deduplicate: 'Remove repeated rows',
  type_cast: 'Change the kind of data',
  add_synthetic_key: 'Add a row number',
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

/** What each kind of step does, for someone who has never cleaned data.
 *
 *  ``safe`` steps change nothing a person typed: they remove blanks, extra
 *  spaces or exact repeats. The rest rewrite values or restructure tables, so
 *  they are offered but left for the person to tick.
 */
const GROUPS = [
  {
    types: ['drop_column'],
    title: 'Remove empty columns',
    blurb: 'Columns with nothing in any row. Removing them loses no data.',
    safe: true,
  },
  {
    types: ['strip_whitespace'],
    title: 'Trim extra spaces',
    blurb:
      'Values with spaces at the start or end. They look the same but a database treats them as different.',
    safe: true,
  },
  {
    types: ['deduplicate'],
    title: 'Remove repeated rows',
    blurb: 'Rows that are exact copies of another row. One copy is kept.',
    safe: true,
  },
  {
    types: ['add_synthetic_key'],
    title: 'Number the rows',
    blurb: 'Adds a counter column to a table that has no way to tell one row from another.',
    safe: true,
  },
  {
    types: ['standardize_casing'],
    title: 'Make capitalisation consistent',
    blurb:
      'The same value written with different capital letters. This changes how some values are spelled, so check the examples before you tick them.',
    safe: false,
  },
  {
    types: ['standardize_format'],
    title: 'Write values in one format',
    blurb: 'Rewrites values (dates, text) into a single style.',
    safe: false,
  },
  {
    types: ['type_cast'],
    title: 'Change the kind of data',
    blurb: 'Treats a column as numbers, dates or text. Values that do not fit become blank.',
    safe: false,
  },
  {
    types: ['rename_column', 'split_column'],
    title: 'Rename and split columns',
    blurb: 'Changes column names or splits one column into several.',
    safe: false,
  },
  {
    types: ['merge_sheets'],
    title: 'Combine tables that were split in two',
    blurb:
      'Two tables with exactly the same columns look like one table that was split. Combining them is the only step that joins tables, so look before you tick it.',
    safe: false,
  },
]

function groupOf(step) {
  return GROUPS.find((group) => group.types.includes(step.type))
}

function StepRow({ step, included, onToggle, onChange, onDelete, editable }) {
  const [open, setOpen] = useState(false)
  const fields = STEP_FIELDS[step.type] ?? []

  return (
    <li className="py-2.5">
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          className="mt-1 shrink-0"
          checked={included}
          disabled={!editable}
          onChange={onToggle}
          aria-label={`Include: ${step.description}`}
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Tag>{step.table}</Tag>
            {step.params?.column && <Tag muted>{step.params.column}</Tag>}
            {step.origin === 'user' && <Tag muted>added by you</Tag>}
          </div>
          <p className="mt-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
            {step.description}
          </p>
          {step.rationale && (
            <p className="mt-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
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

        <div className="flex shrink-0 flex-col items-end gap-1 text-[11px]">
          <button
            type="button"
            className="focus-ring underline"
            style={{ color: 'var(--series-1)' }}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? 'hide settings' : 'settings'}
          </button>
          {editable && step.origin === 'user' && (
            <button
              type="button"
              className="focus-ring underline"
              style={{ color: 'var(--status-critical)' }}
              onClick={onDelete}
            >
              remove
            </button>
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
        <span style={{ color: 'var(--text-muted)' }}>Add a step of your own</span>
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
            description: `${STEP_TITLES[type] ?? type} — added by you. Open “settings” to say which column.`,
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

function sourceNote(planSource) {
  switch (planSource) {
    case 'claude':
      return 'Suggested by the AI assistant from summaries of your data (never the rows themselves), then checked against your tables.'
    case 'heuristic_after_llm_error':
      return "The AI assistant couldn't be used just now, so these suggestions come from checking your data for common problems."
    case 'heuristic_after_llm_rejected':
      return "The AI assistant's suggestions didn't fit your data, so these come from checking it for common problems."
    default:
      return 'Found by checking your data for common problems.'
  }
}

export function PlanBoard({ plan, planSource, rejections, tables, stepTypes, onSubmit, busy }) {
  const [draft, setDraft] = useState(plan)
  const [excluded, setExcluded] = useState(() => initiallyExcluded(plan))
  const [reviewEach, setReviewEach] = useState(false)

  useEffect(() => {
    setDraft(plan)
    setExcluded(initiallyExcluded(plan))
  }, [plan])

  const included = draft.filter((step) => !excluded.has(step.id))
  const toggle = (id) =>
    setExcluded((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  const setGroup = (steps, on) =>
    setExcluded((current) => {
      const next = new Set(current)
      steps.forEach((step) => (on ? next.delete(step.id) : next.add(step.id)))
      return next
    })

  const grouped = GROUPS.map((group) => ({
    group,
    steps: draft.filter((step) => groupOf(step) === group),
  })).filter((entry) => entry.steps.length)
  const ungrouped = draft.filter((step) => !groupOf(step))
  const heldBack = draft.filter((step) => excluded.has(step.id) && groupOf(step)?.safe === false)

  const llmFailed = planSource === 'heuristic_after_llm_error'

  return (
    <Panel
      title="Tidy-up suggestions"
      subtitle={sourceNote(planSource)}
      actions={
        <Button disabled={busy} onClick={() => onSubmit('cancel', null)}>
          Skip cleaning
        </Button>
      }
    >
      <div className="space-y-4">
        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          We checked every column for common problems. Nothing here fills in a blank cell, and your
          uploaded file is never touched — changes are made to a working copy. Tick what you want
          done, then press the button at the bottom.
        </p>

        {(llmFailed || rejections?.length > 0) && (
          <div
            className="rounded-md border px-3 py-2 text-xs"
            style={{ borderColor: 'var(--status-warning)' }}
          >
            {llmFailed ? (
              <p>The AI assistant is unavailable right now (it may have reached its usage limit).</p>
            ) : (
              <p className="font-medium">
                {rejections.length} suggestion{rejections.length === 1 ? ' was' : 's were'} left out
                because {rejections.length === 1 ? 'it' : 'they'} didn&apos;t fit your data.
              </p>
            )}
            {rejections?.length > 0 && (
              <details className="mt-1" style={{ color: 'var(--text-muted)' }}>
                <summary className="cursor-pointer">Technical details</summary>
                <ul className="mt-1 space-y-0.5">
                  {rejections.map((reason) => (
                    <li key={reason}>· {reason}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}

        {draft.length > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span style={{ color: 'var(--text-secondary)' }}>
              {included.length} of {draft.length} selected
              {heldBack.length > 0 &&
                ` · ${heldBack.length} that change how values are written ${
                  heldBack.length === 1 ? 'is' : 'are'
                } switched off until you tick ${heldBack.length === 1 ? 'it' : 'them'}`}
            </span>
            <span className="flex gap-3">
              <button
                type="button"
                className="focus-ring underline"
                style={{ color: 'var(--series-1)' }}
                onClick={() => setGroup(draft, true)}
              >
                select all
              </button>
              <button
                type="button"
                className="focus-ring underline"
                style={{ color: 'var(--series-1)' }}
                onClick={() => setGroup(draft, false)}
              >
                select none
              </button>
            </span>
          </div>
        )}

        {grouped.map(({ group, steps }) => {
          const on = steps.filter((step) => !excluded.has(step.id)).length
          return (
            <details key={group.title} open={!group.safe || steps.length <= 3} className="panel">
              <summary className="flex cursor-pointer flex-wrap items-center justify-between gap-2 px-3 py-2.5">
                <span>
                  <span className="text-sm font-semibold">{group.title}</span>
                  <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                    {on} of {steps.length} selected
                  </span>
                  {!group.safe && (
                    <span className="ml-2">
                      <Tag muted>take a look</Tag>
                    </span>
                  )}
                </span>
                <span className="flex gap-3 text-[11px]">
                  <button
                    type="button"
                    className="focus-ring underline"
                    style={{ color: 'var(--series-1)' }}
                    onClick={(event) => {
                      // Inside a <summary>, a click would also fold the group.
                      event.preventDefault()
                      setGroup(steps, on !== steps.length)
                    }}
                  >
                    {on === steps.length ? 'untick all' : 'tick all'}
                  </button>
                </span>
              </summary>
              <div className="border-t px-3 pb-1 pt-2" style={{ borderColor: 'var(--border)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {group.blurb}
                </p>
                <ul className="divide-y" style={{ borderColor: 'var(--gridline)' }}>
                  {steps.map((step) => (
                    <StepRow
                      key={step.id}
                      step={step}
                      editable
                      included={!excluded.has(step.id)}
                      onToggle={() => toggle(step.id)}
                      onChange={(params) =>
                        setDraft(draft.map((item) => (item.id === step.id ? { ...item, params } : item)))
                      }
                      onDelete={() => setDraft(draft.filter((item) => item.id !== step.id))}
                    />
                  ))}
                </ul>
              </div>
            </details>
          )
        })}

        {ungrouped.length > 0 && (
          <details open className="panel">
            <summary className="cursor-pointer px-3 py-2.5 text-sm font-semibold">
              Other steps
            </summary>
            <ul className="divide-y px-3" style={{ borderColor: 'var(--gridline)' }}>
              {ungrouped.map((step) => (
                <StepRow
                  key={step.id}
                  step={step}
                  editable
                  included={!excluded.has(step.id)}
                  onToggle={() => toggle(step.id)}
                  onChange={(params) =>
                    setDraft(draft.map((item) => (item.id === step.id ? { ...item, params } : item)))
                  }
                  onDelete={() => setDraft(draft.filter((item) => item.id !== step.id))}
                />
              ))}
            </ul>
          </details>
        )}

        {draft.length === 0 && (
          <p className="py-4 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
            Your data looks tidy — nothing needs fixing.
          </p>
        )}

        <details>
          <summary className="cursor-pointer text-xs" style={{ color: 'var(--text-muted)' }}>
            Add a step of your own
          </summary>
          <div className="mt-2">
            <AddStep
              tables={tables}
              stepTypes={stepTypes}
              onAdd={(step) => setDraft([...draft, step])}
            />
          </div>
        </details>

        <div
          className="flex flex-wrap items-center justify-between gap-3 border-t pt-3"
          style={{ borderColor: 'var(--border)' }}
        >
          <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
            <input
              type="checkbox"
              checked={reviewEach}
              onChange={(event) => setReviewEach(event.target.checked)}
            />
            Pause after each step so I can check it
          </label>
          <Button
            variant="primary"
            disabled={busy}
            onClick={() => onSubmit('confirm', included, !reviewEach)}
          >
            {draft.length === 0
              ? 'Continue'
              : included.length === 0
                ? 'Continue without changes'
                : `Apply ${included.length} step${included.length === 1 ? '' : 's'}`}
          </Button>
        </div>
      </div>
    </Panel>
  )
}

/** Only steps that cannot lose a value start ticked. */
function initiallyExcluded(plan) {
  return new Set(plan.filter((step) => groupOf(step)?.safe === false).map((step) => step.id))
}

export { STEP_FIELDS, STEP_TITLES, toInput, fromInput }
