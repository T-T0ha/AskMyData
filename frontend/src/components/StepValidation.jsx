/** Phase 2 view — co-execution.
 *
 *  When the user asked to check each step, the graph pauses here after every
 *  one. The user sees what changed in plain words and the affected rows before
 *  and after, side by side, and decides: continue, undo, or retry with different
 *  settings — or approve everything that is left in one click.
 */

import { useState } from 'react'
import { STEP_FIELDS, STEP_TITLES, fromInput, toInput } from './PlanBoard'
import { Button, DataTable, Panel, StatusBadge, Tag } from './ui'

/** One sentence on what a step just did, from the counts it reported. */
function plainOutcome(outcome) {
  const removed = outcome.rows_removed ?? 0
  const changed = outcome.cells_changed ?? 0
  const columns = outcome.affected_columns ?? []
  if (removed > 0) {
    return `${removed.toLocaleString()} row${removed === 1 ? ' was' : 's were'} removed — the table went from ${(outcome.rows_before ?? 0).toLocaleString()} to ${(outcome.rows_after ?? 0).toLocaleString()} rows.`
  }
  if (changed > 0) {
    return `${changed.toLocaleString()} value${changed === 1 ? ' was' : 's were'} rewritten${columns.length ? ` in ${columns.join(', ')}` : ''}. No rows were removed.`
  }
  if (columns.length) {
    return `The table's structure changed (${columns.join(', ')}). No rows were removed and no values were rewritten.`
  }
  return 'Nothing in the table needed to change.'
}

function Delta({ label, value, tone }) {
  return (
    <div>
      <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        {label}
      </p>
      <p
        className="tabular text-lg font-semibold"
        style={{ color: tone === 'warn' ? 'var(--status-serious)' : 'var(--text-primary)' }}
      >
        {value}
      </p>
    </div>
  )
}

function RetryForm({ step, onRetry, busy }) {
  const [params, setParams] = useState(step.params ?? {})
  const fields = STEP_FIELDS[step.type] ?? []

  return (
    <div
      className="mt-3 rounded-md border p-3"
      style={{ borderColor: 'var(--border)', background: 'var(--surface-2)' }}
    >
      <p className="text-xs font-medium">Change this step's settings and run it again</p>
      <div className="mt-2 grid gap-2 sm:grid-cols-2">
        {fields.map((field) => (
          <label key={field.key} className="text-[11px]">
            <span style={{ color: 'var(--text-muted)' }}>{field.label}</span>
            {field.options ? (
              <select
                className="mt-0.5 w-full text-xs"
                value={toInput(params[field.key])}
                onChange={(event) => setParams({ ...params, [field.key]: event.target.value })}
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
                value={toInput(params[field.key])}
                onChange={(event) =>
                  setParams({ ...params, [field.key]: fromInput(field.key, event.target.value) })
                }
              />
            )}
          </label>
        ))}
      </div>
      <div className="mt-3">
        <Button variant="primary" disabled={busy} onClick={() => onRetry(params)}>
          Undo and run again
        </Button>
      </div>
    </div>
  )
}

export function StepValidation({ pending, progress, onDecide, busy }) {
  const [retrying, setRetrying] = useState(false)
  const step = pending.step ?? {}
  const remaining = Math.max(0, progress.total - pending.cursor - 1)

  if (pending.kind === 'step_failed') {
    return (
      <Panel
        title={`Step ${pending.cursor + 1} couldn't be applied`}
        subtitle={STEP_TITLES[step.type] ?? step.type}
        actions={
          <div className="flex gap-2">
            <Button disabled={busy} onClick={() => onDecide('skip')}>
              Skip this step
            </Button>
            <Button variant="danger" disabled={busy} onClick={() => onDecide('abort')}>
              Stop cleaning
            </Button>
          </div>
        }
      >
        <div
          className="rounded-md border px-3 py-2 text-sm"
          style={{ borderColor: 'var(--status-critical)' }}
        >
          <StatusBadge status="critical">problem</StatusBadge>
          <p className="mt-1" style={{ color: 'var(--text-secondary)' }}>
            {pending.error}
          </p>
        </div>
        <p className="mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
          Your data was not changed by this step. You can skip it and carry on with the rest, stop
          here, or change its settings and try again.
        </p>
        <RetryForm step={step} busy={busy} onRetry={(params) => onDecide('retry', params)} />
      </Panel>
    )
  }

  const outcome = pending.outcome ?? {}
  const rowsRemoved = outcome.rows_removed ?? 0

  return (
    <Panel
      title={`Step ${pending.cursor + 1} of ${progress.total}: ${STEP_TITLES[step.type] ?? step.type}`}
      subtitle="This step has been applied. Check the result, then continue."
      actions={
        <div className="flex flex-wrap justify-end gap-2">
          <Button disabled={busy} onClick={() => setRetrying((value) => !value)}>
            Change settings
          </Button>
          <Button disabled={busy} onClick={() => onDecide('revert')}>
            Undo this step
          </Button>
          {remaining > 0 && (
            <Button disabled={busy} onClick={() => onDecide('approve_all')}>
              Approve this and the other {remaining}
            </Button>
          )}
          <Button variant="primary" disabled={busy} onClick={() => onDecide('approve')}>
            Looks good — continue
          </Button>
        </div>
      }
    >
      <div className="flex flex-wrap items-center gap-2">
        <Tag>{step.table}</Tag>
        {(outcome.affected_columns ?? []).map((column) => (
          <Tag key={column} muted>
            {column}
          </Tag>
        ))}
      </div>

      <p className="mt-2 text-sm">{plainOutcome(outcome)}</p>
      <p className="mt-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
        Why: {step.description}
      </p>

      <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Delta label="Rows before" value={(outcome.rows_before ?? 0).toLocaleString()} />
        <Delta label="Rows after" value={(outcome.rows_after ?? 0).toLocaleString()} />
        <Delta
          label="Rows removed"
          value={rowsRemoved.toLocaleString()}
          tone={rowsRemoved > 0 ? 'warn' : undefined}
        />
        <Delta label="Values rewritten" value={(outcome.cells_changed ?? 0).toLocaleString()} />
      </div>

      {outcome.notes?.length > 0 && (
        <ul className="mt-3 space-y-1 text-xs">
          {outcome.notes.map((note) => (
            <li key={note} className="flex items-start gap-2">
              <StatusBadge status="warning">note</StatusBadge>
              <span style={{ color: 'var(--text-secondary)' }}>{note}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-[11px] font-medium" style={{ color: 'var(--text-muted)' }}>
            Before — the rows this touched
          </p>
          <div className="rounded border" style={{ borderColor: 'var(--border)' }}>
            <DataTable
              columns={outcome.preview_columns}
              rows={outcome.before_preview}
              emptyMessage="Nothing to show"
            />
          </div>
        </div>
        <div>
          <p className="mb-1 text-[11px] font-medium" style={{ color: 'var(--text-muted)' }}>
            After
          </p>
          <div className="rounded border" style={{ borderColor: 'var(--border)' }}>
            <DataTable
              columns={outcome.preview_columns}
              rows={outcome.after_preview}
              highlight="positive"
              emptyMessage={rowsRemoved > 0 ? 'These rows were removed' : 'Nothing to show'}
            />
          </div>
        </div>
      </div>

      {retrying && <RetryForm step={step} busy={busy} onRetry={(params) => onDecide('retry', params)} />}
    </Panel>
  )
}

export function CleaningProgress({ plan, cursor }) {
  const done = plan.filter((step) => ['approved', 'skipped', 'reverted'].includes(step.status)).length
  return (
    <div className="panel p-3">
      <div className="flex items-center justify-between text-xs">
        <span style={{ color: 'var(--text-secondary)' }}>
          Step {Math.min(cursor + 1, plan.length)} of {plan.length}
        </span>
        <span className="tabular" style={{ color: 'var(--text-muted)' }}>
          {done} decided
        </span>
      </div>
      <div className="mt-2 flex" style={{ gap: 2 }}>
        {plan.map((step, index) => (
          <div
            key={step.id}
            title={`${index + 1}. ${STEP_TITLES[step.type] ?? step.type} — ${step.status}`}
            className="h-1.5 flex-1 rounded-full"
            style={{
              background:
                step.status === 'approved'
                  ? 'var(--series-1)'
                  : step.status === 'failed'
                    ? 'var(--status-critical)'
                    : step.status === 'reverted' || step.status === 'skipped'
                      ? 'var(--baseline)'
                      : 'var(--surface-3)',
            }}
          />
        ))}
      </div>
    </div>
  )
}

const SUMMARY_WORDS = { approved: 'done', skipped: 'skipped', reverted: 'undone' }

export function CleaningSummary({ state, onRestart }) {
  const history = (state.history ?? []).filter((entry) => entry.action !== 'finalized')
  const done = history.filter((entry) => entry.action === 'approved').length
  const title =
    state.status === 'cancelled'
      ? 'Cleaning skipped'
      : state.status === 'aborted'
        ? 'Cleaning stopped'
        : 'Cleaning finished'
  const subtitle =
    state.status === 'cancelled'
      ? 'No changes were made to your tables.'
      : `${done} change${done === 1 ? ' was' : 's were'} applied to a working copy of your data. Your uploaded file is untouched.`
  return (
    <Panel
      title={title}
      subtitle={subtitle}
      actions={<Button onClick={onRestart}>Check again for problems</Button>}
    >
      <p className="mb-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
        What was done
      </p>
      <ol className="space-y-1.5 text-xs">
        {history.map((entry, index) => (
          <li key={index} className="flex items-start gap-2">
            <StatusBadge
              status={
                entry.action === 'approved' ? 'good' : entry.action === 'skipped' ? 'neutral' : 'serious'
              }
            >
              {SUMMARY_WORDS[entry.action] ?? entry.action}
            </StatusBadge>
            <span style={{ color: 'var(--text-secondary)' }}>{entry.summary ?? `step ${entry.cursor + 1}`}</span>
          </li>
        ))}
        {history.length === 0 && (
          <li style={{ color: 'var(--text-muted)' }}>No changes were needed or made.</li>
        )}
      </ol>

      <div className="mt-4">
        <p className="mb-1 text-[11px] font-medium" style={{ color: 'var(--text-muted)' }}>
          Your tables now
        </p>
        <DataTable
          columns={['name', 'row_count', 'column_count']}
          rows={state.tables ?? []}
          emptyMessage="No tables"
        />
      </div>
    </Panel>
  )
}
