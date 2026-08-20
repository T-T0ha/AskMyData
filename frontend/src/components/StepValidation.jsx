/** Phase 2 view — co-execution.
 *
 *  After every executed step the graph pauses here. The user sees exactly what
 *  changed — the affected rows before and after, side by side — and decides:
 *  approve, revert, or retry with different parameters. Nothing moves on
 *  without that decision.
 */

import { useState } from 'react'
import { STEP_FIELDS, STEP_TITLES, fromInput, toInput } from './PlanBoard'
import { Button, DataTable, Panel, StatusBadge, Tag } from './ui'

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
      <p className="text-xs font-medium">Change the parameters and run this step again</p>
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
          Undo and retry
        </Button>
      </div>
    </div>
  )
}

export function StepValidation({ pending, progress, onDecide, busy }) {
  const [retrying, setRetrying] = useState(false)
  const step = pending.step ?? {}

  if (pending.kind === 'step_failed') {
    return (
      <Panel
        title={`Step ${pending.cursor + 1} could not run`}
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
          <StatusBadge status="critical">error</StatusBadge>
          <p className="mt-1" style={{ color: 'var(--text-secondary)' }}>
            {pending.error}
          </p>
        </div>
        <p className="mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
          Your data was not changed. Fix the parameters and retry, or skip the step and continue.
        </p>
        <RetryForm step={step} busy={busy} onRetry={(params) => onDecide('retry', params)} />
      </Panel>
    )
  }

  const outcome = pending.outcome ?? {}
  const rowsRemoved = outcome.rows_removed ?? 0

  return (
    <Panel
      title={`Step ${pending.cursor + 1} of ${progress.total} — check the result`}
      subtitle={outcome.summary}
      actions={
        <div className="flex gap-2">
          <Button disabled={busy} onClick={() => setRetrying((value) => !value)}>
            Retry with changes
          </Button>
          <Button disabled={busy} onClick={() => onDecide('revert')}>
            Revert
          </Button>
          <Button variant="primary" disabled={busy} onClick={() => onDecide('approve')}>
            Approve
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

      <p className="mt-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
        {step.description}
      </p>

      <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Delta label="Rows before" value={(outcome.rows_before ?? 0).toLocaleString()} />
        <Delta label="Rows after" value={(outcome.rows_after ?? 0).toLocaleString()} />
        <Delta
          label="Rows removed"
          value={rowsRemoved.toLocaleString()}
          tone={rowsRemoved > 0 ? 'warn' : undefined}
        />
        <Delta label="Cells changed" value={(outcome.cells_changed ?? 0).toLocaleString()} />
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
            Before — affected rows
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

export function CleaningSummary({ state, onRestart }) {
  const history = (state.history ?? []).filter((entry) => entry.action !== 'finalized')
  return (
    <Panel
      title="Cleaning complete"
      subtitle={`Session ${state.status}. Every change below was approved by you.`}
      actions={<Button onClick={onRestart}>Start a new cleaning run</Button>}
    >
      <ol className="space-y-1.5 text-xs">
        {history.map((entry, index) => (
          <li key={index} className="flex items-start gap-2">
            <StatusBadge
              status={
                entry.action === 'approved' ? 'good' : entry.action === 'skipped' ? 'neutral' : 'serious'
              }
            >
              {entry.action}
            </StatusBadge>
            <span style={{ color: 'var(--text-secondary)' }}>{entry.summary ?? `step ${entry.cursor + 1}`}</span>
          </li>
        ))}
        {history.length === 0 && (
          <li style={{ color: 'var(--text-muted)' }}>No steps were executed.</li>
        )}
      </ol>

      <div className="mt-4">
        <p className="mb-1 text-[11px] font-medium" style={{ color: 'var(--text-muted)' }}>
          Resulting tables
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
