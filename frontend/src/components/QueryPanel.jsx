/** Ask questions — Phase 5, plus the Phase 6 hooks that hang off one answer:
 *  pinning it to the dashboard and picking up where a suggested follow-up
 *  leaves off.
 *
 *  The backend already decided which chart form fits the result
 *  (`visualization.chart`, computed in `app.query.shape` from the columns
 *  actually returned); this panel's job is to render that choice, never to
 *  re-derive it, and to make the same table view available underneath every
 *  chart — a chart form is a claim about what is easiest to read, not the
 *  only accessible way to see the answer.  The chart renderer itself lives in
 *  `./charts` — the pinned dashboard renders the exact same shapes.
 */

import { useEffect, useState } from 'react'
import { Chart, toCsv } from './charts'
import { Button, DataTable, EmptyState, Panel, Tag } from './ui'

/** Rules and suggestions above the ask box — a collapsible list of recorded
 *  business rules plus the "Try asking" cards, shown before any question is
 *  typed (what makes them *proactive*, unlike the post-answer follow-ups
 *  below each answer). */
function BusinessRulesPanel({ rules, busy, onAdd, onDelete }) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')

  const submit = (event) => {
    event.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || busy) return
    onAdd(trimmed)
    setText('')
  }

  return (
    <Panel
      title="Business rules"
      subtitle="Plain-English rules the model follows when answering — recorded once, applied to every relevant question."
      actions={
        <Button onClick={() => setOpen((value) => !value)}>
          {open ? 'Hide' : rules.length ? `${rules.length} rule${rules.length === 1 ? '' : 's'}` : 'Add a rule'}
        </Button>
      }
    >
      {open && (
        <div className="space-y-3">
          <form onSubmit={submit} className="flex items-center gap-2">
            <input
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="e.g. only orders with status 'shipped' count as revenue"
              className="focus-ring flex-1 rounded-md border px-3 py-1.5 text-sm"
              style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
              disabled={busy}
            />
            <Button type="submit" variant="primary" disabled={busy || !text.trim()}>
              Add rule
            </Button>
          </form>
          {rules.length === 0 ? (
            <EmptyState>No business rules recorded yet.</EmptyState>
          ) : (
            <ul className="space-y-1.5">
              {rules.map((rule) => (
                <li
                  key={rule.id}
                  className="flex items-center justify-between gap-3 rounded-md px-2.5 py-1.5 text-sm"
                  style={{ background: 'var(--surface-3)' }}
                >
                  <span>{rule.rule_text}</span>
                  <button
                    type="button"
                    onClick={() => onDelete(rule.id)}
                    className="focus-ring shrink-0 text-[11px] underline"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Panel>
  )
}

function SuggestedQuestions({ questions, onAsk }) {
  if (!questions?.length) return null
  return (
    <Panel title="Try asking" subtitle="Dataset-specific questions, generated from the semantic layer.">
      <div className="flex flex-wrap gap-1.5">
        {questions.map((question) => (
          <button
            key={question}
            type="button"
            onClick={() => onAsk(question)}
            className="focus-ring rounded-full border px-2.5 py-1 text-[11px] hover:bg-[var(--surface-3)]"
            style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
          >
            {question}
          </button>
        ))}
      </div>
    </Panel>
  )
}

function AttemptsLog({ attempts }) {
  const [open, setOpen] = useState(false)
  if (!attempts?.length) return null
  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="focus-ring text-[11px] underline"
        style={{ color: 'var(--text-muted)' }}
      >
        {open ? 'Hide' : 'Show'} what was tried ({attempts.length} attempt
        {attempts.length === 1 ? '' : 's'})
      </button>
      {open && (
        <ol className="mt-1.5 space-y-1.5 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          {attempts.map((attempt) => (
            <li key={attempt.attempt}>
              <span style={{ color: 'var(--text-muted)' }}>#{attempt.attempt}</span>{' '}
              {attempt.sql && (
                <code className="rounded px-1 py-0.5" style={{ background: 'var(--surface-3)' }}>
                  {attempt.sql}
                </code>
              )}
              {attempt.error && (
                <div style={{ color: 'var(--status-critical)' }}>{attempt.error}</div>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function ConfidenceBadge({ confidence }) {
  if (confidence == null) return null
  return (
    <Tag title="the model's own confidence that this SQL answers the question correctly">
      {`${Math.round(confidence * 100)}% confidence`}
    </Tag>
  )
}

function AnswerCard({ entry, onPin, onAsk, onSaveExample }) {
  const [view, setView] = useState('chart')
  const [showSql, setShowSql] = useState(false)
  const [copied, setCopied] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [saved, setSaved] = useState(false)
  const { result } = entry

  const saveExample = async () => {
    if (saved) return
    const ok = await onSaveExample(entry)
    if (ok) {
      setSaved(true)
      setTimeout(() => setSaved(false), 1500)
    }
  }

  const copyData = async () => {
    try {
      await navigator.clipboard.writeText(toCsv(result.columns, result.rows))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard access can be denied by the browser; nothing to recover into.
    }
  }

  const pin = async () => {
    if (pinned) return
    const ok = await onPin(entry)
    if (ok) {
      setPinned(true)
      setTimeout(() => setPinned(false), 1500)
    }
  }

  return (
    <Panel
      title={entry.question}
      subtitle={
        result.ok
          ? `${result.row_count.toLocaleString()} row${result.row_count === 1 ? '' : 's'}${
              result.truncated ? ` (capped — more matched)` : ''
            }`
          : undefined
      }
      actions={
        result.ok ? (
          <div className="flex items-center gap-2">
            <ConfidenceBadge confidence={result.confidence} />
            {result.visualization?.chart !== 'table' && (
              <Button onClick={() => setView((value) => (value === 'chart' ? 'table' : 'chart'))}>
                {view === 'chart' ? 'Table view' : 'Chart view'}
              </Button>
            )}
            <Button onClick={copyData}>{copied ? 'Copied' : 'Copy data'}</Button>
            <Button onClick={() => setShowSql((value) => !value)}>
              {showSql ? 'Hide SQL' : 'View SQL'}
            </Button>
            {onSaveExample && (
              <Button onClick={saveExample}>{saved ? 'Saved ✓' : 'Save as example'}</Button>
            )}
            {onPin && (
              <Button variant="primary" onClick={pin}>
                {pinned ? 'Pinned ✓' : 'Pin to dashboard'}
              </Button>
            )}
          </div>
        ) : undefined
      }
    >
      {!result.ok ? (
        <div>
          <p
            className="text-sm"
            style={{
              color:
                result.outcome === 'abstained' ? 'var(--status-warning)' : 'var(--status-critical)',
            }}
          >
            {result.outcome === 'abstained' && !result.ambiguous && '⚠ '}
            {result.error}
          </p>
          {result.confidence != null && (
            <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
              Withheld rather than shown — confidence was only{' '}
              {Math.round(result.confidence * 100)}%.
            </p>
          )}
          {result.ambiguous && result.clarification_options?.length > 0 && (
            <div className="mt-3">
              <p className="mb-1.5 text-[11px] font-medium" style={{ color: 'var(--text-muted)' }}>
                Pick what you meant
              </p>
              <div className="flex flex-wrap gap-1.5">
                {result.clarification_options.map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => onAsk(option)}
                    className="focus-ring rounded-full border px-2.5 py-1 text-[11px] hover:bg-[var(--surface-3)]"
                    style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </div>
          )}
          {result.tables_considered?.length > 0 && (
            <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
              Tables considered: {result.tables_considered.join(', ')}
            </p>
          )}
          <AttemptsLog attempts={result.attempts} />
        </div>
      ) : (
        <div>
          {result.explanation && (
            <p className="mb-3 text-sm" style={{ color: 'var(--text-secondary)' }}>
              {result.explanation}
            </p>
          )}
          {result.rows.length === 0 ? (
            <EmptyState>No rows matched this question.</EmptyState>
          ) : view === 'table' ? (
            <DataTable columns={result.columns} rows={result.rows} />
          ) : (
            <Chart columns={result.columns} rows={result.rows} visualization={result.visualization} />
          )}
          {result.tables_used?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1">
              {result.tables_used.map((table) => (
                <Tag key={table} muted title="table read to answer this question">
                  {table}
                </Tag>
              ))}
            </div>
          )}
          {showSql && (
            <pre
              className="mt-3 overflow-x-auto rounded p-3 text-[11px] leading-relaxed"
              style={{ background: 'var(--surface-3)' }}
            >
              {result.sql}
            </pre>
          )}
          {result.attempts?.length > 1 && <AttemptsLog attempts={result.attempts} />}
          {result.suggestions?.length > 0 && (
            <div className="mt-3">
              <p className="mb-1.5 text-[11px] font-medium" style={{ color: 'var(--text-muted)' }}>
                Ask a follow-up
              </p>
              <div className="flex flex-wrap gap-1.5">
                {result.suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => onAsk(suggestion)}
                    className="focus-ring rounded-full border px-2.5 py-1 text-[11px] hover:bg-[var(--surface-3)]"
                    style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}

export function QueryPanel({
  exported,
  entries,
  busy,
  onAsk,
  onPin,
  rules = [],
  suggestions = [],
  onLoadExtras,
  onAddRule,
  onDeleteRule,
  onSaveExample,
}) {
  const [question, setQuestion] = useState('')

  useEffect(() => {
    if (exported && onLoadExtras) onLoadExtras()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exported])

  const submit = (event) => {
    event.preventDefault()
    const trimmed = question.trim()
    if (!trimmed || busy) return
    onAsk(trimmed)
    setQuestion('')
  }

  if (!exported) {
    return (
      <Panel
        title="Ask questions"
        subtitle="Ask the exported database a question in plain English."
      >
        <EmptyState>Build the semantic layer first — there is nothing to query yet.</EmptyState>
      </Panel>
    )
  }

  return (
    <div className="space-y-4">
      {onAddRule && (
        <BusinessRulesPanel rules={rules} busy={busy} onAdd={onAddRule} onDelete={onDeleteRule} />
      )}

      <SuggestedQuestions questions={suggestions} onAsk={onAsk} />

      <Panel
        title="Ask questions"
        subtitle="Answered against the exported database — only your question and the schema ever reach the model, never a data row."
      >
        <form onSubmit={submit} className="flex items-center gap-2">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="e.g. which city has the most customers?"
            className="focus-ring flex-1 rounded-md border px-3 py-1.5 text-sm"
            style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
            disabled={busy}
          />
          <Button type="submit" variant="primary" disabled={busy || !question.trim()}>
            {busy ? 'Asking…' : 'Ask'}
          </Button>
        </form>
      </Panel>

      {entries.length === 0 ? (
        <EmptyState>Ask a question above to see it answered here.</EmptyState>
      ) : (
        entries.map((entry) => (
          <AnswerCard
            key={entry.id}
            entry={entry}
            onPin={onPin}
            onAsk={onAsk}
            onSaveExample={onSaveExample}
          />
        ))
      )}
    </div>
  )
}
