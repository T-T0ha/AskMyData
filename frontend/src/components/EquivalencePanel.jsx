/** Phase 0 view — which columns, in different tables, hold the same thing.
 *
 *  Most pairs are not close calls: the same name, the same kind of values, many
 *  of them shared. Those are linked without asking, and the screen says so. A
 *  person is asked only about a pair the evidence cannot settle, in plain words,
 *  and an unanswered pair is simply treated as "different" — nothing waits on it.
 *
 *  A confirmed pair feeds the cleaning plan (two sheets that are one table split
 *  in two) and is shown to the model that helps plan it; it is not itself a
 *  foreign key. Every automatic call can be reversed from here.
 */

import { Button, EmptyState, Panel, StatusBadge } from './ui'

function Column({ table, column }) {
  return (
    <span>
      <code className="rounded px-1" style={{ background: 'var(--surface-3)' }}>
        {column}
      </code>{' '}
      <span style={{ color: 'var(--text-muted)' }}>in {table}</span>
    </span>
  )
}

function Pair({ candidate }) {
  return (
    <span className="text-sm">
      <Column table={candidate.left_table} column={candidate.left_column} />
      <span className="mx-2" style={{ color: 'var(--text-muted)' }}>
        =
      </span>
      <Column table={candidate.right_table} column={candidate.right_column} />
    </span>
  )
}

function overlapNote(candidate) {
  if (candidate.value_overlap === null || candidate.value_overlap === undefined) return null
  const share = Math.round(candidate.value_overlap * 100)
  return share === 0
    ? 'They have no values in common.'
    : `${share}% of the values in the smaller column also appear in the other.`
}

function Question({ candidate, onDecide, busy }) {
  return (
    <article
      className="rounded-md border p-3"
      style={{ borderColor: 'var(--status-warning)', background: 'var(--surface-2)' }}
    >
      <p className="text-sm">
        Do these two columns mean the same thing?
      </p>
      <p className="mt-1">
        <Pair candidate={candidate} />
      </p>
      <p className="mt-1 text-xs" style={{ color: 'var(--text-muted)' }}>
        {overlapNote(candidate)} The names are similar but not identical.
      </p>
      <div className="mt-2 flex gap-2">
        <Button variant="primary" disabled={busy} onClick={() => onDecide(candidate.id, true)}>
          Yes, the same
        </Button>
        <Button disabled={busy} onClick={() => onDecide(candidate.id, false)}>
          No, different
        </Button>
      </div>
    </article>
  )
}

function Row({ candidate, actionLabel, onAction, busy }) {
  return (
    <li className="flex flex-wrap items-center justify-between gap-2 py-1.5">
      <Pair candidate={candidate} />
      <span className="flex items-center gap-3 text-xs">
        {candidate.decided_by === 'user' && (
          <span style={{ color: 'var(--text-muted)' }}>your choice</span>
        )}
        <button
          type="button"
          className="focus-ring underline"
          style={{ color: 'var(--series-1)' }}
          disabled={busy}
          onClick={onAction}
        >
          {actionLabel}
        </button>
      </span>
    </li>
  )
}

export function EquivalencePanel({ equivalences, onDecide, busy }) {
  const asking = equivalences.filter((candidate) => candidate.confirmed === null)
  const linked = equivalences.filter((candidate) => candidate.confirmed === true)
  const ruledOut = equivalences.filter((candidate) => candidate.confirmed === false)

  return (
    <Panel
      title="Matching columns"
      subtitle="Columns in different tables that hold the same information — a customer number in both an orders table and a customers table, say. The obvious matches are linked for you."
    >
      {equivalences.length === 0 ? (
        <EmptyState>
          No matching columns were found between your tables. There is nothing for you to do here.
        </EmptyState>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <StatusBadge status="good">{linked.length} linked automatically</StatusBadge>
            {ruledOut.length > 0 && (
              <StatusBadge status="neutral">{ruledOut.length} ruled out</StatusBadge>
            )}
            {asking.length > 0 && (
              <StatusBadge status="warning">{asking.length} to check</StatusBadge>
            )}
            {asking.length === 0 && (
              <span style={{ color: 'var(--text-muted)' }}>Nothing needs your attention.</span>
            )}
          </div>

          {asking.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                These are close calls. Answer if you know; if you skip them they are treated as
                different, and nothing else waits on your answer.
              </p>
              {asking.map((candidate) => (
                <Question
                  key={candidate.id}
                  candidate={candidate}
                  onDecide={onDecide}
                  busy={busy}
                />
              ))}
            </div>
          )}

          {linked.length > 0 && (
            <details>
              <summary className="cursor-pointer text-sm font-medium">
                See the {linked.length} linked pair{linked.length === 1 ? '' : 's'}
              </summary>
              <ul className="mt-1 divide-y" style={{ borderColor: 'var(--gridline)' }}>
                {linked.map((candidate) => (
                  <Row
                    key={candidate.id}
                    candidate={candidate}
                    actionLabel="not the same"
                    busy={busy}
                    onAction={() => onDecide(candidate.id, false)}
                  />
                ))}
              </ul>
            </details>
          )}

          {ruledOut.length > 0 && (
            <details>
              <summary className="cursor-pointer text-sm font-medium">
                See the {ruledOut.length} pair{ruledOut.length === 1 ? '' : 's'} ruled out
              </summary>
              <ul className="mt-1 divide-y" style={{ borderColor: 'var(--gridline)' }}>
                {ruledOut.map((candidate) => (
                  <Row
                    key={candidate.id}
                    candidate={candidate}
                    actionLabel="they are the same"
                    busy={busy}
                    onAction={() => onDecide(candidate.id, true)}
                  />
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </Panel>
  )
}
