/** Phase 0 view — cross-sheet column equivalence confirmation.
 *
 *  These suggestions are the only thing that tells the platform two sheets
 *  talk about the same entity. A confirmation is recorded as a *relationship*
 *  between the two tables — the prior that foreign-key detection starts from —
 *  not as an instruction to fold them into one, so each card shows the
 *  evidence behind the score rather than just the score.
 */

import { Button, EmptyState, Panel, StatusBadge } from './ui'

function ScoreBar({ score }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 rounded-full" style={{ background: 'var(--surface-3)' }}>
        <div
          className="bar-h h-full"
          style={{ width: `${Math.round(score * 100)}%`, background: 'var(--series-1)' }}
        />
      </div>
      <span className="tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {score.toFixed(2)}
      </span>
    </div>
  )
}

function Evidence({ candidate }) {
  const items = [
    ['name embedding', candidate.embedding_similarity?.toFixed(2)],
    ['name spelling', candidate.lexical_similarity?.toFixed(2)],
    [
      'shared values',
      candidate.value_overlap === null || candidate.value_overlap === undefined
        ? '—'
        : `${Math.round(candidate.value_overlap * 100)}%`,
    ],
    ['storage types', candidate.type_compatible ? 'compatible' : 'different'],
  ]
  return (
    <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
      {items.map(([label, value]) => (
        <div key={label} className="flex justify-between gap-2">
          <dt style={{ color: 'var(--text-muted)' }}>{label}</dt>
          <dd className="tabular" style={{ color: 'var(--text-secondary)' }}>
            {value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

export function EquivalencePanel({ equivalences, onDecide, busy }) {
  const undecided = equivalences.filter((candidate) => candidate.confirmed === null)
  const decided = equivalences.filter((candidate) => candidate.confirmed !== null)

  return (
    <Panel
      title="Do these columns mean the same thing?"
      subtitle="A confirmed pair is kept as a link between two tables, not merged into one — that link is what becomes a foreign key."
    >
      {equivalences.length === 0 ? (
        <EmptyState>
          No cross-sheet column pairs scored above the similarity threshold. Sheets will be treated
          as unrelated until you say otherwise.
        </EmptyState>
      ) : (
        <div className="space-y-3">
          {undecided.map((candidate) => (
            <article
              key={candidate.id}
              className="rounded-md border p-3"
              style={{ borderColor: 'var(--border)', background: 'var(--surface-2)' }}
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm">
                    <code className="rounded px-1" style={{ background: 'var(--surface-3)' }}>
                      {candidate.left_table}.{candidate.left_column}
                    </code>
                    <span className="mx-2" style={{ color: 'var(--text-muted)' }}>
                      ≈
                    </span>
                    <code className="rounded px-1" style={{ background: 'var(--surface-3)' }}>
                      {candidate.right_table}.{candidate.right_column}
                    </code>
                  </p>
                  <Evidence candidate={candidate} />
                </div>
                <div className="flex shrink-0 flex-col items-end gap-2">
                  <ScoreBar score={candidate.score} />
                  <div className="flex gap-2">
                    <Button
                      variant="primary"
                      disabled={busy}
                      onClick={() => onDecide(candidate.id, true)}
                    >
                      Same concept
                    </Button>
                    <Button disabled={busy} onClick={() => onDecide(candidate.id, false)}>
                      Different
                    </Button>
                  </div>
                </div>
              </div>
            </article>
          ))}

          {decided.length > 0 && (
            <div className="space-y-2 pt-1">
              {decided.map((candidate) => (
                <div key={candidate.id} className="flex items-center justify-between gap-3 text-xs">
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {candidate.left_table}.{candidate.left_column} ≈ {candidate.right_table}.
                    {candidate.right_column}
                  </span>
                  <span className="flex items-center gap-2">
                    <StatusBadge status={candidate.confirmed ? 'good' : 'neutral'}>
                      {candidate.confirmed ? 'confirmed' : 'rejected'}
                    </StatusBadge>
                    <button
                      type="button"
                      className="focus-ring underline"
                      style={{ color: 'var(--series-1)' }}
                      disabled={busy}
                      onClick={() => onDecide(candidate.id, !candidate.confirmed)}
                    >
                      undo
                    </button>
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}
