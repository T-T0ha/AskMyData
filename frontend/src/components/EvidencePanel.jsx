/** Phase 3 view — the evidence behind one relationship, and the verdict on it.
 *
 *  SemTabla's validation framework: every claim is paired with rows that
 *  support it and rows that contradict it, each with the SQL that found them.
 *  That pairing is the whole reason a ~75 %-accurate detector can produce a
 *  semantic layer that is right — the user is not asked to trust a score, they
 *  are shown what it rests on.
 *
 *  The negative sample leads. A foreign key claims *every* value on the left is
 *  a real key on the right, so one contradicting row settles it, while ten
 *  supporting rows settle nothing.
 */

import { Button, DataTable, EmptyState, Panel, Spinner, StatusBadge } from './ui'

function Sql({ sql }) {
  return (
    <pre
      className="mt-2 overflow-x-auto rounded p-2 text-[10.5px] leading-relaxed"
      style={{ background: 'var(--surface-3)', color: 'var(--text-secondary)' }}
    >
      <code>{sql}</code>
    </pre>
  )
}

function SampleBlock({ sample, tone }) {
  const border = tone === 'negative' ? 'var(--status-critical)' : 'var(--status-good)'
  return (
    <section className="rounded-md border p-3" style={{ borderColor: border }}>
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="text-xs font-semibold">{sample.title}</h4>
        <StatusBadge status={tone === 'negative' ? 'critical' : 'good'}>
          {sample.row_count.toLocaleString()} {tone === 'negative' ? 'contradicting' : 'supporting'}
        </StatusBadge>
      </header>
      <p className="mt-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {sample.explanation}
      </p>
      <Sql sql={sample.sql} />
      <div className="mt-2">
        <DataTable
          rows={sample.rows}
          emptyMessage={
            tone === 'negative'
              ? 'Nothing contradicts this — the query returned no rows.'
              : 'No supporting rows found.'
          }
          highlight={tone === 'positive' ? 'positive' : undefined}
        />
      </div>
      {sample.rows.length < sample.row_count && (
        <p className="mt-1 text-[10.5px]" style={{ color: 'var(--text-muted)' }}>
          showing {sample.rows.length} of {sample.row_count.toLocaleString()}
        </p>
      )}
    </section>
  )
}

function Arithmetic({ relationship }) {
  const evidence = relationship.evidence ?? {}
  const rows =
    relationship.rel_type === 'foreign_key'
      ? [
          ['values that are real keys', percent(evidence.ratio_overlap)],
          ['share of the key reached', percent(evidence.ratio_distinct)],
          ['name similarity', evidence.embedding_similarity?.toFixed?.(2) ?? '—'],
          ['matched / distinct', `${fmt(evidence.matched_values)} / ${fmt(evidence.source_distinct)}`],
        ]
      : [
          ['values that repeat', fmt(evidence.witness_groups)],
          ['of those, consistent', fmt(evidence.consistent_witnesses)],
          ['distinct values in all', fmt(evidence.total_groups)],
          ['rows with a blank ignored', fmt(evidence.null_rows_dropped)],
        ]

  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-4">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt style={{ color: 'var(--text-muted)' }}>{label}</dt>
          <dd className="tabular" style={{ color: 'var(--text-secondary)' }}>
            {value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function percent(value) {
  return value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`
}

function fmt(value) {
  return value === null || value === undefined ? '—' : Number(value).toLocaleString()
}

export function EvidencePanel({ relationship, evidence, loading, busy, onDecide, onClose }) {
  if (!relationship) {
    return (
      <Panel title="Evidence">
        <EmptyState>
          Pick a relationship — from the diagram or the list — to see the rows that support it and
          the rows that contradict it.
        </EmptyState>
      </Panel>
    )
  }

  const isForeignKey = relationship.rel_type === 'foreign_key'
  const arrow = isForeignKey ? '→' : '⇒'
  const decided = relationship.status !== 'proposed'

  return (
    <Panel
      title={
        <span>
          {relationship.from_table}.{relationship.from_column}{' '}
          <span style={{ color: 'var(--text-muted)' }}>{arrow}</span> {relationship.to_table}.
          {relationship.to_column}
        </span>
      }
      subtitle={relationship.explanation}
      actions={
        <>
          <StatusBadge
            status={
              relationship.status === 'confirmed'
                ? 'good'
                : relationship.status === 'rejected'
                  ? 'neutral'
                  : relationship.uncertain
                    ? 'warning'
                    : 'neutral'
            }
          >
            {decided ? relationship.status : relationship.uncertain ? 'uncertain' : 'proposed'}
          </StatusBadge>
          {onClose && (
            <button type="button" className="focus-ring text-xs underline" onClick={onClose}>
              close
            </button>
          )}
        </>
      }
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-3 text-[11px]">
          <span
            className="rounded px-1.5 py-0.5"
            style={{ background: 'var(--surface-3)', color: 'var(--text-secondary)' }}
          >
            {relationship.rel_type.replace(/_/g, ' ')}
          </span>
          <span style={{ color: 'var(--text-muted)' }}>
            found by {relationship.origin} · score {relationship.score?.toFixed(2)}
          </span>
        </div>

        <Arithmetic relationship={relationship} />

        {relationship.llm_explanation && (
          <blockquote
            className="rounded-md border-l-2 py-1 pl-3 text-[11px]"
            style={{ borderColor: 'var(--series-1)', color: 'var(--text-secondary)' }}
          >
            {relationship.llm_explanation}
            <footer className="mt-1" style={{ color: 'var(--text-muted)' }}>
              — Claude, asked to read this borderline candidate. It did not detect or score it.
            </footer>
          </blockquote>
        )}

        {loading ? (
          <Spinner label="Querying the data…" />
        ) : evidence ? (
          <>
            {evidence.notes?.map((note) => (
              <p key={note} className="text-[11px]" style={{ color: 'var(--status-warning)' }}>
                {note}
              </p>
            ))}
            <SampleBlock sample={evidence.negative} tone="negative" />
            <SampleBlock sample={evidence.positive} tone="positive" />
          </>
        ) : null}

        <div className="flex flex-wrap justify-end gap-2 pt-1">
          {decided && (
            <span className="mr-auto self-center text-[11px]" style={{ color: 'var(--text-muted)' }}>
              You marked this {relationship.status}. Changing your mind is fine — nothing downstream
              has run yet.
            </span>
          )}
          <Button
            disabled={busy || relationship.status === 'rejected'}
            onClick={() => onDecide(relationship.id, false)}
          >
            Not a relationship
          </Button>
          <Button
            variant="primary"
            disabled={busy || relationship.status === 'confirmed'}
            onClick={() => onDecide(relationship.id, true)}
          >
            Confirm
          </Button>
        </div>
      </div>
    </Panel>
  )
}

export function RelationshipList({ relationships, selectedId, onSelect, busy }) {
  if (!relationships.length) {
    return (
      <EmptyState>
        Nothing was detected. Tables with no shared values are genuinely unrelated as far as the
        data shows — draw an edge in the diagram if you know otherwise.
      </EmptyState>
    )
  }

  return (
    <ul className="space-y-1">
      {relationships.map((row) => (
        <li key={row.id}>
          <button
            type="button"
            disabled={busy}
            onClick={() => onSelect(row.id)}
            className="focus-ring flex w-full items-center justify-between gap-2 rounded px-2 py-1.5 text-left text-[11px]"
            style={{
              background: row.id === selectedId ? 'var(--surface-3)' : 'transparent',
              opacity: row.status === 'rejected' ? 0.5 : 1,
            }}
          >
            <span className="min-w-0 truncate">
              <span style={{ color: 'var(--text-secondary)' }}>
                {row.from_table}.{row.from_column}
              </span>
              <span className="mx-1" style={{ color: 'var(--text-muted)' }}>
                {row.rel_type === 'foreign_key' ? '→' : '⇒'}
              </span>
              <span style={{ color: 'var(--text-secondary)' }}>
                {row.to_table === row.from_table ? row.to_column : `${row.to_table}.${row.to_column}`}
              </span>
            </span>
            <span className="flex shrink-0 items-center gap-2">
              <span className="tabular" style={{ color: 'var(--text-muted)' }}>
                {row.score?.toFixed(2)}
              </span>
              <StatusBadge
                status={
                  row.status === 'confirmed'
                    ? 'good'
                    : row.status === 'rejected'
                      ? 'neutral'
                      : row.uncertain
                        ? 'warning'
                        : 'neutral'
                }
              >
                {row.status === 'proposed' && row.uncertain ? 'uncertain' : row.status}
              </StatusBadge>
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}
