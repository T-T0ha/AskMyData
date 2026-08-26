/** Table semantic view — what each table *is*.
 *
 *  The fourth and last of SemTabla's enrichment steps, and the first one that
 *  says something about a whole table rather than a column or an edge. Each
 *  card carries the table's type, the confidence the decision tree reached it
 *  with, the sentence explaining that branch, and the labels the tree read.
 *
 *  Everything here is correctable, because everything here is inferred: the
 *  type is a dropdown, and each label can be struck out or added back. A
 *  correction sits *beside* the detected value rather than replacing it — the
 *  card keeps showing what the detector said, which is the only way a user can
 *  tell a corrected table from an uncontested one.
 */

import { useState } from 'react'
import { Button, EmptyState, Panel, Tag } from './ui'

/** Plain-English readings of the thirteen labels of SemTabla's Table 8. The
 *  evidence line under each one is computed; this is only the name. */
const LABEL_TITLES = {
  is_primary_key_time: 'the key is a moment in time',
  is_primary_key_periodic: 'the time key ticks at a fixed interval',
  is_single_value_column: 'exactly one measured value',
  is_single_enum_column: 'exactly one enumerated column',
  is_data_discrete: 'the numbers repeat rather than spread',
  is_enum_containing_most: 'mostly codes and categories',
  is_label_having_hierarchy_column: 'the columns name levels of something',
  is_mostly_referenced: 'other tables lean on this one',
  is_self_reference: 'rows point at other rows here',
  is_no_single_value_as_primary_key: 'no single column identifies a row',
  is_exactly_two_foreign_keys_existing: 'exactly two foreign keys',
  is_having_dependency_chain: 'a chain of dependencies three columns long',
  is_fd_stable_after_null_drop: 'the dependencies survive dropping empty columns',
}

const TYPE_BLURB = {
  fact: 'measurements, hanging off other tables',
  dimension: 'describes an entity others point at',
  bridge: 'joins two tables and holds nothing else',
  time_series: 'one row per moment',
  hierarchy: 'rows sit above and below one another',
  lookup: 'a short closed list of codes',
  wide: 'everything about everything, denormalised',
  unknown: 'no branch of the tree fit — say what it is',
}

function confidenceBand(value) {
  if (value >= 0.85) return { label: 'high', color: 'var(--ok)' }
  if (value >= 0.7) return { label: 'medium', color: 'var(--series-1)' }
  if (value > 0) return { label: 'low', color: 'var(--warn)' }
  return { label: 'none', color: 'var(--text-muted)' }
}

function LabelChip({ label, removed, onToggle }) {
  const title = [LABEL_TITLES[label.name] ?? label.name, label.evidence].filter(Boolean).join(' — ')
  return (
    <button
      type="button"
      onClick={onToggle}
      title={`${title}\n\nClick to ${removed ? 'restore' : 'remove'} this label.`}
      className="rounded px-1.5 py-0.5 text-[11px] font-medium transition-opacity hover:opacity-70"
      style={{
        background: 'var(--surface-3)',
        color: removed ? 'var(--text-muted)' : 'var(--text-secondary)',
        textDecoration: removed ? 'line-through' : 'none',
        border: label.origin === 'user' ? '1px dashed var(--border)' : '1px solid transparent',
      }}
    >
      {label.name.replace(/^is_/, '').replaceAll('_', ' ')}
    </button>
  )
}

function ProfileCard({ profile, tableTypes, onCorrect, busy }) {
  const [open, setOpen] = useState(false)
  const detected = profile.table_type
  const effective = profile.effective_type
  const corrected = Boolean(profile.user_table_type)
  const band = confidenceBand(profile.type_confidence)
  const removed = new Set(profile.removed_labels ?? [])
  const added = (profile.added_labels ?? []).map((name) => ({
    name,
    evidence: 'added by you',
    origin: 'user',
  }))
  const labels = [...(profile.labels ?? []), ...added]
  const features = profile.features ?? {}

  return (
    <li className="rounded border p-3" style={{ borderColor: 'var(--border)' }}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm font-semibold">{profile.table}</span>
            <select
              value={effective}
              disabled={busy}
              onChange={(event) => onCorrect(profile.table, { table_type: event.target.value })}
              title={TYPE_BLURB[effective] ?? ''}
              className="rounded border px-1.5 py-0.5 text-xs"
              style={{
                borderColor: 'var(--border)',
                background: 'var(--surface-1)',
                color: 'var(--text-primary)',
              }}
            >
              {tableTypes.map((value) => (
                <option key={value} value={value}>
                  {value.replaceAll('_', ' ')}
                </option>
              ))}
            </select>
            {corrected ? (
              <button
                type="button"
                onClick={() => onCorrect(profile.table, { table_type: '' })}
                title={`The detector said "${detected}". Click to go back to it.`}
                className="text-[11px] underline"
                style={{ color: 'var(--text-muted)' }}
              >
                corrected from {detected}
              </button>
            ) : (
              <span
                className="tabular text-[11px]"
                style={{ color: band.color }}
                title={`Decision-tree confidence. Below 0.55 the table is reported as unknown rather than guessed.`}
              >
                {band.label} · {profile.type_confidence.toFixed(2)}
              </span>
            )}
          </div>
          <p className="mt-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
            {profile.rationale}
          </p>
          {/* The sentence the query stage will use to decide whether a question
              is about this table — shown here so a wrong one can be seen. */}
          {profile.description && (
            <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
              {profile.description}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          className="shrink-0 text-[11px] underline"
          style={{ color: 'var(--text-muted)' }}
        >
          {open ? 'hide evidence' : 'evidence'}
        </button>
      </div>

      {labels.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {labels.map((label) => (
            <LabelChip
              key={label.name}
              label={label}
              removed={removed.has(label.name)}
              onToggle={() =>
                onCorrect(
                  profile.table,
                  removed.has(label.name) || label.origin === 'user'
                    ? { add_label: label.name }
                    : { remove_label: label.name },
                )
              }
            />
          ))}
        </div>
      )}

      {open && (
        <dl
          className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 border-t pt-2 text-[11px] sm:grid-cols-2"
          style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
        >
          {(labels.filter((label) => !removed.has(label.name)) ?? []).map((label) => (
            <div key={label.name} className="sm:col-span-2">
              <dt className="font-medium">{LABEL_TITLES[label.name] ?? label.name}</dt>
              <dd style={{ color: 'var(--text-muted)' }}>{label.evidence}</dd>
            </div>
          ))}
          <Detail term="key" value={(features.primary_key ?? []).join(' + ')} />
          <Detail term="measures" value={(features.measure_columns ?? []).join(', ')} />
          <Detail term="enumerations" value={(features.enum_columns ?? []).join(', ')} />
          <Detail term="points at" value={(features.references_tables ?? []).join(', ')} />
          <Detail term="pointed at by" value={(features.referenced_by_tables ?? []).join(', ')} />
          <Detail
            term="rows / columns"
            value={`${features.row_count ?? 0} / ${features.column_count ?? 0}`}
          />
          {features.unconfirmed_edges > 0 && (
            <div className="sm:col-span-2" style={{ color: 'var(--warn)' }}>
              {features.unconfirmed_edges} of the references this reads are still proposals —
              confirm or reject them and the profile follows.
            </div>
          )}
        </dl>
      )}
    </li>
  )
}

function Detail({ term, value }) {
  if (!value) return null
  return (
    <div className="flex gap-2">
      <dt className="shrink-0" style={{ color: 'var(--text-muted)' }}>
        {term}
      </dt>
      <dd className="font-mono">{value}</dd>
    </div>
  )
}

export function TableProfilePanel({ profiles, counts, tableTypes = [], onCorrect, onRefresh, busy }) {
  if (!profiles?.length) {
    return (
      <Panel title="Table semantics">
        <EmptyState>
          Run relationship detection — table profiling reads its output, and the keys and
          references are half of what decides what a table is.
        </EmptyState>
      </Panel>
    )
  }

  const byType = counts?.by_type ?? {}
  return (
    <Panel
      title="Table semantics"
      subtitle="What each table is, read off its columns, keys, references and dependencies"
      actions={
        <Button onClick={onRefresh} disabled={busy}>
          Recompute
        </Button>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        {Object.entries(byType)
          .sort(([, a], [, b]) => b - a)
          .map(([type, count]) => (
            <Tag key={type} title={TYPE_BLURB[type] ?? ''} muted={type === 'unknown'}>
              {count} {type.replaceAll('_', ' ')}
            </Tag>
          ))}
        {counts?.corrected > 0 && <Tag muted>{counts.corrected} corrected by you</Tag>}
      </div>
      <ul className="space-y-2">
        {profiles.map((profile) => (
          <ProfileCard
            key={profile.table}
            profile={profile}
            tableTypes={tableTypes}
            onCorrect={onCorrect}
            busy={busy}
          />
        ))}
      </ul>
    </Panel>
  )
}
