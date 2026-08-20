/** Inline distribution charts for the Field Semantic View.
 *
 *  Every chart here plots exactly one series, so there is one hue (the
 *  categorical slot-1 blue) and no legend — the column name above it says what
 *  is plotted. Marks follow the fixed specs: 4px rounded data-end squared at
 *  the baseline, a 2px surface gap between adjacent bars, hairline recessive
 *  gridlines, and values on hover rather than a number on every mark.
 */

import { useState } from 'react'

const BAR_GAP = 2 // px of surface between touching marks
const BAR_MAX = 24 // marks spec: a bar is capped, never fills its slot

function Tooltip({ text }) {
  return (
    <span
      role="tooltip"
      className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1 -translate-x-1/2 whitespace-nowrap rounded px-1.5 py-1 text-[10px] shadow-sm"
      style={{
        background: 'var(--surface-1)',
        color: 'var(--text-primary)',
        border: '1px solid var(--border)',
      }}
    >
      {text}
    </span>
  )
}

function formatNumber(value) {
  if (value === null || value === undefined) return '—'
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return Number(value.toFixed?.(2) ?? value).toLocaleString()
}

/** Null ratio as a single filled bar. One measure, one hue. */
export function NullBar({ ratio, height = 6 }) {
  const percent = Math.round((ratio ?? 0) * 100)
  return (
    <div className="flex items-center gap-2">
      <div
        className="relative w-16 overflow-hidden rounded-full"
        style={{ height, background: 'var(--surface-3)' }}
        title={`${percent}% of values are missing`}
      >
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.max(percent === 0 ? 0 : 2, percent)}%`,
            background: percent === 0 ? 'transparent' : 'var(--series-1)',
          }}
        />
      </div>
      <span className="tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {percent}%
      </span>
    </div>
  )
}

/** Histogram for numeric columns. */
function Histogram({ distribution, height }) {
  const [hover, setHover] = useState(null)
  const counts = distribution.histogram?.counts ?? []
  const edges = distribution.histogram?.edges ?? []
  const max = Math.max(...counts, 1)
  if (!counts.length) return <Muted>no numeric values</Muted>

  return (
    <div>
      <div
        className="relative flex items-end"
        style={{ height, gap: BAR_GAP, borderBottom: '1px solid var(--baseline)' }}
      >
        {counts.map((count, index) => (
          <div
            key={index}
            className="relative flex-1"
            style={{ height: '100%' }}
            onMouseEnter={() => setHover(index)}
            onMouseLeave={() => setHover(null)}
          >
            {hover === index && (
              <Tooltip
                text={`${formatNumber(edges[index])} – ${formatNumber(edges[index + 1])}: ${count} row${count === 1 ? '' : 's'}`}
              />
            )}
            <div
              className="bar-v absolute bottom-0 left-1/2 w-full -translate-x-1/2"
              style={{
                height: `${(count / max) * 100}%`,
                minHeight: count > 0 ? 2 : 0,
                maxWidth: BAR_MAX,
                background: 'var(--series-1)',
                opacity: hover === null || hover === index ? 1 : 0.55,
              }}
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <span className="tabular">{formatNumber(distribution.min)}</span>
        <span className="tabular">{formatNumber(distribution.max)}</span>
      </div>
    </div>
  )
}

/** Top-k frequency bars for categorical columns. */
function CategoryBars({ distribution, limit }) {
  const [hover, setHover] = useState(null)
  const values = (distribution.top_values ?? []).slice(0, limit)
  if (!values.length) return <Muted>no values</Muted>
  const max = Math.max(...values.map((v) => v.count), 1)

  return (
    <div className="flex flex-col" style={{ gap: BAR_GAP }}>
      {values.map((entry, index) => (
        <div
          key={entry.value}
          className="relative flex items-center gap-2"
          onMouseEnter={() => setHover(index)}
          onMouseLeave={() => setHover(null)}
        >
          <span
            className="w-24 shrink-0 truncate text-[11px]"
            style={{ color: 'var(--text-secondary)' }}
            title={entry.value}
          >
            {entry.value}
          </span>
          <div className="relative h-2.5 flex-1" style={{ background: 'var(--surface-3)', borderRadius: 4 }}>
            {hover === index && (
              <Tooltip text={`${entry.value}: ${entry.count} row${entry.count === 1 ? '' : 's'} (${Math.round(entry.share * 100)}%)`} />
            )}
            <div
              className="bar-h absolute left-0 top-0 h-full"
              style={{ width: `${(entry.count / max) * 100}%`, background: 'var(--series-1)' }}
            />
          </div>
          <span className="tabular w-8 shrink-0 text-right text-[11px]" style={{ color: 'var(--text-muted)' }}>
            {entry.count}
          </span>
        </div>
      ))}
      {distribution.distinct > values.length && (
        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          +{distribution.distinct - values.length} more distinct value(s)
        </p>
      )}
    </div>
  )
}

/** Monthly counts for date columns. */
function TemporalBars({ distribution, height }) {
  const [hover, setHover] = useState(null)
  const months = distribution.by_month ?? []
  if (!months.length) return <Muted>no dates</Muted>
  const max = Math.max(...months.map((m) => m.count), 1)

  return (
    <div>
      <div
        className="relative flex items-end"
        style={{ height, gap: BAR_GAP, borderBottom: '1px solid var(--baseline)' }}
      >
        {months.map((month, index) => (
          <div
            key={month.period}
            className="relative flex-1"
            style={{ height: '100%' }}
            onMouseEnter={() => setHover(index)}
            onMouseLeave={() => setHover(null)}
          >
            {hover === index && <Tooltip text={`${month.period}: ${month.count} row${month.count === 1 ? '' : 's'}`} />}
            <div
              className="bar-v absolute bottom-0 left-1/2 w-full -translate-x-1/2"
              style={{
                height: `${(month.count / max) * 100}%`,
                minHeight: 2,
                maxWidth: BAR_MAX,
                background: 'var(--series-1)',
                opacity: hover === null || hover === index ? 1 : 0.55,
              }}
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <span>{String(distribution.min).slice(0, 10)}</span>
        <span>{String(distribution.max).slice(0, 10)}</span>
      </div>
    </div>
  )
}

function BooleanSplit({ distribution }) {
  const values = distribution.values ?? []
  if (!values.length) return <Muted>no values</Muted>
  return (
    <div className="flex flex-col gap-1">
      {values.map((entry) => (
        <div key={entry.value} className="flex items-center gap-2">
          <span className="w-14 shrink-0 truncate text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            {entry.value}
          </span>
          <div className="h-2.5 flex-1" style={{ background: 'var(--surface-3)', borderRadius: 4 }}>
            <div
              className="bar-h h-full"
              style={{ width: `${entry.share * 100}%`, background: 'var(--series-1)' }}
            />
          </div>
          <span className="tabular w-10 shrink-0 text-right text-[11px]" style={{ color: 'var(--text-muted)' }}>
            {Math.round(entry.share * 100)}%
          </span>
        </div>
      ))}
    </div>
  )
}

function Muted({ children }) {
  return (
    <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
      {children}
    </span>
  )
}

export function Distribution({ distribution, height = 32, limit = 4 }) {
  if (!distribution || distribution.empty) return <Muted>no data</Muted>
  switch (distribution.kind) {
    case 'numeric':
      return <Histogram distribution={distribution} height={height} />
    case 'categorical':
      return <CategoryBars distribution={distribution} limit={limit} />
    case 'temporal':
      return <TemporalBars distribution={distribution} height={height} />
    case 'boolean':
      return <BooleanSplit distribution={distribution} />
    default:
      return <Muted>—</Muted>
  }
}

/** The numbers behind the chart — a chart always has a readable table form. */
export function DistributionStats({ distribution }) {
  if (!distribution || distribution.empty) return null
  const rows =
    distribution.kind === 'numeric'
      ? [
          ['min', formatNumber(distribution.min)],
          ['max', formatNumber(distribution.max)],
          ['mean', formatNumber(distribution.mean)],
          ['median', formatNumber(distribution.median)],
          ['std dev', formatNumber(distribution.std)],
          ['zeros', distribution.zero_count],
        ]
      : distribution.kind === 'categorical'
        ? [
            ['distinct', distribution.distinct],
            ['longest value', `${distribution.longest_value} chars`],
            ['top-10 coverage', `${Math.round((distribution.covered_share ?? 0) * 100)}%`],
          ]
        : distribution.kind === 'temporal'
          ? [
              ['earliest', String(distribution.min).slice(0, 10)],
              ['latest', String(distribution.max).slice(0, 10)],
              ['span', `${distribution.span_days} days`],
            ]
          : (distribution.values ?? []).map((v) => [v.value, `${v.count} (${Math.round(v.share * 100)}%)`])

  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
      {rows.map(([label, value]) => (
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
