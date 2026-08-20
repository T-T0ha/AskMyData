/** Shared primitives.
 *
 *  Every status colour here is paired with a text label and a glyph, so state
 *  is never carried by colour alone — the rule that governs the fixed status
 *  palette.
 */

export function Button({ variant = 'default', className = '', ...props }) {
  const styles = {
    default: 'bg-[var(--surface-3)] hover:brightness-95',
    primary: 'text-white hover:brightness-110',
    ghost: 'hover:bg-[var(--surface-3)]',
    danger: 'text-white hover:brightness-110',
  }
  const inline =
    variant === 'primary'
      ? { background: 'var(--series-1)' }
      : variant === 'danger'
        ? { background: 'var(--status-critical)' }
        : undefined
  return (
    <button
      type="button"
      style={inline}
      className={`focus-ring rounded-md px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${styles[variant]} ${className}`}
      {...props}
    />
  )
}

const STATUS_TOKENS = {
  good: { color: 'var(--status-good)', glyph: '✓' },
  warning: { color: 'var(--status-warning)', glyph: '!' },
  serious: { color: 'var(--status-serious)', glyph: '▲' },
  critical: { color: 'var(--status-critical)', glyph: '✕' },
  neutral: { color: 'var(--text-muted)', glyph: '·' },
}

export function StatusBadge({ status = 'neutral', children, title }) {
  const token = STATUS_TOKENS[status] ?? STATUS_TOKENS.neutral
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium"
      style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
    >
      <span aria-hidden="true" style={{ color: token.color }} className="text-[10px] leading-none">
        {token.glyph}
      </span>
      <span style={{ color: 'var(--text-secondary)' }}>{children}</span>
    </span>
  )
}

export function Tag({ children, title, muted = false }) {
  return (
    <span
      title={title}
      className="inline-block rounded px-1.5 py-0.5 text-[11px] font-medium"
      style={{
        background: 'var(--surface-3)',
        color: muted ? 'var(--text-muted)' : 'var(--text-secondary)',
      }}
    >
      {children}
    </span>
  )
}

export function Panel({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header
          className="flex items-start justify-between gap-4 border-b px-4 py-3"
          style={{ borderColor: 'var(--border)' }}
        >
          <div>
            {title && <h2 className="text-sm font-semibold">{title}</h2>}
            {subtitle && (
              <p className="mt-0.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
                {subtitle}
              </p>
            )}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

export function EmptyState({ children }) {
  return (
    <p className="py-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
      {children}
    </p>
  )
}

export function ErrorBanner({ error, onDismiss }) {
  if (!error) return null
  return (
    <div
      role="alert"
      className="flex items-start justify-between gap-3 rounded-md border px-3 py-2 text-sm"
      style={{ borderColor: 'var(--status-critical)', background: 'var(--surface-1)' }}
    >
      <span>
        <span aria-hidden="true" style={{ color: 'var(--status-critical)' }} className="mr-2">
          ✕
        </span>
        {String(error)}
      </span>
      {onDismiss && (
        <button type="button" onClick={onDismiss} className="focus-ring text-xs underline">
          dismiss
        </button>
      )}
    </div>
  )
}

export function Spinner({ label = 'Working…' }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
      <span
        className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-t-transparent"
        style={{ borderColor: 'var(--series-1)', borderTopColor: 'transparent' }}
      />
      {label}
    </span>
  )
}

/** Small labelled table used for previews and evidence. */
export function DataTable({ columns, rows, emptyMessage = 'No rows', highlight }) {
  if (!rows?.length) {
    return (
      <p className="px-2 py-3 text-xs" style={{ color: 'var(--text-muted)' }}>
        {emptyMessage}
      </p>
    )
  }
  const keys = columns?.length ? columns : Object.keys(rows[0])
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr style={{ color: 'var(--text-muted)' }}>
            {keys.map((key) => (
              <th
                key={key}
                className="whitespace-nowrap border-b px-2 py-1.5 text-left font-medium"
                style={{ borderColor: 'var(--gridline)' }}
              >
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {keys.map((key) => (
                <td
                  key={key}
                  className="tabular whitespace-nowrap border-b px-2 py-1.5"
                  style={{
                    borderColor: 'var(--gridline)',
                    background: highlight === 'positive' ? 'var(--series-1-wash)' : undefined,
                  }}
                >
                  {formatCell(row[key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function formatCell(value) {
  if (value === null || value === undefined) {
    return <span style={{ color: 'var(--text-muted)' }}>—</span>
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 4 })
  }
  const text = String(value)
  return text.length > 48 ? `${text.slice(0, 45)}…` : text
}
