/** The adaptive chart renderer — shared between "Ask questions" (Phase 5) and
 *  the pinned dashboard (Phase 6), since both render exactly the same shape
 *  the backend decided (`visualization.chart`, computed in `app.query.shape`
 *  from the columns actually returned): this module's job is to render that
 *  choice, never to re-derive it, and to make the same table view available
 *  underneath every chart — a chart form is a claim about what is easiest to
 *  read, not the only accessible way to see the answer.
 *
 *  Colour: `--series-1` for anything single-series (bar, line, the metric
 *  card needs none). A pie chart uses up to three slots — the platform's
 *  validated categorical palette only clears the accessibility floors for a
 *  three-way all-pairs comparison, which is also why the backend caps a pie
 *  at three rows and falls back to a bar past that. A multi-line chart uses
 *  up to six — validated for the *adjacent* comparison a line chart's
 *  neighbouring series actually are — and folds anything past six into a
 *  note pointing at the table view rather than reusing a colour.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { DataTable, formatCell } from './ui'

const PIE_COLORS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']
const LINE_COLORS = [
  'var(--series-1)',
  'var(--series-2)',
  'var(--series-3)',
  'var(--series-4)',
  'var(--series-5)',
  'var(--series-6)',
]

function axisTick(value) {
  if (typeof value === 'number') return value.toLocaleString()
  const text = String(value ?? '')
  return text.length > 14 ? `${text.slice(0, 12)}…` : text
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div
      className="rounded border px-2.5 py-1.5 text-xs shadow-sm"
      style={{ background: 'var(--surface-1)', borderColor: 'var(--border)' }}
    >
      {label !== undefined && <div className="mb-1 font-medium">{axisTick(label)}</div>}
      {payload.map((entry) => (
        <div key={entry.dataKey} className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: entry.color }}
            aria-hidden="true"
          />
          <span style={{ color: 'var(--text-secondary)' }}>{entry.name}:</span>
          <span className="tabular font-medium">{formatCell(entry.value)}</span>
        </div>
      ))}
    </div>
  )
}

function MetricCard({ columns, rows }) {
  const value = rows[0]?.[columns[0]]
  return (
    <div className="py-6 text-center">
      <div className="text-4xl font-semibold">{formatCell(value)}</div>
      <div className="mt-1 text-xs" style={{ color: 'var(--text-muted)' }}>
        {columns[0]}
      </div>
    </div>
  )
}

function BarView({ columns, rows }) {
  const [category, metric] = columns
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
        <CartesianGrid vertical={false} stroke="var(--gridline)" />
        <XAxis
          dataKey={category}
          tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
          tickFormatter={axisTick}
          axisLine={{ stroke: 'var(--baseline)' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
          tickFormatter={axisTick}
          axisLine={false}
          tickLine={false}
          width={56}
        />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: 'var(--surface-3)' }} />
        <Bar dataKey={metric} fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={40} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// Recharts' default pie label inherits no particular fill, so identity would
// otherwise leak into the text itself; this renders the label by hand in a
// text token, leaving colour to the slice and the leader line alone.
function pieLabel({ cx, cy, midAngle, outerRadius, name, percent }) {
  const radians = (-midAngle * Math.PI) / 180
  const x = cx + (outerRadius + 16) * Math.cos(radians)
  const y = cy + (outerRadius + 16) * Math.sin(radians)
  return (
    <text
      x={x}
      y={y}
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      fontSize={11}
      fill="var(--text-secondary)"
    >
      {`${axisTick(name)} ${(percent * 100).toFixed(0)}%`}
    </text>
  )
}

function PieView({ columns, rows }) {
  const [category, metric] = columns
  return (
    <ResponsiveContainer width="100%" height={280}>
      <PieChart margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
        <Pie
          data={rows}
          dataKey={metric}
          nameKey={category}
          innerRadius={0}
          outerRadius={100}
          stroke="var(--surface-1)"
          strokeWidth={2}
          label={pieLabel}
        >
          {rows.map((_, index) => (
            <Cell key={index} fill={PIE_COLORS[index % PIE_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<ChartTooltip />} />
        <Legend
          verticalAlign="bottom"
          formatter={(value) => <span style={{ color: 'var(--text-secondary)' }}>{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

function LineOrMultiLineView({ rows, dateColumns, numericColumns }) {
  const [xKey] = dateColumns
  const shown = numericColumns.slice(0, LINE_COLORS.length)
  const hidden = numericColumns.length - shown.length
  return (
    <div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={rows} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
          <CartesianGrid vertical={false} stroke="var(--gridline)" />
          <XAxis
            dataKey={xKey}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            tickFormatter={axisTick}
            axisLine={{ stroke: 'var(--baseline)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            tickFormatter={axisTick}
            axisLine={false}
            tickLine={false}
            width={56}
          />
          <Tooltip content={<ChartTooltip />} />
          {shown.length > 1 && (
            <Legend
              formatter={(value) => <span style={{ color: 'var(--text-secondary)' }}>{value}</span>}
            />
          )}
          {shown.map((key, index) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={LINE_COLORS[index]}
              strokeWidth={2}
              dot={{ r: 3, fill: LINE_COLORS[index], strokeWidth: 0 }}
              activeDot={{ r: 5 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {hidden > 0 && (
        <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {hidden} more column{hidden === 1 ? '' : 's'} in this result — see the table view.
        </p>
      )}
    </div>
  )
}

export function Chart({ columns, rows, visualization }) {
  const { chart, numeric_columns: numericColumns = [], date_columns: dateColumns = [] } =
    visualization
  if (chart === 'metric') return <MetricCard columns={columns} rows={rows} />
  if (chart === 'bar') return <BarView columns={columns} rows={rows} />
  if (chart === 'pie') return <PieView columns={columns} rows={rows} />
  if (chart === 'line' || chart === 'multi_line')
    return <LineOrMultiLineView rows={rows} dateColumns={dateColumns} numericColumns={numericColumns} />
  return <DataTable columns={columns} rows={rows} />
}

export function toCsv(columns, rows) {
  const escape = (value) => {
    const text = value === null || value === undefined ? '' : String(value)
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
  }
  const lines = [columns.map(escape).join(',')]
  for (const row of rows) lines.push(columns.map((column) => escape(row[column])).join(','))
  return lines.join('\n')
}
