/** Phase 3 view — the data model as a force-directed node-link diagram.
 *
 *  D3 owns the simulation and the SVG; React owns everything around it. The
 *  two do not share the DOM — React renders one empty <svg> and never touches
 *  its children again, because both libraries reconciling the same nodes is
 *  the classic way to get elements that vanish mid-drag.
 *
 *  What the drawing encodes, and why:
 *
 *    solid edge     confirmed — part of the model
 *    dashed edge    proposed — detected but nobody has looked at it yet
 *    orange dashed  uncertain — scored in the fuzzy band, below the accept
 *                   threshold, shown because dropping it silently would hide a
 *                   relationship that messy data degraded
 *    gold column    primary key
 *    blue column    foreign key
 *
 *  Every one of those is paired with text somewhere — the edge list, the badge
 *  on the panel — so the diagram never carries meaning by colour alone.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  drag as d3drag,
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  select,
  zoom as d3zoom,
  zoomIdentity,
} from 'd3'

const NODE_WIDTH = 168
const HEADER_HEIGHT = 26
const ROW_HEIGHT = 15
const MAX_ROWS = 7

function nodeHeight(node, expanded) {
  const rows = expanded ? Math.min(node.columns.length, 40) : Math.min(node.columns.length, MAX_ROWS)
  return HEADER_HEIGHT + rows * ROW_HEIGHT + 8
}

/** Where an edge meets a node: the border point on the line between centres,
 *  so an arrowhead lands on the box rather than under it. */
function anchor(from, to, width, height) {
  const dx = to.x - from.x
  const dy = to.y - from.y
  if (!dx && !dy) return { x: from.x, y: from.y }
  const halfW = width / 2
  const halfH = height / 2
  const scale = Math.min(halfW / Math.abs(dx || 1e-6), halfH / Math.abs(dy || 1e-6))
  return { x: from.x + dx * scale, y: from.y + dy * scale }
}

export function RelationshipDiagram({
  graph,
  selectedId,
  onSelectEdge,
  onDrawEdge,
  height = 460,
}) {
  const svgRef = useRef(null)
  const containerRef = useRef(null)
  const simulationRef = useRef(null)
  const [expanded, setExpanded] = useState(() => new Set())
  const [pending, setPending] = useState(null) // first end of a hand-drawn edge

  // Latest callbacks without restarting the simulation on every render.
  const handlers = useRef({ onSelectEdge, onDrawEdge, expanded, pending })
  handlers.current = { onSelectEdge, onDrawEdge, expanded, pending }

  const data = useMemo(() => {
    const nodes = (graph?.nodes ?? []).map((node) => ({ ...node }))
    const byId = new Map(nodes.map((node) => [node.id, node]))
    const links = (graph?.links ?? [])
      .filter((link) => byId.has(link.source) && byId.has(link.target))
      .map((link) => ({ ...link }))
    return { nodes, links }
  }, [graph])

  const toggle = useCallback((id) => {
    setExpanded((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const clickColumn = useCallback(
    (table, column) => {
      setPending((current) => {
        if (!current) return { table, column }
        if (current.table === table && current.column === column) return null
        handlers.current.onDrawEdge?.({
          rel_type: 'foreign_key',
          from_table: current.table,
          from_column: current.column,
          to_table: table,
          to_column: column,
        })
        return null
      })
    },
    [],
  )

  useEffect(() => {
    const svg = select(svgRef.current)
    svg.selectAll('*').remove()
    if (!data.nodes.length) return undefined

    const width = containerRef.current?.clientWidth || 800
    const root = svg.append('g')

    svg.call(
      d3zoom()
        .scaleExtent([0.35, 2.5])
        .on('zoom', (event) => root.attr('transform', event.transform)),
    ).on('dblclick.zoom', null)

    // Arrowheads point from the referencing column to the key it references,
    // which is the direction a join reads in.
    const defs = svg.append('defs')
    for (const [id, color] of [
      ['arrow-confirmed', 'var(--series-1)'],
      ['arrow-proposed', 'var(--text-muted)'],
      ['arrow-uncertain', 'var(--status-warning)'],
    ]) {
      defs
        .append('marker')
        .attr('id', id)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 10)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', color)
    }

    const edgeColor = (link) =>
      link.status === 'confirmed'
        ? 'var(--series-1)'
        : link.uncertain
          ? 'var(--status-warning)'
          : 'var(--text-muted)'
    const edgeMarker = (link) =>
      link.status === 'confirmed'
        ? 'url(#arrow-confirmed)'
        : link.uncertain
          ? 'url(#arrow-uncertain)'
          : 'url(#arrow-proposed)'

    const linkGroup = root.append('g')
    const link = linkGroup
      .selectAll('path')
      .data(data.links)
      .join('path')
      .attr('fill', 'none')
      .attr('stroke', edgeColor)
      .attr('stroke-width', (d) => (d.id === selectedId ? 3 : d.status === 'confirmed' ? 2 : 1.5))
      .attr('stroke-dasharray', (d) => (d.status === 'confirmed' ? null : '5 3'))
      .attr('marker-end', edgeMarker)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation()
        handlers.current.onSelectEdge?.(d.id)
      })

    link.append('title').text((d) => `${d.from_column} → ${d.target}.${d.to_column}`)

    const node = root
      .append('g')
      .selectAll('g')
      .data(data.nodes)
      .join('g')
      .style('cursor', 'grab')

    node.each(function renderNode(d) {
      const group = select(this)
      const isExpanded = handlers.current.expanded.has(d.id)
      const columns = isExpanded ? d.columns.slice(0, 40) : d.columns.slice(0, MAX_ROWS)
      const h = nodeHeight(d, isExpanded)
      d.width = NODE_WIDTH
      d.height = h

      group
        .append('rect')
        .attr('x', -NODE_WIDTH / 2)
        .attr('y', -h / 2)
        .attr('width', NODE_WIDTH)
        .attr('height', h)
        .attr('rx', 6)
        .attr('fill', 'var(--surface-1)')
        .attr('stroke', 'var(--border)')

      group
        .append('rect')
        .attr('x', -NODE_WIDTH / 2)
        .attr('y', -h / 2)
        .attr('width', NODE_WIDTH)
        .attr('height', HEADER_HEIGHT)
        .attr('rx', 6)
        .attr('fill', 'var(--surface-3)')

      group
        .append('text')
        .attr('x', -NODE_WIDTH / 2 + 8)
        .attr('y', -h / 2 + 17)
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .attr('fill', 'var(--text-primary)')
        .text(d.id)

      group
        .append('text')
        .attr('x', NODE_WIDTH / 2 - 8)
        .attr('y', -h / 2 + 17)
        .attr('text-anchor', 'end')
        .attr('font-size', 9)
        .attr('fill', 'var(--text-muted)')
        .style('cursor', 'pointer')
        .text(`${d.row_count} rows${d.columns.length > MAX_ROWS ? (isExpanded ? ' ▾' : ' ▸') : ''}`)
        .on('click', (event) => {
          event.stopPropagation()
          toggle(d.id)
        })

      columns.forEach((column, index) => {
        const y = -h / 2 + HEADER_HEIGHT + index * ROW_HEIGHT + 11
        const isPending =
          handlers.current.pending?.table === d.id &&
          handlers.current.pending?.column === column.name
        group
          .append('text')
          .attr('x', -NODE_WIDTH / 2 + 8)
          .attr('y', y)
          .attr('font-size', 9.5)
          .attr('fill', () => {
            if (column.is_primary_key) return 'var(--series-3, #b8860b)'
            if (column.is_foreign_key) return 'var(--series-1)'
            return 'var(--text-secondary)'
          })
          .attr('font-weight', column.is_primary_key || column.is_foreign_key ? 600 : 400)
          .style('cursor', 'crosshair')
          .style('text-decoration', isPending ? 'underline' : null)
          .text(
            `${column.is_primary_key ? '◆ ' : column.is_foreign_key ? '↗ ' : '  '}${column.name}`.slice(
              0,
              24,
            ),
          )
          .on('click', (event) => {
            event.stopPropagation()
            clickColumn(d.id, column.name)
          })
          .append('title')
          .text(
            [
              `${d.id}.${column.name}`,
              column.label ? `means: ${column.label}` : null,
              column.is_primary_key ? 'primary key' : null,
              column.is_foreign_key ? 'foreign key' : null,
              'click two columns to draw a relationship',
            ]
              .filter(Boolean)
              .join('\n'),
          )
      })

      if (!isExpanded && d.columns.length > MAX_ROWS) {
        group
          .append('text')
          .attr('x', -NODE_WIDTH / 2 + 8)
          .attr('y', -h / 2 + HEADER_HEIGHT + columns.length * ROW_HEIGHT + 10)
          .attr('font-size', 9)
          .attr('fill', 'var(--text-muted)')
          .text(`+${d.columns.length - MAX_ROWS} more`)
      }
    })

    const simulation = forceSimulation(data.nodes)
      .force(
        'link',
        forceLink(data.links)
          .id((d) => d.id)
          .distance(230)
          .strength(0.35),
      )
      .force('charge', forceManyBody().strength(-900))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collide', forceCollide().radius(120))
      // Gentle pull to the middle so an isolated table — a real outcome, not a
      // bug — stays on screen instead of being flung past the viewport edge.
      .force('x', forceX(width / 2).strength(0.04))
      .force('y', forceY(height / 2).strength(0.06))

    simulationRef.current = simulation

    simulation.on('tick', () => {
      link.attr('d', (d) => {
        const from = anchor(d.source, d.target, d.source.width ?? NODE_WIDTH, d.source.height ?? 80)
        const to = anchor(d.target, d.source, d.target.width ?? NODE_WIDTH, d.target.height ?? 80)
        // A gentle arc, so two edges between the same pair of tables do not
        // sit exactly on top of each other.
        const dx = to.x - from.x
        const dy = to.y - from.y
        const radius = Math.hypot(dx, dy) * 1.8
        return `M${from.x},${from.y}A${radius},${radius} 0 0,1 ${to.x},${to.y}`
      })
      node.attr('transform', (d) => `translate(${d.x},${d.y})`)
    })

    node.call(
      d3drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.2).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          // Left pinned: a user who arranged the diagram meant it to stay.
          d.fx = event.x
          d.fy = event.y
        }),
    )

    svg.on('click', () => setPending(null))

    return () => {
      simulation.stop()
      simulationRef.current = null
    }
  }, [data, height, selectedId, expanded, pending, toggle, clickColumn])

  const resetLayout = () => {
    const simulation = simulationRef.current
    if (!simulation) return
    simulation.nodes().forEach((node) => {
      node.fx = null
      node.fy = null
    })
    simulation.alpha(0.9).restart()
    select(svgRef.current).transition().duration(300).call(d3zoom().transform, zoomIdentity)
  }

  if (!data.nodes.length) {
    return (
      <p className="py-10 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
        No tables to draw yet.
      </p>
    )
  }

  return (
    <div ref={containerRef} className="relative">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[11px]">
        <div className="flex flex-wrap items-center gap-3" style={{ color: 'var(--text-muted)' }}>
          <span>
            <span style={{ color: 'var(--series-3, #b8860b)' }}>◆</span> primary key
          </span>
          <span>
            <span style={{ color: 'var(--series-1)' }}>↗</span> foreign key
          </span>
          <span>— confirmed</span>
          <span>- - proposed</span>
          <span style={{ color: 'var(--status-warning)' }}>- - uncertain</span>
        </div>
        <button type="button" className="focus-ring underline" onClick={resetLayout}>
          reset layout
        </button>
      </div>

      {pending && (
        <p
          className="mb-2 rounded-md border px-2 py-1 text-[11px]"
          style={{ borderColor: 'var(--series-1)', color: 'var(--text-secondary)' }}
        >
          Drawing from <strong>{pending.table}.{pending.column}</strong> — click the key column it
          references, or click the background to cancel.
        </p>
      )}

      <svg
        ref={svgRef}
        width="100%"
        height={height}
        role="img"
        aria-label="Relationship diagram: tables as boxes, foreign keys as arrows"
        style={{ background: 'var(--surface-2)', borderRadius: 8, border: '1px solid var(--border)' }}
      />
      <p className="mt-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        Drag to rearrange · scroll to zoom · click an arrow for its evidence · click two columns to
        draw a relationship the detector missed
      </p>
    </div>
  )
}
