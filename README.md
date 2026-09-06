# AskMyData

A human-in-the-loop semantic enrichment platform for reliable NL2SQL — upload
a messy spreadsheet or database, clean it with a human approving every step,
and then ask it questions in plain English. Built on two CHI '26 papers:
SemTabla (semantic enrichment / table profiling) and Cocoa (human-agent
co-planning).

## Phases

```
0. Ingestion        Multi-format loading, header repair, value cleaning,
                     cross-sheet equivalence detection, key discovery
1. Semantics         Column type + taxonomy classification, table profiling
2. Cleaning          LangGraph co-planned cleaning (propose → review →
                     execute → validate, human-in-the-loop throughout)
3. Relationships     Primary/foreign key and functional-dependency detection
4. Export            Materialization to a per-dataset PostgreSQL schema,
                      with an embedded semantic layer (sem_metadata)
5. NL2SQL query      Schema + column retrieval, SELECT-only guard,
                      generate → validate → EXPLAIN with self-correction
6. Dashboard         Pin questions as live-refreshing cards, question
                      history, follow-up suggestions
```

Each phase is implemented and tested end to end; see `claude.md` for the full
technical spec and `docs/phase4-plan.md`, `docs/phase5-plan.md`,
`docs/phase6-plan.md` for how Phases 4-6 were built.

Author: Tamim Hasan Toha