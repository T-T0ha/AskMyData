/** AskMyData — application shell.
 *
 *  Everything sits behind a sign-in: datasets belong to accounts, and the
 *  backend will not answer for one without knowing whose it is.
 *
 *  Eight stages, in the order the pipeline runs them:
 *
 *    Add data          → ingestion and structural repair (files or a database)
 *    Triage            → data quality buckets + cross-sheet equivalences
 *    Field semantics   → the editable field semantic view
 *    Co-planned cleaning → the agent's plan, approved step by step
 *    Relationships     → keys, references and dependencies, with evidence
 *    Semantic layer    → the exported database and its sem_metadata
 *    Ask questions     → natural language in, SQL and a chart out
 *    Dashboard         → pinned questions, re-run live, arranged on a grid
 *
 *  The stage names are the user's vocabulary, not the pipeline's: the internal
 *  phase numbers are a fact about the implementation and appear only in the
 *  source and the report, never in the interface.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, getToken, onSessionExpired, setToken } from './api'
import { AccountMenu, AuthPanel } from './components/AuthPanel'
import { EquivalencePanel } from './components/EquivalencePanel'
import { ExportPanel, download } from './components/ExportPanel'
import { EvidencePanel, RelationshipList } from './components/EvidencePanel'
import { RelationshipDiagram } from './components/RelationshipDiagram'
import { TableProfilePanel } from './components/TableProfilePanel'
import { FieldSemanticGrid } from './components/FieldSemanticGrid'
import { DashboardPanel } from './components/DashboardPanel'
import { PlanBoard } from './components/PlanBoard'
import { QualityReportPanel } from './components/QualityReportPanel'
import { QueryPanel } from './components/QueryPanel'
import { SessionSwitcher } from './components/SessionSwitcher'
import { SourcePicker } from './components/SourcePicker'
import { CleaningProgress, CleaningSummary, StepValidation } from './components/StepValidation'
import { TriageBoard } from './components/TriageBoard'
import { Button, EmptyState, ErrorBanner, Panel, Spinner, StatusBadge } from './components/ui'

const STAGES = [
  { key: 'upload', label: 'Add data' },
  { key: 'triage', label: 'Triage' },
  { key: 'semantics', label: 'Field semantics' },
  { key: 'cleaning', label: 'Co-planned cleaning' },
  { key: 'relationships', label: 'Relationships' },
  { key: 'export', label: 'Semantic layer' },
  { key: 'query', label: 'Ask questions' },
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'quality', label: 'Data quality' },
]

function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') ?? 'system')
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])
  return [theme, setTheme]
}

/** Which model, which embedder and which checkpoint store are in use are facts
 *  about the deployment, not about the user's data — a working system should
 *  not spend header space announcing that it is working.  What the user does
 *  need to know is when the system is running on less than it normally has,
 *  because the answers get worse: that is the Degradability requirement, and
 *  it is why this renders nothing at all until something is actually missing.
 */
function DegradedNotice({ status }) {
  if (!status) return null
  const reasons = []
  if (!status.claude.available) {
    reasons.push(
      `The language model is unavailable${
        status.claude.reason ? ` (${status.claude.reason})` : ''
      }, so labelling falls back to the rule engine and cleaning to the built-in statistical planner.`,
    )
  }
  if (!status.embeddings.semantic) {
    reasons.push(
      'Semantic embeddings are unavailable, so column-name matching falls back to string comparison and will miss synonyms.',
    )
  }
  if (status.durable === false) {
    reasons.push(
      'Cleaning progress is not being saved durably (no PostgreSQL or SQLite checkpoint store is reachable) — a server restart will lose any in-progress cleaning session.',
    )
  }
  if (!reasons.length) return null
  return (
    <StatusBadge status="warning" title={reasons.join(' ')}>
      Limited mode
    </StatusBadge>
  )
}

/** The last dataset a given account was looking at.  Namespaced by account so
 *  that signing in as somebody else on a shared machine does not open — or
 *  even name — the previous user's dataset. */
const lastDatasetKey = (userId) => `sessionId:${userId}`

export default function App() {
  const [theme, setTheme] = useTheme()
  const [status, setStatus] = useState(null)
  const [vocabulary, setVocabulary] = useState(null)
  const [user, setUser] = useState(null)
  const [authConfig, setAuthConfig] = useState(null)
  const [authChecked, setAuthChecked] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [session, setSession] = useState(null)
  //: Every dataset this account has ever created — the list behind the
  //: "Your databases" switcher. Kept separate from `session` (the one that's
  //: open) so switching between them never has to re-fetch the whole list.
  const [sessions, setSessions] = useState([])
  const [triage, setTriage] = useState(null)
  const [equivalences, setEquivalences] = useState([])
  const [semantics, setSemantics] = useState(null)
  const [cleaning, setCleaning] = useState(null)
  const [relationships, setRelationships] = useState(null)
  const [profiles, setProfiles] = useState(null)
  const [exportState, setExportState] = useState(null)
  //: Every question asked this session, newest first — a scrollback, not a
  //: persisted history: nothing here survives a reload. The server keeps its
  //: own copy (`queryHistory` below) for the dashboard's history sidebar,
  //: which does survive one.
  const [queryEntries, setQueryEntries] = useState([])
  const [dashboardCards, setDashboardCards] = useState([])
  const [queryHistory, setQueryHistory] = useState([])
  const [businessRules, setBusinessRules] = useState([])
  const [suggestedQuestions, setSuggestedQuestions] = useState([])
  const [qualityReport, setQualityReport] = useState(null)
  const [selectedEdge, setSelectedEdge] = useState(null)
  const [evidence, setEvidence] = useState(null)
  const [evidenceLoading, setEvidenceLoading] = useState(false)
  const [preview, setPreview] = useState(null)
  const [stage, setStage] = useState('upload')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  //: A file drop / "Choose file(s)" / database connect that is waiting on the
  //: user to say whether it belongs in the currently open session or in a new
  //: one — see `handleUpload`/`handleConnect` below.
  const [pendingSource, setPendingSource] = useState(null)

  const run = useCallback(async (task) => {
    setBusy(true)
    setError(null)
    try {
      return await task()
    } catch (err) {
      setError(err.message)
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  const resetWorkspace = useCallback(() => {
    setSessionId(null)
    setSession(null)
    setTriage(null)
    setEquivalences([])
    setSemantics(null)
    setCleaning(null)
    setRelationships(null)
    setProfiles(null)
    setExportState(null)
    setQueryEntries([])
    setDashboardCards([])
    setQueryHistory([])
    setSelectedEdge(null)
    setEvidence(null)
    setPreview(null)
    setStage('upload')
  }, [])

  useEffect(() => {
    api.status().then(setStatus).catch(() => setError('Backend is not reachable on :8000'))
    api.authConfig().then(setAuthConfig).catch(() => {})
  }, [])

  // Restore the signed-in account before rendering anything: a stored token
  // may have expired, been revoked, or belong to a deleted account.
  useEffect(() => {
    if (!getToken()) {
      setAuthChecked(true)
      return
    }
    api
      .me()
      .then((payload) => setUser(payload.user))
      .catch(() => setToken(null))
      .finally(() => setAuthChecked(true))
  }, [])

  // The vocabulary endpoint is public, but there is no reason to fetch the
  // dropdown contents for somebody who is not signed in.
  useEffect(() => {
    if (user) api.vocabulary().then(setVocabulary).catch(() => {})
  }, [user])

  const refreshSessions = useCallback(
    () => api.listSessions().then((payload) => setSessions(payload.sessions)).catch(() => {}),
    [],
  )

  useEffect(() => {
    if (user) refreshSessions()
  }, [user, refreshSessions])

  useEffect(
    () =>
      onSessionExpired(() => {
        setUser(null)
        resetWorkspace()
        setError('Your session expired — sign in again.')
      }),
    [resetWorkspace],
  )

  const loadSession = useCallback(
    async (id) => {
      if (!id) return
      // Asked for first, on its own: opening an older session is when its
      // identical copies are set aside, and everything fetched below should
      // describe the session as it is afterwards.
      const triageData = await api.triage(id)
      const [
        sessionData,
        equivalenceData,
        semanticsData,
        cleaningData,
        relationshipData,
        profileData,
        exportData,
      ] = await Promise.all([
        api.getSession(id),
        api.equivalences(id),
        api.getSemantics(id),
        api.cleaningState(id),
        api.relationships(id),
        api.profiles(id),
        api.exportState(id),
      ])
      setSession(sessionData)
      setTriage(triageData)
      setEquivalences(equivalenceData.equivalences)
      setSemantics(semanticsData)
      setCleaning(cleaningData)
      setRelationships(relationshipData)
      setProfiles(profileData)
      setExportState(exportData)
    },
    [],
  )

  // Reopen whatever this account was last working on.
  useEffect(() => {
    if (!user) return
    setSessionId(localStorage.getItem(lastDatasetKey(user.id)))
  }, [user])

  useEffect(() => {
    if (!user || !sessionId) return
    localStorage.setItem(lastDatasetKey(user.id), sessionId)
    run(async () => {
      try {
        await loadSession(sessionId)
      } catch (err) {
        if (err.status !== 404) throw err
        // Deleted, or created under a different account. Forget it quietly —
        // an error banner about an id the user never typed explains nothing.
        localStorage.removeItem(lastDatasetKey(user.id))
        resetWorkspace()
      }
      return true
    })
  }, [user, sessionId, loadSession, run, resetWorkspace])

  // Shared by the explicit "New session" button and by "start a new database
  // instead", the choice offered when a drop/connect would otherwise land
  // quietly in an already-populated open session (see `pendingSource`).
  const createAndOpenSession = async (name) => {
    const created = await api.createSession(name)
    setSessionId(created.id)
    setSession(created)
    setTriage(null)
    setEquivalences([])
    setSemantics(null)
    setCleaning(null)
    setQueryEntries([])
    setDashboardCards([])
    setQueryHistory([])
    setStage('upload')
    await refreshSessions()
    return created
  }

  const startSession = () =>
    run(() => createAndOpenSession(`Session ${new Date().toLocaleString()}`))

  const removeSession = (id) =>
    run(async () => {
      await api.deleteSession(id)
      await refreshSessions()
      if (id === sessionId) {
        if (user) localStorage.removeItem(lastDatasetKey(user.id))
        resetWorkspace()
      }
      return true
    })

  const signIn = async (email, password) => {
    const payload = await api.login(email, password)
    setToken(payload.access_token)
    setError(null)
    setUser(payload.user)
  }

  const registerAccount = async (email, password, name) => {
    const payload = await api.register(email, password, name)
    setToken(payload.access_token)
    setError(null)
    setUser(payload.user)
  }

  const signOut = () =>
    run(async () => {
      // Best effort: a token the server has already forgotten still has to
      // disappear from this browser.
      await api.logout().catch(() => {})
      setToken(null)
      setUser(null)
      resetWorkspace()
      return true
    })

  const changePassword = (current, next) => api.changePassword(current, next)

  const deleteAccount = async (password) => {
    await api.deleteAccount(password)
    if (user) localStorage.removeItem(lastDatasetKey(user.id))
    setToken(null)
    setUser(null)
    resetWorkspace()
  }

  // A 409 here means this exact file/database is already sitting in another
  // session of this account. Rather than error out, drop back into that
  // session — that is what "cannot be stored twice" should feel like from
  // the outside. `freshId` is deleted first when it was created just for
  // this attempt and nothing has landed in it yet (`safeToDelete`), so a
  // rejected upload never leaves an empty orphan session behind in the
  // switcher — but a session that already holds an earlier file from the
  // same batch is kept, because it has real data now.
  const handleDuplicateSource = async (err, freshId, safeToDelete) => {
    if (err.status !== 409 || !err.detail?.existing_session_id) throw err
    if (safeToDelete) await api.deleteSession(freshId).catch(() => {})
    await refreshSessions()
    setSessionId(err.detail.existing_session_id)
    setStage('triage')
    setError(err.message)
    return true
  }

  // One request per file: each is ingested, triaged and named on its own, and
  // a failure on the third file must not discard the first two. `freshlyCreated`
  // (this session exists solely for this call, nothing has landed in it yet)
  // is what lets a duplicate-source rejection safely delete it instead of
  // leaving an empty orphan in the switcher.
  const uploadFilesInto = async (id, files, freshlyCreated) => {
    let ingestedAny = false
    try {
      for (const file of files) {
        await api.upload(id, file)
        ingestedAny = true
      }
    } catch (err) {
      return handleDuplicateSource(err, id, freshlyCreated && !ingestedAny)
    }
    await loadSession(id)
    await refreshSessions()
    setStage('triage')
    return true
  }

  const connectDatabaseInto = async (id, url, tables, freshlyCreated) => {
    try {
      await api.connectDatabase(id, url, tables)
    } catch (err) {
      return handleDuplicateSource(err, id, freshlyCreated)
    }
    await loadSession(id)
    await refreshSessions()
    setStage('triage')
    return true
  }

  // A session that already holds tables is somebody's real, named dataset —
  // silently folding an unrelated upload into it is how two databases "merge"
  // without anyone choosing that. Only an empty (just-created or never-used)
  // open session gets a source attached without asking first.
  const handleUpload = (files) => {
    const chosen = (Array.isArray(files) ? files : [files]).filter(Boolean)
    if (!chosen.length) return null
    if (sessionId && session?.sheets?.length > 0) {
      setPendingSource({ type: 'files', files: chosen })
      return null
    }
    const wasFresh = !sessionId
    return run(async () => {
      const id = sessionId ?? (await createAndOpenSession(chosen[0].name)).id
      return uploadFilesInto(id, chosen, wasFresh)
    })
  }

  const inspectSource = (url) => run(() => api.inspectSource(url))

  const handleConnect = (url, tables) => {
    if (sessionId && session?.sheets?.length > 0) {
      setPendingSource({ type: 'database', url, tables })
      return null
    }
    const wasFresh = !sessionId
    return run(async () => {
      // Never name a session after the raw URL: it carries the password.
      const id = sessionId ?? (await createAndOpenSession(`Database ${url.split('@').pop()}`)).id
      return connectDatabaseInto(id, url, tables, wasFresh)
    })
  }

  const confirmAddToSession = () => {
    const pending = pendingSource
    setPendingSource(null)
    if (!pending) return null
    return run(() =>
      pending.type === 'files'
        ? uploadFilesInto(sessionId, pending.files, false)
        : connectDatabaseInto(sessionId, pending.url, pending.tables, false),
    )
  }

  const confirmStartNewSession = () => {
    const pending = pendingSource
    setPendingSource(null)
    if (!pending) return null
    return run(async () => {
      const name =
        pending.type === 'files'
          ? pending.files[0].name
          : `Database ${pending.url.split('@').pop()}`
      const created = await createAndOpenSession(name)
      return pending.type === 'files'
        ? uploadFilesInto(created.id, pending.files, true)
        : connectDatabaseInto(created.id, pending.url, pending.tables, true)
    })
  }

  const cancelPendingSource = () => setPendingSource(null)

  const runSemantics = () =>
    run(async () => {
      const result = await api.runSemantics(sessionId)
      setSemantics(result)
      setStage('semantics')
      return result
    })

  const overrideColumn = (columnId, patch) =>
    run(async () => {
      await api.overrideColumn(sessionId, columnId, patch)
      setSemantics(await api.getSemantics(sessionId))
      return true
    })

  const decideKey = (table, columns) =>
    run(async () => {
      await api.decideKey(sessionId, table, columns)
      setTriage(await api.triage(sessionId))
      return true
    })

  const decideEquivalence = (candidateId, confirmed) =>
    run(async () => {
      await api.decideEquivalence(sessionId, candidateId, confirmed)
      const refreshed = await api.equivalences(sessionId)
      setEquivalences(refreshed.equivalences)
      return true
    })

  const startCleaning = () =>
    run(async () => {
      const state = await api.startCleaning(sessionId)
      setCleaning(state)
      setStage('cleaning')
      return state
    })

  const submitPlan = (action, plan, autoApprove = false) =>
    run(async () => {
      setCleaning(await api.submitPlan(sessionId, action, plan, autoApprove))
      return true
    })

  const submitStep = (action, params) =>
    run(async () => {
      setCleaning(await api.submitStep(sessionId, action, params))
      return true
    })

  const detectRelationships = () =>
    run(async () => {
      const payload = await api.detectRelationships(sessionId)
      setRelationships(payload)
      // Detection ends with table profiling; read the counts back rather than
      // deriving them from the profiles the response inlines.
      setProfiles(await api.profiles(sessionId))
      setStage('relationships')
      return payload
    })

  // Evidence reads whole tables, so it is fetched when a relationship is
  // actually opened rather than for all of them up front.
  const selectEdge = (relationshipId) =>
    run(async () => {
      if (selectedEdge === relationshipId) {
        setSelectedEdge(null)
        setEvidence(null)
        return null
      }
      setSelectedEdge(relationshipId)
      setEvidence(null)
      setEvidenceLoading(true)
      try {
        const payload = await api.relationshipEvidence(sessionId, relationshipId)
        setEvidence(payload.evidence)
      } finally {
        setEvidenceLoading(false)
      }
      return true
    })

  const decideRelationship = (relationshipId, confirmed) =>
    run(async () => {
      await api.decideRelationship(sessionId, relationshipId, confirmed)
      const [next, nextProfiles] = await Promise.all([
        api.relationships(sessionId),
        api.profiles(sessionId),
      ])
      setRelationships(next)
      setProfiles(nextProfiles)
      return true
    })

  const drawRelationship = (edge) =>
    run(async () => {
      const created = await api.drawRelationship(sessionId, edge)
      const [next, nextProfiles] = await Promise.all([
        api.relationships(sessionId),
        api.profiles(sessionId),
      ])
      setRelationships(next)
      setProfiles(nextProfiles)
      setSelectedEdge(created.id)
      setEvidence(null)
      return true
    })

  const runExport = () =>
    run(async () => {
      const payload = await api.runExport(sessionId)
      setExportState({ exported: true, ...payload })
      setStage('export')
      return payload
    })

  // Every download goes through the API rather than being rebuilt here: what
  // the user takes away has to be the export that was made, not a fresh
  // rendering of an analysis that may have moved on since.
  const downloadBundle = () =>
    run(async () => {
      const payload = await api.exportBundle(sessionId)
      download(`${session?.name ?? 'semantic-layer'}.json`, JSON.stringify(payload, null, 2), 'application/json')
      return true
    })

  const downloadDdl = () =>
    run(async () => {
      const script = await api.exportDdl(sessionId)
      download(`${session?.name ?? 'semantic-layer'}.sql`, script, 'application/sql')
      return true
    })

  // For most of these datasets this is the first documentation they have ever
  // had, so it is a file the user keeps rather than a screen they close.
  const downloadDocs = () =>
    run(async () => {
      const document = await api.exportDocs(sessionId)
      download(`${session?.name ?? 'semantic-layer'}.md`, document, 'text/markdown')
      return true
    })

  const refreshProfiles = () =>
    run(async () => {
      setProfiles(await api.runProfiles(sessionId))
      return true
    })

  // A question the model could not answer is a normal result (`ok: false`),
  // not a thrown error — it is appended to the scrollback exactly like a
  // successful one, so `run()`'s shared error banner stays for genuine
  // failures (no export yet, the network is down).
  const askQuestion = (question) =>
    run(async () => {
      const result = await api.askQuestion(sessionId, question)
      setQueryEntries((entries) => [
        { id: `${Date.now()}-${entries.length}`, question, result },
        ...entries,
      ])
      return result
    })

  // A pin is echoed straight back from the answer already on screen — no
  // second round trip to Claude or to the database is needed to know what a
  // card should show; the backend re-validates the SQL independently before
  // storing it (§Phase 6). `run()`'s shared error banner covers a rejection
  // (e.g. the export changed shape since the answer was given).
  const pinCard = (entry) =>
    run(async () => {
      await api.pinCard(sessionId, {
        question: entry.question,
        sql: entry.result.sql,
        title: entry.question,
        explanation: entry.result.explanation,
        tables_used: entry.result.tables_used,
        visualization: entry.result.visualization,
      })
      return true
    })

  const refreshDashboard = ({ start, end } = {}) =>
    run(async () => {
      const [cardsPayload, historyPayload] = await Promise.all([
        api.dashboardCards(sessionId, { start, end }),
        api.queryHistory(sessionId),
      ])
      setDashboardCards(cardsPayload.cards)
      setQueryHistory(historyPayload.history)
      return true
    })

  // Business rules and proactive suggestions load together — both are read
  // once when the "Ask questions" panel opens, not on every keystroke.
  const loadQueryExtras = () =>
    run(async () => {
      const [rulesPayload, suggestionsPayload] = await Promise.all([
        api.businessRules(sessionId),
        api.suggestedQuestions(sessionId),
      ])
      setBusinessRules(rulesPayload.rules)
      setSuggestedQuestions(suggestionsPayload.questions)
      return true
    })

  const addRule = (ruleText) =>
    run(async () => {
      const rule = await api.addBusinessRule(sessionId, ruleText)
      setBusinessRules((rules) => [rule, ...rules])
      return true
    })

  const deleteRule = (ruleId) =>
    run(async () => {
      await api.deleteBusinessRule(sessionId, ruleId)
      setBusinessRules((rules) => rules.filter((rule) => rule.id !== ruleId))
      return true
    })

  // Saving an example re-validates the SQL server-side (§Phase 5's third
  // retrieval index) — there is nothing to reflect back into query state,
  // only a confirmation that the next relevant question will see it.
  //
  // It is saved under the *standalone* question the server resolved ("and by
  // month?" is meaningless without the turns before it), and also under the
  // vaguer question the user settled by choosing a reading — so asking that
  // again lands on this SQL instead of being asked about all over again.
  const saveExample = (entry) =>
    run(async () => {
      const { result } = entry
      await api.saveExample(sessionId, {
        question: result.resolved_question || entry.question,
        sql: result.sql,
        explanation: result.explanation,
        alsoAskedAs: result.clarifies ? [result.clarifies] : [],
      })
      return true
    })

  const refreshQualityReport = () =>
    run(async () => {
      setQualityReport(await api.qualityReport(sessionId))
      return true
    })

  const renameCard = (cardId, title) =>
    run(async () => {
      await api.updateCard(sessionId, cardId, { title })
      setDashboardCards((cards) => cards.map((c) => (c.id === cardId ? { ...c, title } : c)))
      return true
    })

  const moveCard = (cardId, layout) =>
    run(async () => {
      await api.updateCard(sessionId, cardId, { layout })
      setDashboardCards((cards) => cards.map((c) => (c.id === cardId ? { ...c, layout } : c)))
      return true
    })

  const unpinCard = (cardId) =>
    run(async () => {
      await api.unpinCard(sessionId, cardId)
      setDashboardCards((cards) => cards.filter((c) => c.id !== cardId))
      return true
    })

  // A history entry is re-asked exactly like a fresh question — the data may
  // have changed since, so replaying stored SQL would be a smaller-scoped
  // (and quietly stale) thing to call "re-run" than actually asking again.
  const rerunFromHistory = (question) => {
    setStage('query')
    askQuestion(question)
  }

  const correctProfile = (table, patch) =>
    run(async () => {
      await api.correctProfile(sessionId, table, patch)
      setProfiles(await api.profiles(sessionId))
      return true
    })

  const showPreview = (table) =>
    run(async () => {
      if (preview?.table === table) {
        setPreview(null)
        return null
      }
      setPreview(await api.preview(sessionId, table))
      return true
    })

  const tableNames = useMemo(
    () => (cleaning?.tables ?? session?.tables ?? []).map((table) => table.name),
    [cleaning, session],
  )

  const stageAvailable = {
    upload: true,
    triage: Boolean(triage?.sheets?.length),
    semantics: Boolean(triage?.sheets?.length),
    cleaning: Boolean(triage?.sheets?.length),
    relationships: Boolean(triage?.sheets?.length),
    export: Boolean(triage?.sheets?.length),
    query: Boolean(triage?.sheets?.length),
    dashboard: Boolean(triage?.sheets?.length),
    quality: Boolean(triage?.sheets?.length),
  }

  const selectedRelationship =
    relationships?.relationships?.find((row) => row.id === selectedEdge) ?? null

  if (!authChecked) {
    return (
      <div className="flex min-h-full items-center justify-center">
        <Spinner label="Restoring your session…" />
      </div>
    )
  }

  if (!user) {
    return (
      <div className="min-h-full">
        {error && (
          <div className="mx-auto max-w-md px-6 pt-6">
            <ErrorBanner error={error} onDismiss={() => setError(null)} />
          </div>
        )}
        <AuthPanel
          config={authConfig}
          backendReachable={Boolean(status)}
          busy={busy}
          onSignIn={signIn}
          onRegister={registerAccount}
          theme={theme}
          onThemeChange={setTheme}
        />
      </div>
    )
  }

  return (
    <div className="min-h-full">
      {/* z-30 clears the z-20 chart tooltips in the main column, which would
          otherwise draw over a header the page can now be scrolled under. */}
      <header
        className="no-print sticky top-0 z-30 border-b"
        style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
      >
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-6 py-3">
          <div className="flex min-w-0 items-baseline gap-2.5">
            <h1 className="shrink-0 text-[15px] font-semibold tracking-tight">AskMyData</h1>
            {session?.name && (
              <>
                <span aria-hidden="true" style={{ color: 'var(--border)' }}>
                  /
                </span>
                <span className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>
                  {session.name}
                </span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2">
            <DegradedNotice status={status} />
            <select
              className="focus-ring cursor-pointer rounded-md border-0 bg-transparent py-1 pl-2 pr-1 text-xs"
              style={{ color: 'var(--text-muted)' }}
              value={theme}
              onChange={(event) => setTheme(event.target.value)}
              aria-label="Colour theme"
            >
              <option value="system">Auto</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
            <SessionSwitcher
              sessions={sessions}
              activeId={sessionId}
              busy={busy}
              onOpen={refreshSessions}
              onSelect={setSessionId}
              onDelete={removeSession}
            />
            <Button variant="primary" onClick={startSession} disabled={busy}>
              New session
            </Button>
            <AccountMenu
              user={user}
              busy={busy}
              onSignOut={signOut}
              onChangePassword={changePassword}
              onDeleteAccount={deleteAccount}
            />
          </div>
        </div>

        <nav className="mx-auto flex max-w-7xl gap-0.5 overflow-x-auto px-6">
          {STAGES.map((item) => {
            const active = stage === item.key
            return (
              <button
                key={item.key}
                type="button"
                disabled={!stageAvailable[item.key]}
                onClick={() => setStage(item.key)}
                aria-current={active ? 'page' : undefined}
                className={`focus-ring relative shrink-0 rounded-t-md px-3 py-2.5 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-35 ${
                  active ? '' : 'hover:bg-[var(--surface-3)]'
                }`}
                style={{ color: active ? 'var(--text-primary)' : 'var(--text-secondary)' }}
              >
                {item.label}
                {active && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-2 -bottom-px h-0.5 rounded-full"
                    style={{ background: 'var(--series-1)' }}
                  />
                )}
              </button>
            )
          })}
        </nav>
      </header>

      <main className="mx-auto max-w-7xl space-y-4 px-6 py-6">
        <ErrorBanner error={error} onDismiss={() => setError(null)} />
        {busy && <Spinner />}

        {stage === 'upload' && (
          <SourcePicker
            vocabulary={vocabulary}
            session={session}
            busy={busy}
            onUpload={handleUpload}
            onInspect={inspectSource}
            onConnect={handleConnect}
            pendingSource={pendingSource}
            onConfirmAdd={confirmAddToSession}
            onConfirmNewSession={confirmStartNewSession}
            onCancelPending={cancelPendingSource}
          />
        )}

        {stage === 'triage' && (
          <>
            <TriageBoard
              triage={triage}
              onPreview={showPreview}
              preview={preview}
              onDecideKey={decideKey}
              busy={busy}
            />
            <EquivalencePanel
              equivalences={equivalences}
              onDecide={decideEquivalence}
              busy={busy}
            />
            <div className="flex justify-end">
              <Button variant="primary" disabled={busy} onClick={runSemantics}>
                Analyse field semantics →
              </Button>
            </div>
          </>
        )}

        {stage === 'semantics' && (
          <>
            {semantics?.tables?.length ? (
              <FieldSemanticGrid
                semantics={semantics}
                vocabulary={vocabulary ?? { column_types: [], taxonomy_labels: [] }}
                onOverride={overrideColumn}
                busy={busy}
              />
            ) : (
              <Panel title="Field semantics">
                <EmptyState>
                  Nothing analysed yet.
                  <span className="ml-2">
                    <Button variant="primary" disabled={busy} onClick={runSemantics}>
                      Run analysis
                    </Button>
                  </span>
                </EmptyState>
              </Panel>
            )}
            {semantics?.tables?.length > 0 && (
              <div className="flex justify-end gap-2">
                <Button disabled={busy} onClick={runSemantics}>
                  Re-run detection
                </Button>
                <Button variant="primary" disabled={busy} onClick={startCleaning}>
                  Check for problems to tidy →
                </Button>
              </div>
            )}
          </>
        )}

        {stage === 'cleaning' && (
          <>
            <CleaningStage
              cleaning={cleaning}
              vocabulary={vocabulary}
              tableNames={tableNames}
              busy={busy}
              onStart={startCleaning}
              onSubmitPlan={submitPlan}
              onSubmitStep={submitStep}
            />
            {['completed', 'cancelled'].includes(cleaning?.status) && (
              <div className="flex justify-end">
                <Button variant="primary" disabled={busy} onClick={detectRelationships}>
                  Find how my tables connect →
                </Button>
              </div>
            )}
          </>
        )}

        {stage === 'relationships' && (
          <RelationshipStage
            relationships={relationships}
            selectedId={selectedEdge}
            selectedRelationship={selectedRelationship}
            evidence={evidence}
            evidenceLoading={evidenceLoading}
            busy={busy}
            onDetect={detectRelationships}
            onSelect={selectEdge}
            onDecide={decideRelationship}
            onDraw={drawRelationship}
            profiles={profiles}
            tableTypes={vocabulary?.table_types ?? []}
            onCorrectProfile={correctProfile}
            onRefreshProfiles={refreshProfiles}
            onExport={runExport}
          />
        )}

        {stage === 'export' && (
          <ExportPanel
            state={exportState}
            busy={busy}
            onExport={runExport}
            onDownloadBundle={downloadBundle}
            onDownloadDdl={downloadDdl}
            onDownloadDocs={downloadDocs}
          />
        )}

        {stage === 'query' && (
          <QueryPanel
            exported={Boolean(exportState?.exported)}
            entries={queryEntries}
            busy={busy}
            onAsk={askQuestion}
            onPin={pinCard}
            rules={businessRules}
            suggestions={suggestedQuestions}
            onLoadExtras={loadQueryExtras}
            onAddRule={addRule}
            onDeleteRule={deleteRule}
            onSaveExample={saveExample}
          />
        )}

        {stage === 'dashboard' && (
          <DashboardPanel
            exported={Boolean(exportState?.exported)}
            cards={dashboardCards}
            history={queryHistory}
            busy={busy}
            onRefresh={refreshDashboard}
            onRename={renameCard}
            onUnpin={unpinCard}
            onLayoutChange={moveCard}
            onRerun={rerunFromHistory}
          />
        )}

        {stage === 'quality' && (
          <QualityReportPanel
            exported={Boolean(exportState?.exported)}
            report={qualityReport}
            busy={busy}
            onRefresh={refreshQualityReport}
          />
        )}
      </main>
    </div>
  )
}

/** Patterns inside one table, read-only: each describes the data, none is a decision. */
function TablePatterns({ patterns }) {
  if (!patterns.length) return null
  const byTable = {}
  patterns.forEach((row) => {
    ;(byTable[row.from_table] ??= []).push(row)
  })
  return (
    <details className="panel">
      <summary className="cursor-pointer px-4 py-3 text-sm font-semibold">
        Patterns inside single tables ({patterns.length})
      </summary>
      <div className="border-t px-4 py-3" style={{ borderColor: 'var(--border)' }}>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          These describe your data — for example, that every district code goes with one county.
          They do not connect tables, so there is nothing to approve.
        </p>
        {Object.entries(byTable).map(([table, rows]) => (
          <div key={table} className="mt-3">
            <p className="text-xs font-medium">{table}</p>
            <ul className="mt-1 space-y-0.5 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              {rows.slice(0, 8).map((row) => (
                <li key={row.id}>
                  · each <code>{row.from_column}</code> goes with{' '}
                  {row.rel_type === 'functional_dependency' ? 'exactly one' : 'almost always one'}{' '}
                  <code>{row.to_column}</code>
                </li>
              ))}
              {rows.length > 8 && <li>· and {rows.length - 8} more</li>}
            </ul>
          </div>
        ))}
      </div>
    </details>
  )
}

function RelationshipStage({
  relationships,
  selectedId,
  selectedRelationship,
  evidence,
  evidenceLoading,
  busy,
  onDetect,
  onSelect,
  onDecide,
  onDraw,
  profiles,
  tableTypes,
  onCorrectProfile,
  onRefreshProfiles,
  onExport,
}) {
  if (!relationships || !relationships.counts?.total) {
    return (
      <Panel
        title="How your tables connect"
        subtitle="Finds the values that link one table to another — a customer number in an orders table that points to a row in the customers table, for example."
      >
        <EmptyState>
          <span className="mr-2">Not looked for yet.</span>
          <Button variant="primary" disabled={busy} onClick={onDetect}>
            Find how my tables connect
          </Button>
        </EmptyState>
      </Panel>
    )
  }

  const rows = relationships.relationships ?? []
  const links = rows.filter((row) => row.rel_type === 'foreign_key')
  const patterns = rows.filter((row) => row.rel_type !== 'foreign_key')
  const needsCheck = links.filter((row) => row.status === 'proposed')
  const linked = links.filter((row) => row.status === 'confirmed')
  const ruledOut = links.filter((row) => row.status === 'rejected')
  const listProps = { selectedId, onSelect, busy }

  return (
    <div className="space-y-4">
      <Panel
        title="How your tables connect"
        subtitle={
          links.length === 0
            ? 'No links between your tables were found.'
            : `${linked.length} link${linked.length === 1 ? '' : 's'} between your tables${
                needsCheck.length ? ` · ${needsCheck.length} for you to check` : ' · nothing needs your attention'
              }`
        }
        actions={
          <Button disabled={busy} onClick={onDetect}>
            Look again
          </Button>
        }
      >
        <p className="mb-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
          A link means a value in one table picks out a row in another. The clear ones are made for
          you; click any to see why, or to undo it. You can also draw a link between two columns in
          the diagram.
        </p>
        <RelationshipDiagram
          graph={relationships.graph}
          selectedId={selectedId}
          onSelectEdge={onSelect}
          onDrawEdge={onDraw}
        />
        {relationships.skipped_tables?.length > 0 && (
          <p className="mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            Not searched for links: {relationships.skipped_tables.join(', ')} — with that few rows,
            values land inside another column&apos;s by coincidence often enough that any verdict
            would be noise.
          </p>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
        <Panel title="Links between tables">
          <div className="space-y-4">
            {needsCheck.length > 0 && (
              <div>
                <p className="mb-1 text-xs font-medium" style={{ color: 'var(--status-warning)' }}>
                  Please check ({needsCheck.length})
                </p>
                <RelationshipList relationships={needsCheck} {...listProps} />
              </div>
            )}
            <div>
              {needsCheck.length > 0 && (
                <p className="mb-1 text-xs font-medium">Linked ({linked.length})</p>
              )}
              <RelationshipList
                relationships={linked}
                {...listProps}
                emptyMessage="No links were made. Draw one in the diagram if you know two tables are connected."
              />
            </div>
            {ruledOut.length > 0 && (
              <details>
                <summary className="cursor-pointer text-xs font-medium">
                  Ruled out ({ruledOut.length})
                </summary>
                <div className="mt-1">
                  <RelationshipList relationships={ruledOut} {...listProps} />
                </div>
              </details>
            )}
          </div>
        </Panel>
        <EvidencePanel
          relationship={selectedRelationship}
          evidence={evidence}
          loading={evidenceLoading}
          busy={busy}
          onDecide={onDecide}
          onClose={() => onSelect(selectedId)}
        />
      </div>

      <TablePatterns patterns={patterns} />

      <TableProfilePanel
        profiles={profiles?.profiles}
        counts={profiles?.counts}
        tableTypes={tableTypes}
        busy={busy}
        onCorrect={onCorrectProfile}
        onRefresh={onRefreshProfiles}
      />

      <div className="flex justify-end">
        <Button variant="primary" disabled={busy} onClick={onExport}>
          Build the database →
        </Button>
      </div>
    </div>
  )
}

function CleaningStage({ cleaning, vocabulary, tableNames, busy, onStart, onSubmitPlan, onSubmitStep }) {
  if (!cleaning || cleaning.status === 'not_started') {
    return (
      <Panel
        title="Tidy up your data"
        subtitle="We look for common problems — extra spaces, repeated rows, empty columns — and suggest fixes. You choose which to apply; nothing changes until you do."
      >
        <EmptyState>
          <span className="mr-2">Not checked yet.</span>
          <Button variant="primary" disabled={busy} onClick={onStart}>
            Look for problems to tidy
          </Button>
        </EmptyState>
      </Panel>
    )
  }

  const pending = cleaning.pending
  const plan = cleaning.plan ?? []

  if (pending?.kind === 'plan_review') {
    return (
      <PlanBoard
        plan={pending.plan ?? plan}
        planSource={cleaning.plan_source}
        rejections={cleaning.plan_rejections}
        tables={tableNames}
        stepTypes={vocabulary?.step_types ?? []}
        onSubmit={onSubmitPlan}
        busy={busy}
      />
    )
  }

  if (pending) {
    return (
      <div className="space-y-4">
        <CleaningProgress plan={plan} cursor={cleaning.cursor ?? 0} />
        <StepValidation
          pending={pending}
          progress={{ total: plan.length }}
          onDecide={onSubmitStep}
          busy={busy}
        />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <CleaningProgress plan={plan} cursor={plan.length} />
      <CleaningSummary state={cleaning} onRestart={onStart} />
    </div>
  )
}
