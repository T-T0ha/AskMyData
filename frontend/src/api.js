/** Thin wrapper over the backend API. Every call funnels through `request`
 *  so error handling, JSON parsing and the bearer token live in exactly one
 *  place. */

const BASE = '/api'
const TOKEN_KEY = 'accessToken'

let token = localStorage.getItem(TOKEN_KEY)
const expiryListeners = new Set()

/** Called with no argument to sign out. */
export function setToken(next) {
  token = next ?? null
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
}

export function getToken() {
  return token
}

/** Notified when the backend rejects the stored token — expired, revoked, or
 *  signed out from another device. The shell uses it to return to the sign-in
 *  screen rather than showing an error on every panel. */
export function onSessionExpired(listener) {
  expiryListeners.add(listener)
  return () => expiryListeners.delete(listener)
}

class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, options = {}) {
  const headers = options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }
  if (token) headers.Authorization = `Bearer ${token}`

  const response = await fetch(`${BASE}${path}`, { ...options, headers: { ...headers, ...options.headers } })

  if (response.status === 401 && token) {
    // The token we held is no longer good for anything; drop it before any
    // caller can retry with it.
    setToken(null)
    expiryListeners.forEach((listener) => listener())
  }

  if (response.status === 204) return null
  const text = await response.text()
  const payload = text ? JSON.parse(text) : null
  if (!response.ok) {
    const detail = payload?.detail
    throw new ApiError(
      typeof detail === 'string' ? detail : `${response.status} ${response.statusText}`,
      response.status,
    )
  }
  return payload
}

/** Plain-text responses (the DDL script) — `request` parses JSON and would
 *  throw on a CREATE TABLE statement. */
async function requestText(path) {
  const headers = {}
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetch(`${BASE}${path}`, { headers })
  const body = await response.text()
  if (!response.ok) throw new ApiError(`${response.status} ${response.statusText}`, response.status)
  return body
}

export const api = {
  status: () => request('/status'),
  vocabulary: () => request('/vocabulary'),

  // ---- accounts ----------------------------------------------------------
  authConfig: () => request('/auth/config'),
  register: (email, password, name) =>
    request('/auth/register', { method: 'POST', body: JSON.stringify({ email, password, name }) }),
  login: (email, password) =>
    request('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  logout: () => request('/auth/logout', { method: 'POST' }),
  me: () => request('/auth/me'),
  changePassword: (currentPassword, newPassword) =>
    request('/auth/password', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    }),
  deleteAccount: (password) =>
    request('/auth/delete', { method: 'POST', body: JSON.stringify({ password }) }),

  listSessions: () => request('/sessions'),
  createSession: (name) => request('/sessions', { method: 'POST', body: JSON.stringify({ name }) }),
  getSession: (id) => request(`/sessions/${id}`),
  deleteSession: (id) => request(`/sessions/${id}`, { method: 'DELETE' }),

  upload: (id, file) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/sessions/${id}/upload`, { method: 'POST', body: form })
  },
  inspectSource: (url) =>
    request('/sources/inspect', { method: 'POST', body: JSON.stringify({ url }) }),
  connectDatabase: (id, url, tables) =>
    request(`/sessions/${id}/connect`, {
      method: 'POST',
      body: JSON.stringify({ url, tables: tables?.length ? tables : null }),
    }),

  triage: (id) => request(`/sessions/${id}/triage`),
  // An empty `columns` list is a real answer — "none of these is the key".
  decideKey: (id, table, columns) =>
    request(`/sessions/${id}/tables/${table}/key`, {
      method: 'POST',
      body: JSON.stringify({ columns }),
    }),
  preview: (id, table, limit = 25) => request(`/sessions/${id}/tables/${table}/preview?limit=${limit}`),

  equivalences: (id) => request(`/sessions/${id}/equivalences`),
  decideEquivalence: (id, candidateId, confirmed) =>
    request(`/sessions/${id}/equivalences/${candidateId}`, {
      method: 'POST',
      body: JSON.stringify({ confirmed }),
    }),

  runSemantics: (id) => request(`/sessions/${id}/semantics`, { method: 'POST' }),
  getSemantics: (id) => request(`/sessions/${id}/semantics`),
  overrideColumn: (id, columnId, patch) =>
    request(`/sessions/${id}/semantics/${columnId}`, { method: 'PATCH', body: JSON.stringify(patch) }),

  detectRelationships: (id) => request(`/sessions/${id}/relationships`, { method: 'POST' }),
  relationships: (id) => request(`/sessions/${id}/relationships`),
  decideRelationship: (id, relationshipId, confirmed) =>
    request(`/sessions/${id}/relationships/${relationshipId}`, {
      method: 'POST',
      body: JSON.stringify({ confirmed }),
    }),
  relationshipEvidence: (id, relationshipId) =>
    request(`/sessions/${id}/relationships/${relationshipId}/evidence`),
  drawRelationship: (id, edge) =>
    request(`/sessions/${id}/relationships/manual`, { method: 'POST', body: JSON.stringify(edge) }),

  runProfiles: (id) => request(`/sessions/${id}/profiles`, { method: 'POST' }),
  profiles: (id) => request(`/sessions/${id}/profiles`),
  // `table_type: ''` withdraws an override rather than setting one.
  correctProfile: (id, table, patch) =>
    request(`/sessions/${id}/tables/${table}/profile`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  runExport: (id) => request(`/sessions/${id}/export`, { method: 'POST' }),
  exportState: (id) => request(`/sessions/${id}/export`),
  exportBundle: (id) => request(`/sessions/${id}/export/bundle`),
  exportDdl: (id) => requestText(`/sessions/${id}/export/ddl`),
  exportDocs: (id, version) =>
    requestText(
      `/sessions/${id}/export/documentation${version ? `?version=${version}` : ''}`,
    ),
  exportVersions: (id) => request(`/sessions/${id}/export/versions`),

  askQuestion: (id, question) =>
    request(`/sessions/${id}/query`, { method: 'POST', body: JSON.stringify({ question }) }),
  queryHistory: (id) => request(`/sessions/${id}/query/history`),

  pinCard: (id, card) =>
    request(`/sessions/${id}/dashboard/cards`, { method: 'POST', body: JSON.stringify(card) }),
  dashboardCards: (id, { start, end } = {}) => {
    const params = new URLSearchParams()
    if (start) params.set('start', start)
    if (end) params.set('end', end)
    const query = params.toString()
    return request(`/sessions/${id}/dashboard/cards${query ? `?${query}` : ''}`)
  },
  updateCard: (id, cardId, patch) =>
    request(`/sessions/${id}/dashboard/cards/${cardId}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),
  unpinCard: (id, cardId) =>
    request(`/sessions/${id}/dashboard/cards/${cardId}`, { method: 'DELETE' }),

  startCleaning: (id) => request(`/sessions/${id}/cleaning/start`, { method: 'POST' }),
  cleaningState: (id) => request(`/sessions/${id}/cleaning/state`),
  submitPlan: (id, action, plan) =>
    request(`/sessions/${id}/cleaning/plan`, { method: 'POST', body: JSON.stringify({ action, plan }) }),
  submitStep: (id, action, params) =>
    request(`/sessions/${id}/cleaning/step`, { method: 'POST', body: JSON.stringify({ action, params }) }),
}
