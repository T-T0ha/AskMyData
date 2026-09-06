/** Sign-in screen and the header's account menu.
 *
 *  The whole application sits behind this: datasets are owner-scoped on the
 *  backend, so there is nothing to show a caller the server will not identify.
 */

import { useState } from 'react'
import { Button, ErrorBanner, Panel, StatusBadge } from './ui'

function Field({ label, hint, ...props }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </span>
      <input
        className="focus-ring w-full rounded-md border px-3 py-1.5 text-sm"
        style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
        {...props}
      />
      {hint && (
        <span className="mt-1 block text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {hint}
        </span>
      )}
    </label>
  )
}

export function AuthPanel({ config, backendReachable, busy, onSignIn, onRegister, theme, onThemeChange }) {
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState(null)

  const registering = mode === 'register'
  const minimum = config?.min_password_length ?? 10
  const registrationOpen = config?.registration_open !== false

  const submit = async (event) => {
    event.preventDefault()
    setError(null)
    try {
      if (registering) await onRegister(email, password, name)
      else await onSignIn(email, password)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="mx-auto max-w-md py-16">
      {onThemeChange && (
        <div className="mb-4 flex justify-end">
          <select
            className="focus-ring cursor-pointer rounded-md border-0 bg-transparent py-1 pl-2 pr-1 text-xs"
            style={{ color: 'var(--text-muted)' }}
            value={theme}
            onChange={(event) => onThemeChange(event.target.value)}
            aria-label="Colour theme"
          >
            <option value="system">Auto</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>
      )}
      <div className="mb-6 text-center">
        <h1 className="text-xl font-semibold tracking-tight">AskMyData</h1>
        <p className="mt-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
          Upload a spreadsheet, clean it with a human in the loop, and ask it questions in plain English.
        </p>
      </div>

      <Panel
        title={registering ? 'Create an account' : 'Sign in'}
        subtitle={
          registering
            ? 'Create an account to start uploading and querying your datasets.'
            : 'Sign in to access your datasets.'
        }
      >
        {!backendReachable && (
          <p className="mb-3">
            <StatusBadge status="critical">Backend unreachable on :8000</StatusBadge>
          </p>
        )}

        <form className="space-y-3" onSubmit={submit}>
          {registering && (
            <Field
              label="Name"
              type="text"
              autoComplete="name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="optional"
            />
          )}
          <Field
            label="Email"
            type="email"
            required
            autoComplete="username"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
          <Field
            label="Password"
            type="password"
            required
            autoComplete={registering ? 'new-password' : 'current-password'}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            hint={registering ? `At least ${minimum} characters. Length beats punctuation.` : undefined}
          />

          <ErrorBanner error={error} onDismiss={() => setError(null)} />

          <Button
            type="submit"
            variant="primary"
            className="w-full"
            disabled={busy || !email || !password}
          >
            {registering ? 'Create account' : 'Sign in'}
          </Button>
        </form>

        {registrationOpen ? (
          <p className="mt-4 text-center text-xs" style={{ color: 'var(--text-muted)' }}>
            {registering ? 'Already have an account?' : 'No account yet?'}{' '}
            <button
              type="button"
              className="focus-ring underline"
              onClick={() => {
                setMode(registering ? 'login' : 'register')
                setError(null)
              }}
            >
              {registering ? 'Sign in' : 'Create one'}
            </button>
          </p>
        ) : (
          <p className="mt-4 text-center text-xs" style={{ color: 'var(--text-muted)' }}>
            Registration is closed on this deployment.
          </p>
        )}
      </Panel>
    </div>
  )
}

export function AccountMenu({ user, busy, onSignOut, onChangePassword, onDeleteAccount }) {
  const [open, setOpen] = useState(false)
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirmDelete, setConfirmDelete] = useState('')
  const [notice, setNotice] = useState(null)
  const [error, setError] = useState(null)

  const guard = async (task, success) => {
    setError(null)
    setNotice(null)
    try {
      await task()
      setNotice(success)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="relative">
      <button
        type="button"
        className="focus-ring rounded-md px-2 py-1 text-xs"
        style={{ color: 'var(--text-secondary)' }}
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        {user.name || user.email} ▾
      </button>

      {open && (
        <div
          className="absolute right-0 z-20 mt-1 w-72 rounded-md border p-3 text-xs shadow-lg"
          style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
        >
          <p className="mb-3 break-all" style={{ color: 'var(--text-muted)' }}>
            {user.email}
          </p>

          <Button className="mb-3 w-full" disabled={busy} onClick={onSignOut}>
            Sign out
          </Button>

          <details className="mb-2">
            <summary className="focus-ring cursor-pointer py-1">Change password</summary>
            <div className="mt-2 space-y-2">
              <Field
                label="Current password"
                type="password"
                autoComplete="current-password"
                value={current}
                onChange={(event) => setCurrent(event.target.value)}
              />
              <Field
                label="New password"
                type="password"
                autoComplete="new-password"
                value={next}
                onChange={(event) => setNext(event.target.value)}
                hint="Signs you out on every other device."
              />
              <Button
                className="w-full"
                disabled={busy || !current || !next}
                onClick={() =>
                  guard(async () => {
                    await onChangePassword(current, next)
                    setCurrent('')
                    setNext('')
                  }, 'Password changed; other devices signed out.')
                }
              >
                Update password
              </Button>
            </div>
          </details>

          <details>
            <summary className="focus-ring cursor-pointer py-1" style={{ color: 'var(--status-critical)' }}>
              Delete account
            </summary>
            <div className="mt-2 space-y-2">
              <p style={{ color: 'var(--text-muted)' }}>
                Removes every dataset, its semantic layer, its cleaning history and its uploaded
                files. This cannot be undone.
              </p>
              <Field
                label="Confirm with your password"
                type="password"
                autoComplete="current-password"
                value={confirmDelete}
                onChange={(event) => setConfirmDelete(event.target.value)}
              />
              <Button
                variant="danger"
                className="w-full"
                disabled={busy || !confirmDelete}
                onClick={() => guard(() => onDeleteAccount(confirmDelete), 'Account deleted.')}
              >
                Delete everything
              </Button>
            </div>
          </details>

          {notice && (
            <p className="mt-2" style={{ color: 'var(--status-good)' }}>
              {notice}
            </p>
          )}
          {error && (
            <p className="mt-2" style={{ color: 'var(--status-critical)' }}>
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
