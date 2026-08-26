"""Phase 4 — semantic layer construction and export.

Turns the validated analysis into a real database: a schema of its own, one
table per sheet with the primary and foreign keys the user confirmed, the rows
themselves, and a ``sem_metadata`` table describing every column so that Phase
5 can retrieve against it.

Four rules hold across the whole package:

``nothing invented``      a blank cell exports as ``NULL``, and a declared type
                          that would round or truncate a value is widened
                          instead (see :mod:`app.export.types`)
``nothing unconfirmed``   only a relationship a human confirmed becomes a
                          ``FOREIGN KEY``
``nothing unquoted``      identifiers are sanitised and then emitted through
                          SQLAlchemy objects, never formatted into SQL text
                          (see :mod:`app.export.naming`)
``nothing outside``       the exporter creates and drops exactly one schema,
                          named after the session, and checks the name before
                          every destructive statement
"""
