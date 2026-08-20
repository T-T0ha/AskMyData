"""Authentication: accounts, passwords, login tokens and ownership.

Three modules, in dependency order:

``passwords``    hashing and verification, no database, no framework
``service``      register / authenticate / issue / revoke, database only
``deps``         the FastAPI dependencies routes actually depend on

FR-13 and FR-14 of the requirements: users register and authenticate, and
every dataset-scoped endpoint verifies ownership *before* reading anything.
"""
