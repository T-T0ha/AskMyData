-- The role Phase 5's query execution connects as, so a generated SELECT that
-- somehow got past the application-level guard (see app/query/guard.py) still
-- cannot write or drop anything: it fails at the database, not merely at the
-- application. Table/schema-level SELECT is granted per dataset by
-- app.export.runner whenever a session's schema is (re)built, not here.
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'semantic_readonly') THEN
      CREATE ROLE semantic_readonly WITH LOGIN PASSWORD 'semantic_readonly'
        NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
   END IF;
END
$$;

GRANT CONNECT ON DATABASE semanticlayer TO semantic_readonly;
REVOKE ALL ON SCHEMA public FROM semantic_readonly;
