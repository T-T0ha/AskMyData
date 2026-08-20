-- Enable pgvector on first boot.
-- The application also runs CREATE EXTENSION IF NOT EXISTS on startup, so a
-- database created outside docker-compose still works; this just makes a fresh
-- container ready before the first request.
CREATE EXTENSION IF NOT EXISTS vector;
