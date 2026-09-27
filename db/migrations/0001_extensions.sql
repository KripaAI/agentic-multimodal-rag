-- Phase 0: enable pgvector. Tables arrive with Phase 3 migrations (LLD §4.1).
CREATE EXTENSION IF NOT EXISTS vector;
