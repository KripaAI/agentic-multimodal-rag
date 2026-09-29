-- Phase 3: the searchable library (spec §5.5, LLD §4.1, §4.3).
-- Everything below documents cascades from it, so replacing or removing a document
-- is one DELETE inside the write transaction (LLD §3.8).

CREATE TABLE documents (
    doc_id        text PRIMARY KEY,           -- first 16 hex of SHA-256(PDF bytes)
    source_file   text NOT NULL,
    content_hash  text NOT NULL,              -- full SHA-256 of the PDF
    build_hash    text NOT NULL,              -- hash of everything written: re-ingest is a no-op when unchanged
    version       int  NOT NULL CHECK (version >= 1),
    status        text NOT NULL DEFAULT 'active' CHECK (status IN ('active')),
    ingested_at   timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX documents_one_active_per_file ON documents (source_file) WHERE status = 'active';

CREATE TABLE elements (
    element_id    text PRIMARY KEY,
    doc_id        text NOT NULL REFERENCES documents ON DELETE CASCADE,
    page          int  NOT NULL CHECK (page >= 1),
    bbox          float8[] NOT NULL CHECK (array_length(bbox, 1) = 4),
    type          text NOT NULL CHECK (type IN ('text', 'image', 'vector_figure', 'table', 'scanned_page')),
    section_path  text[] NOT NULL DEFAULT '{}',
    text          text,
    caption       text,
    asset_path    text,                       -- relative to paths.data_dir
    content_hash  text NOT NULL,
    status        text NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'skipped', 'needs_review')),
    skip_reason   text
);
CREATE INDEX elements_doc_page ON elements (doc_id, page);

CREATE TABLE figure_captions (
    element_id           text PRIMARY KEY REFERENCES elements ON DELETE CASCADE,
    figure_type          text NOT NULL,
    short_caption        text NOT NULL,
    detailed_description text NOT NULL,
    visible_text         text[] NOT NULL DEFAULT '{}',
    extracted_data       jsonb,
    keywords             text[] NOT NULL DEFAULT '{}',
    confidence           text NOT NULL,
    status               text NOT NULL CHECK (status IN ('ok', 'needs_review')),
    review_note          text,               -- why it needs review, or what the flag check changed
    model_id             text NOT NULL,
    prompt_version       text NOT NULL
);

CREATE TABLE doc_tables (
    element_id      text PRIMARY KEY REFERENCES elements ON DELETE CASCADE,
    columns         jsonb NOT NULL,
    rows            jsonb NOT NULL,
    units           jsonb NOT NULL DEFAULT '{}',
    numeric_columns text[] NOT NULL DEFAULT '{}',
    title           text,
    summary         text,
    low_confidence  boolean NOT NULL DEFAULT false
);

CREATE TABLE element_links (
    target_id        text NOT NULL REFERENCES elements ON DELETE CASCADE,   -- figure or table
    text_element_id  text NOT NULL REFERENCES elements ON DELETE CASCADE,
    method           text NOT NULL CHECK (method IN ('explicit', 'deictic', 'related')),
    score            real NOT NULL,
    PRIMARY KEY (target_id, text_element_id)
);
CREATE INDEX element_links_text ON element_links (text_element_id);

CREATE TABLE search_chunks (
    chunk_id        text PRIMARY KEY,
    doc_id          text NOT NULL REFERENCES documents ON DELETE CASCADE,
    collection      text NOT NULL CHECK (collection IN ('text', 'figure', 'table')),
    element_ids     text[] NOT NULL,          -- in reading order
    dense_text      text NOT NULL,
    keyword_text    text NOT NULL,
    embedding       vector(1536) NOT NULL,    -- = embed.dims; changing it needs a migration (spec §6.5)
    embedding_model text NOT NULL,
    tsv_english     tsvector GENERATED ALWAYS AS (to_tsvector('english', keyword_text)) STORED,
    tsv_simple      tsvector GENERATED ALWAYS AS (to_tsvector('simple', keyword_text)) STORED
);
CREATE INDEX search_chunks_collection_doc ON search_chunks (collection, doc_id);
-- One partial HNSW index per collection, so a filtered search still returns k results.
CREATE INDEX search_chunks_hnsw_text   ON search_chunks USING hnsw (embedding vector_cosine_ops) WHERE collection = 'text';
CREATE INDEX search_chunks_hnsw_figure ON search_chunks USING hnsw (embedding vector_cosine_ops) WHERE collection = 'figure';
CREATE INDEX search_chunks_hnsw_table  ON search_chunks USING hnsw (embedding vector_cosine_ops) WHERE collection = 'table';
CREATE INDEX search_chunks_tsv_english ON search_chunks USING gin (tsv_english);
CREATE INDEX search_chunks_tsv_simple  ON search_chunks USING gin (tsv_simple);
