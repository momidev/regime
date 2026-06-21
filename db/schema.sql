-- ============================================================
-- Regime — schema PostgreSQL / Supabase
-- Eseguire questo script nel SQL editor di Supabase (o via psql)
-- prima del primo training quando si usa il database.
-- ============================================================

-- Storico giornaliero delle classificazioni di regime per asset.
CREATE TABLE IF NOT EXISTS regime_classifications (
    id            BIGSERIAL PRIMARY KEY,
    asset         TEXT        NOT NULL,
    date          DATE        NOT NULL,
    state_index   INTEGER     NOT NULL,
    state_label   TEXT        NOT NULL,
    probs         JSONB       NOT NULL,          -- {label: probabilità}
    close         DOUBLE PRECISION,              -- prezzo di chiusura (per overlay prezzo+regimi)
    log_return    DOUBLE PRECISION,
    volatility    DOUBLE PRECISION,
    momentum      DOUBLE PRECISION,
    model_version TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_classification_asset_date UNIQUE (asset, date)
);

CREATE INDEX IF NOT EXISTS idx_classifications_asset_date
    ON regime_classifications (asset, date DESC);

-- Snapshot della matrice di transizione più recente per asset.
CREATE TABLE IF NOT EXISTS transition_matrices (
    asset         TEXT        PRIMARY KEY,
    matrix        JSONB       NOT NULL,          -- lista di liste (n x n)
    state_labels  JSONB       NOT NULL,          -- etichette ordinate per indice
    model_version TEXT        NOT NULL,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Log dei cambi di regime, usato per gli alert sulle ultime 24h.
CREATE TABLE IF NOT EXISTS regime_changes (
    id          BIGSERIAL PRIMARY KEY,
    asset       TEXT        NOT NULL,
    from_label  TEXT,
    to_label    TEXT        NOT NULL,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_changes_asset_time
    ON regime_changes (asset, changed_at DESC);
