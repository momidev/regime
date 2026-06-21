# Regime — Market Regime Detection API

Backend e logica quantitativa per **Regime**, una web app di *market regime
detection*. Classifica in quale regime statistico si trova un asset (es.
*bull-calmo*, *bull-volatile*, *bear-calmo*, *bear-volatile*) usando un **Hidden
Markov Model** e fornisce le probabilità di transizione tra regimi.

Questo repository contiene **solo il backend**, esposto come API REST pulita
(FastAPI + OpenAPI/Swagger) pensata per essere consumata dal frontend Lovable.

> ⚠️ **Disclaimer.** L'output è **descrittivo e statistico**, non costituisce
> consiglio di investimento né segnale operativo. Il sistema **non esegue
> ordini**, non si connette ad alcun exchange e non fa trading reale o paper:
> legge solo dati di mercato pubblici e ne fa analisi statistica.

---

## Indice
1. [Caratteristiche](#caratteristiche)
2. [Asset supportati](#asset-supportati)
3. [Architettura / struttura cartelle](#architettura--struttura-cartelle)
4. [Installazione](#1-installazione)
5. [Configurazione (Supabase opzionale)](#2-configurazione-supabase-opzionale)
6. [Primo training](#3-primo-training)
7. [Avvio del server FastAPI](#4-avvio-del-server-fastapi)
8. [Endpoint API + esempi curl](#5-endpoint-api)
9. [Refresh giornaliero (cron)](#6-refresh-giornaliero-cron)
10. [Test](#7-test)
11. [Deploy su Railway](#8-deploy-su-railway)

---

## Caratteristiche
- **Gaussian HMM** (`hmmlearn`) con numero di stati configurabile (default 4).
- **Feature engineering**: ritorni log, volatilità rolling annualizzata (20gg),
  momentum/ROC (10gg).
- **Etichettatura automatica** degli stati in regimi leggibili e deterministici.
- **Storico giornaliero** delle classificazioni per il grafico timeline.
- **Matrice di transizione** tra regimi.
- **Alert** di cambio regime nelle ultime 24h.
- **Database opzionale**: con `DATABASE_URL` usa PostgreSQL/Supabase; senza,
  ricade automaticamente su uno store a file JSON locali (zero setup).
- **OpenAPI/Swagger** automatici su `/docs` (da passare a Lovable).

## Asset supportati

| ID API    | Ticker Yahoo | Nome                 | Classe        |
|-----------|--------------|----------------------|---------------|
| `BTC-USD` | `BTC-USD`    | Bitcoin              | crypto        |
| `SPY`     | `SPY`        | S&P 500 ETF          | equity-index  |
| `XLK`     | `XLK`        | Technology Sector    | sector        |
| `XLE`     | `XLE`        | Energy Sector        | sector        |
| `XLF`     | `XLF`        | Financials Sector    | sector        |
| `EURUSD`  | `EURUSD=X`   | Euro / US Dollar     | fx            |

## Architettura / struttura cartelle

```
regime/
├── api/            # FastAPI: app, route, schemi Pydantic
├── models/         # HMM, feature engineering, etichettatura regimi
├── data/           # fetch dati yfinance + artifacts/ (modelli .pkl, store JSON)
├── db/             # schema SQL, connessione, repository (SQL | file)
├── scripts/        # training iniziale e refresh (cron)
├── tests/          # test pytest (senza rete)
├── config.py       # impostazioni da .env
├── assets.py       # registry asset supportati
├── service.py      # orchestrazione (usata da API e script)
├── requirements.txt
├── Procfile / railway.json
└── .env.example
```

---

## 1. Installazione

Richiede **Python 3.11+**.

```bash
cd regime
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configurazione (Supabase opzionale)

Copia il template e (opzionalmente) compila i valori:

```bash
cp .env.example .env
```

- **Senza database** (più rapido per iniziare / Lovable): lascia `DATABASE_URL`
  vuoto. Storico e transizioni vengono salvati in `data/artifacts/store/`.
- **Con Supabase/PostgreSQL**:
  1. Su Supabase apri **SQL Editor** ed esegui il contenuto di
     [`db/schema.sql`](db/schema.sql) per creare le tabelle.
  2. Vai su **Project Settings → Database → Connection string (URI)**, copia la
     stringa e impostala in `.env` usando il driver `psycopg`:
     ```
     DATABASE_URL=postgresql+psycopg://postgres:PASSWORD@db.xxxx.supabase.co:5432/postgres
     ```

Altri parametri (numero di stati, finestre, lookback) sono documentati in
[`.env.example`](.env.example).

## 3. Primo training

Scarica lo storico, addestra l'HMM per ogni asset, salva gli artifact in
`data/artifacts/` e fa il backfill dello storico delle classificazioni:

```bash
# Tutti gli asset
python -m scripts.train_all

# Oppure solo alcuni
python -m scripts.train_all BTC-USD SPY
```

Lo stesso comando può essere rilanciato periodicamente per il **re-training**.

## 4. Avvio del server FastAPI

```bash
uvicorn api.main:app --reload
```

- API: <http://localhost:8000>
- **Swagger UI**: <http://localhost:8000/docs>  ← *da passare a Lovable*
- ReDoc: <http://localhost:8000/redoc>
- OpenAPI JSON: <http://localhost:8000/openapi.json>

## 5. Endpoint API

| Metodo | Path                                   | Descrizione                                  |
|--------|----------------------------------------|----------------------------------------------|
| GET    | `/assets`                              | Lista asset supportati                       |
| GET    | `/regime/{asset}/current`              | Stato attuale + probabilità                  |
| GET    | `/regime/{asset}/history?days=90`      | Storico classificazioni (timeline)           |
| GET    | `/regime/{asset}/transition-matrix`    | Matrice di transizione                       |
| GET    | `/regime/{asset}/alert-status`         | Cambio di regime nelle ultime 24h            |
| POST   | `/regime/refresh`                      | Ricalcola tutti gli asset (cron)             |
| GET    | `/health`                              | Healthcheck                                  |

### Esempi curl

```bash
# Lista asset
curl http://localhost:8000/assets

# Stato corrente di BTC
curl http://localhost:8000/regime/BTC-USD/current

# Storico ultimi 90 giorni (per il grafico)
curl "http://localhost:8000/regime/BTC-USD/history?days=90"

# Matrice di transizione
curl http://localhost:8000/regime/SPY/transition-matrix

# Alert cambio regime
curl http://localhost:8000/regime/XLK/alert-status

# Refresh di tutti gli asset (tipicamente via cron)
curl -X POST http://localhost:8000/regime/refresh
```

Esempio di risposta di `/regime/BTC-USD/current`:

```json
{
  "asset": "BTC-USD",
  "as_of": "2026-06-20",
  "state_index": 0,
  "state_label": "bull-calmo",
  "description": "Il modello classifica il regime come tendenza rialzista con bassa volatilità (mercato in salita ordinata).",
  "probabilities": {
    "bull-calmo": 0.82,
    "bull-volatile": 0.10,
    "bear-calmo": 0.05,
    "bear-volatile": 0.03
  },
  "model_version": "20260620231500"
}
```

### Codici di errore

| Status | Quando                                                        |
|--------|--------------------------------------------------------------|
| 404    | Asset non presente nel registry                              |
| 409    | Modello/dati non ancora disponibili (serve training/refresh)|
| 422    | Parametri non validi o dati insufficienti per il training   |
| 503    | Errore nel recupero dati di mercato (yfinance)              |

## 6. Refresh giornaliero (cron)

Da eseguire **dopo la chiusura dei mercati**. Due modalità equivalenti:

```bash
# Via HTTP (utile su Railway con uno scheduler)
curl -X POST https://<tuo-deploy>/regime/refresh

# Via CLI
python -m scripts.refresh
```

## 7. Test

I test non effettuano chiamate di rete (prezzi sintetici, store temporaneo):

```bash
pytest -q
```

## 8. Deploy su Railway

Il progetto include sia [`Procfile`](Procfile) sia
[`railway.json`](railway.json) (builder Nixpacks, start con `uvicorn`,
healthcheck su `/health`).

1. Crea un nuovo progetto Railway dal repo.
2. Imposta le **variabili d'ambiente** (almeno `DATABASE_URL` se usi Supabase;
   gli altri valori hanno default sensati).
3. Railway espone automaticamente `$PORT` — già gestita dallo start command.
4. Esegui il primo training: una volta deployato, lancia un `POST /regime/refresh`
   (effettua il bootstrap addestrando i modelli mancanti) oppure esegui
   `python -m scripts.train_all` dalla shell del servizio.
5. Configura uno **scheduler** (cron Railway) per chiamare `POST /regime/refresh`
   una volta al giorno.

> **Persistenza modelli**: gli artifact `.pkl` sono salvati in
> `data/artifacts/`. Su Railway, se vuoi conservarli tra i deploy, monta un
> volume su quella cartella; in alternativa, dato che le classificazioni sono
> già su Supabase, basta rilanciare il training dopo ogni deploy.
```
