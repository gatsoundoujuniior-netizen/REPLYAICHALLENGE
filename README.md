# Fraud Detection Agent Pipeline

Multi-agent fraud detection system built for the Reply AI Challenges (Netizen track).
Given a bank's raw activity for a period (transactions, SMS, emails, GPS history, user
profiles), the pipeline flags the transactions most likely to be fraudulent and reports
the total value protected.

## Approach

Detection runs in two stages so that the (paid) LLM calls are reserved for the cases
that actually need judgment:

1. **Rule-based pre-filter** (`src/tools.py`) scores every transaction on missing
   descriptions, unusual hours, unknown recipients, phishing exposure (SMS/email),
   GPS mismatches, spending relative to salary, and burst patterns. Clear-cut cases
   (obvious fraud, obvious normal activity) are resolved here without calling a model.
2. **LLM agent escalation** (`src/agents.py`) handles the remaining ambiguous cases
   through up to three agents of increasing scrutiny and cost:
   - **Agent 1 - Fast screener**: cheap model, strict binary rule, requires
     corroborating signals before flagging.
   - **Agent 2 - Validator**: reviews Agent 1's call with more context (user profile,
     balance) and produces a confidence score.
   - **Agent 3 - Chief Fraud Officer**: strongest model, makes the final call on
     high-value transactions or any case Agent 2 wasn't confident about.

Batches of transactions are processed concurrently with `asyncio`, and every LLM
call is traced through [Langfuse](https://langfuse.com) for observability (see
`assets/screenshots/`).

## Results

| Dataset | Transactions | Flagged | Value protected | Pipeline |
|---|---|---|---|---|
| Brave New World - validation | 524 | 10 | 21,127 EUR | 3-agent (final version) |
| The Truman Show - validation | 80 | 7 | 450.36 EUR | 2-agent (earlier iteration) |

**Competition result:** about 30,600 EUR flagged across the 2 completed datasets, on a 40 USD API budget. Ranked 256th of 1,971 teams at the 16/04/2026 snapshot (top 13%, see `assets/screenshots/leaderboard_score.jpg`).

The table shows the runs stored in this repository. The Truman Show row comes from an earlier 2-agent iteration: the later submission on this dataset flagged about 9,500 EUR, and its output is not stored here.

`Brave New World - train` and `Deus Ex - validation` were loaded for exploration but
were not run through the full pipeline, so no result is reported for them.

Detailed run logs and reasoning per flagged transaction are in the corresponding
notebooks under `notebooks/`. Flagged transaction IDs are in `results/`.

## Repository structure

```
src/
  tools.py     Rule-based scoring engine (pre-filter stage)
  agents.py    3-agent LLM escalation pipeline
config/
  known_users.example.json   Schema for the local, gitignored user profile config
notebooks/
  01_brave_new_world_pipeline.ipynb        Final pipeline run and results
  02_truman_show_pipeline.ipynb            Earlier 2-agent iteration and results
  03_scratch_brave_new_world_train.ipynb   Exploratory notes (incomplete)
results/
  output_brave_new_world_validation.txt    Flagged transaction IDs
  output_truman_show_validation.txt        Flagged transaction IDs
  submission_brave_new_world_validation.zip
assets/
  screenshots/   Langfuse observability dashboards and leaderboard score
main.py          CLI entry point
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file at the repository root with:

```
OPENROUTER_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
TEAM_NAME=
```

Copy `config/known_users.example.json` to `config/known_users.json` and fill it in
with the user profiles from the dataset's `users.json` (name, salary, city, phone,
email). This file is gitignored because it contains personal data from the challenge
dataset, not project code.

## Data

The challenge datasets (`Brave New World - *`, `The Truman Show - *`, `Deus Ex - *`)
are not included in this repository — they belong to the challenge organizer, not to
this project. Each dataset folder is expected at the repository root and must contain
`transactions.csv`, `users.json`, `sms.json`, `mails.json` and `locations.json`.

## Running the pipeline

```bash
uv run main.py "Brave New World - validation"
```

This writes the flagged transaction IDs to `output_<dataset>.txt` in the working
directory. The same pipeline is also available interactively in
`notebooks/01_brave_new_world_pipeline.ipynb`.
