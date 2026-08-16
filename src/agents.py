"""Three-stage LLM agent pipeline for fraud confirmation.

Transactions that the rule engine (tools.py) cannot classify with confidence
are escalated through up to three agents of increasing scrutiny and cost:

  1. agent1  - fast screener (cheap model, strict binary rules)
  2. agent2  - validator (reviews agent1's call, produces a confidence score)
  3. agent3  - chief fraud officer (final, irreversible decision on the
               highest-value or lowest-confidence cases)

High-value transactions and unambiguous rule-based matches skip straight to
agent3 or are auto-classified by the pre-filter, keeping LLM calls for the
cases that actually need judgment.
"""

import asyncio
import json
import os
import re
import time

import pandas as pd
import ulid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

from .tools import known_users, pre_filter, score_transaction

load_dotenv()

model_a1 = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o",
    temperature=0,
    max_tokens=120,
)

model_a2 = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o",
    temperature=0,
    max_tokens=150,
)

model_a3 = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4",
    temperature=0,
    max_tokens=250,
)

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


def parse_json_safe(text: str) -> dict:
    text = text.strip().replace("```json", "").replace("```", "")
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}


async def agent1_async(tx: dict, score_result: dict, session_id: str) -> dict:
    """Fast, strict screener. Only flags fraud when at least two signals
    corroborate each other, or one signal is unambiguous."""
    amount = float(tx.get('amount', 0))
    desc = str(tx.get('description', ''))
    hour = pd.to_datetime(tx['timestamp']).hour
    zscore = score_result.get('zscore', 0)

    prompt = f"""You are a strict bank fraud detection agent.

## TRANSACTION
```json
{{
  "type": "{tx.get('transaction_type')}",
  "amount_eur": {int(amount)},
  "hour": {hour},
  "description": "{desc[:60]}",
  "recipient_id": "{tx.get('recipient_id')}",
  "risk_score": {score_result['combined_score']},
  "z_score": {zscore:.2f},
  "signals": {json.dumps(score_result['signals'])}
}}
```

## RULES (apply strictly)
FRAUD if:
- transfer + missing_description + null_recipient -> certain fraud
- burst_transaction signal detected
- phishing_sms signal + amount > 100EUR
- night_transfer + missing_description
- z_score > 2.5 + missing_description OR unknown_recipient

NORMAL if:
- withdrawal or in-person payment
- risk_score < 0.15 with no strong signals
- single weak signal only

REQUIREMENT: Need AT LEAST 2 corroborating signals OR 1 certain signal to flag fraud.

## RESPONSE
Return ONLY valid JSON:
{{"fraud": 0, "reason": "explanation"}}"""

    try:
        response = await model_a1.ainvoke([HumanMessage(content=prompt)])
        result = parse_json_safe(response.content)
        fraud = int(result.get("fraud", 0))
        if fraud not in [0, 1]:
            fraud = 1 if score_result['combined_score'] >= 0.55 else 0
        return {"fraud": fraud, "reason": str(result.get("reason", ""))[:100]}
    except Exception:
        fraud = 1 if score_result['combined_score'] >= 0.55 else 0
        return {"fraud": fraud, "reason": f"fallback_{score_result['combined_score']}"}


async def agent2_async(tx: dict, score_result: dict, agent1_reason: str, session_id: str) -> dict:
    """Reviews agent1's flag with more context (user profile) and decides
    whether the case needs to go to agent3 for a final call."""
    amount = float(tx.get('amount', 0))
    desc = str(tx.get('description', ''))
    hour = pd.to_datetime(tx['timestamp']).hour
    zscore = score_result.get('zscore', 0)
    sender = tx.get('sender_id', '')
    user_info = known_users.get(sender, {'name': 'unknown', 'salary': 0, 'city': 'unknown'})

    prompt = f"""You are a senior fraud analyst. Confirm or reject Agent1's fraud flag with high precision.

## FLAGGED TRANSACTION
```json
{{
  "type": "{tx.get('transaction_type')}",
  "amount_eur": {int(amount)},
  "hour": {hour},
  "description": "{desc[:60]}",
  "recipient_id": "{tx.get('recipient_id')}",
  "balance_after_eur": {tx.get('balance_after')},
  "z_score": {zscore:.2f},
  "risk_score": {score_result['combined_score']},
  "user_name": "{user_info['name']}",
  "user_monthly_salary_eur": {round(user_info['salary'])},
  "user_city": "{user_info['city']}"
}}
```

## SIGNALS DETECTED
{json.dumps(score_result['signals'], indent=2)}

## AGENT1 SAID
"{agent1_reason}"

## VALIDATION CRITERIA
CONFIRM FRAUD if:
- 2+ strong corroborating signals
- amount > 500EUR + missing_description + unknown_recipient
- phishing within 30 days + any transaction signal
- z_score > 2.5 + context supports fraud

REJECT (normal) if:
- Single weak signal + plausible legitimate explanation
- Amount consistent with normal user spending + description present
- Agent1 reason seems insufficient

ECONOMIC RULE: Missing {int(amount)}EUR fraud costs {int(amount)}EUR. False positive costs only minor friction.
{"Bias toward FRAUD confirmation at this amount." if amount > 400 else "Require clear evidence at this amount."}

## RESPONSE
Return ONLY valid JSON:
{{"fraud": 0, "confidence": 0.85, "reason": "detailed_explanation"}}"""

    try:
        response = await model_a2.ainvoke([HumanMessage(content=prompt)])
        result = parse_json_safe(response.content)
        fraud = int(result.get("fraud", 0))
        confidence = float(result.get("confidence", 0.5))
        if fraud not in [0, 1]:
            fraud = 0
        confidence = max(0.0, min(1.0, confidence))
        needs_a3 = (confidence < 0.80 and amount > 350) or (fraud == 1 and confidence < 0.70)
        return {
            "fraud": fraud, "confidence": confidence,
            "reason": str(result.get("reason", ""))[:120],
            "needs_agent3": needs_a3,
        }
    except Exception:
        fraud = 1 if score_result['combined_score'] >= 0.55 else 0
        return {"fraud": fraud, "confidence": 0.5, "reason": "fallback", "needs_agent3": False}


@observe()
def agent3_secure(tx: dict, score_result: dict, prev_result: dict, session_id: str) -> dict:
    """Chief Fraud Officer: final, irreversible decision on high-value or
    low-confidence cases. Uses the strongest model and a stricter framework."""
    handler = CallbackHandler()
    amount = float(tx.get('amount', 0))
    desc = str(tx.get('description', ''))
    zscore = score_result.get('zscore', 0)
    sender = tx.get('sender_id', '')
    user_info = known_users.get(sender, {'name': 'unknown', 'salary': 0, 'city': 'unknown'})

    prompt = f"""You are the Chief Fraud Officer. Make the FINAL irreversible decision.

## HIGH-VALUE TRANSACTION
```json
{{
  "type": "{tx.get('transaction_type')}",
  "amount_eur": {int(amount)},
  "hour": {pd.to_datetime(tx['timestamp']).hour},
  "description": "{desc[:80]}",
  "recipient_id": "{tx.get('recipient_id')}",
  "balance_after_eur": {tx.get('balance_after')},
  "z_score": {zscore:.3f},
  "risk_score": {score_result['combined_score']},
  "salary_multiple": {round(amount / max(user_info['salary'], 1), 2)}
}}
```

## USER PROFILE
```json
{{
  "name": "{user_info['name']}",
  "monthly_salary_eur": {round(user_info['salary'])},
  "home_city": "{user_info['city']}"
}}
```

## ALL SIGNALS
{json.dumps(score_result['signals'], indent=2)}

## PREVIOUS ANALYSIS
"{prev_result.get('reason', 'N/A')}"
Previous confidence: {prev_result.get('confidence', 'N/A')}

## DECISION FRAMEWORK

CERTAIN FRAUD (any ONE):
- Transfer + null_recipient + missing_description
- Z-score > 3.5
- Phishing SMS <=7 days before + amount > 200EUR

PROBABLE FRAUD (need TWO or more):
- Z-score > 2.0
- Missing description on transfer
- First-time unknown recipient
- Burst transaction pattern
- Phishing SMS <=30 days before
- Amount > 2x monthly salary ({round(user_info['salary'] * 2)}EUR)

NOT FRAUD:
- Withdrawal or in-person payment
- Regular bill keywords in description
- Known frequent recipient

## ECONOMIC IMPACT
{"CRITICAL VALUE - default to FRAUD if any ambiguity" if amount > 800 else "SIGNIFICANT - require 2 signals minimum"}
Fraud loss if missed: {int(amount)} EUR
False positive cost: minor customer friction

## RESPONSE
Return ONLY this exact JSON:
{{
  "fraud": <0_or_1>,
  "confidence": <0.00_to_1.00>,
  "reason": "<list specific signals that drove decision>"
}}"""

    response = model_a3.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [handler], "metadata": {"langfuse_session_id": session_id}},
    )

    result = parse_json_safe(response.content)
    fraud = int(result.get("fraud", 0))
    confidence = float(result.get("confidence", 0.5))
    if fraud not in [0, 1]:
        fraud = prev_result.get('fraud', 0)
    return {
        "fraud": fraud,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(result.get("reason", ""))[:200],
    }


async def process_batch(batch, sms_list, mails_list, locations, session_id):
    results = []

    scored = []
    for tx_dict in batch:
        sr = score_transaction(tx_dict, sms_list, mails_list, locations)
        pf_dec, pf_reason = pre_filter(tx_dict, sr)
        scored.append((tx_dict, sr, pf_dec, pf_reason))

    certain, high_value, uncertain = [], [], []
    for tx_dict, sr, pf_dec, pf_reason in scored:
        if pf_dec == "fraud":
            certain.append((tx_dict, sr, pf_reason))
        elif pf_dec == "high_value_priority":
            high_value.append((tx_dict, sr, pf_reason))
        elif pf_dec == "uncertain":
            uncertain.append((tx_dict, sr))

    for tx_dict, sr, reason in certain:
        results.append({'transaction_id': tx_dict['transaction_id'],
                         'amount': tx_dict['amount'], 'confidence': 0.97,
                         'reason': reason, 'agent': 'prefilter'})

    for tx_dict, sr, reason in high_value:
        a3 = agent3_secure(tx_dict, sr, {'reason': reason, 'fraud': 1, 'confidence': 0.7}, session_id)
        if a3['fraud'] == 1 or a3['confidence'] >= 0.65:
            results.append({'transaction_id': tx_dict['transaction_id'],
                             'amount': tx_dict['amount'], 'confidence': a3['confidence'],
                             'reason': a3['reason'], 'agent': 'agent3_priority'})

    if not uncertain:
        return results

    a1_tasks = [agent1_async(tx, sr, session_id) for tx, sr in uncertain]
    a1_results = await asyncio.gather(*a1_tasks)

    needs_a2 = []
    for (tx_dict, sr), a1 in zip(uncertain, a1_results):
        if a1['fraud'] == 1:
            amount = float(tx_dict.get('amount', 0))
            if amount > 100:
                needs_a2.append((tx_dict, sr, a1))
            elif sr['combined_score'] >= 0.40:
                results.append({'transaction_id': tx_dict['transaction_id'],
                                 'amount': tx_dict['amount'], 'confidence': 0.72,
                                 'reason': a1['reason'], 'agent': 'agent1'})

    if not needs_a2:
        return results

    a2_tasks = [agent2_async(tx, sr, a1['reason'], session_id) for tx, sr, a1 in needs_a2]
    a2_results = await asyncio.gather(*a2_tasks)

    needs_a3 = []
    for (tx_dict, sr, _), a2 in zip(needs_a2, a2_results):
        if a2.get('needs_agent3'):
            needs_a3.append((tx_dict, sr, a2))
        elif a2['fraud'] == 1 and a2['confidence'] >= 0.72:
            results.append({'transaction_id': tx_dict['transaction_id'],
                             'amount': tx_dict['amount'], 'confidence': a2['confidence'],
                             'reason': a2['reason'], 'agent': 'agent2'})

    for tx_dict, sr, a2 in needs_a3:
        a3 = agent3_secure(tx_dict, sr, a2, session_id)
        if a3['fraud'] == 1 or a3['confidence'] >= 0.70:
            results.append({'transaction_id': tx_dict['transaction_id'],
                             'amount': tx_dict['amount'], 'confidence': a3['confidence'],
                             'reason': a3['reason'], 'agent': 'agent3'})

    return results


def _output_filename(folder: str) -> str:
    slug = folder.strip().lower().replace(' - ', '_').replace(' ', '_')
    return f"output_{slug}.txt"


async def run_pipeline_async(folder: str, batch_size: int = 12) -> str:
    """Run the full pipeline against a dataset folder (transactions.csv,
    sms.json, mails.json, locations.json) and write the flagged transaction
    IDs to results/output_<dataset>.txt."""
    start = time.time()
    session_id = f"NETIZEN-{ulid.new().str}"
    print(f"Session: {session_id}")
    print(f"Dataset: {folder}\n")

    df_tx = pd.read_csv(f'{folder}/transactions.csv')
    with open(f'{folder}/sms.json') as f:
        sms_raw = json.load(f)
    with open(f'{folder}/mails.json') as f:
        mails_raw = json.load(f)
    with open(f'{folder}/locations.json') as f:
        locations = json.load(f)

    sms_list = [{'sms': s} if isinstance(s, str) else s for s in sms_raw]
    mails_list = [{'mail': m} if isinstance(m, str) else m for m in mails_raw]

    all_txs = [row.to_dict() for _, row in df_tx.iterrows()]
    batches = [all_txs[i:i + batch_size] for i in range(0, len(all_txs), batch_size)]
    print(f"{len(all_txs)} transactions -> {len(batches)} batches of {batch_size}\n")

    all_frauds = []
    for i, batch in enumerate(batches):
        elapsed = time.time() - start
        print(f"  Batch {i + 1}/{len(batches)} | {elapsed:.0f}s | frauds so far: {len(all_frauds)}", end="\r")
        all_frauds.extend(await process_batch(batch, sms_list, mails_list, locations, session_id))

    langfuse_client.flush()
    elapsed = time.time() - start

    agents = {}
    for f in all_frauds:
        agents[f.get('agent', '?')] = agents.get(f.get('agent', '?'), 0) + 1

    print(f"\n\n{'=' * 60}")
    print(f"RESULTS ({elapsed:.0f}s) -- {len(all_frauds)}/{len(all_txs)} ({100 * len(all_frauds) / len(all_txs):.1f}%)")
    for agent, count in sorted(agents.items()):
        print(f"  {agent:22s}: {count}")

    sorted_frauds = sorted(all_frauds, key=lambda x: x['amount'], reverse=True)
    print(f"\nValue protected: {sum(f['amount'] for f in all_frauds):.0f} EUR")
    print("\nFlagged transactions:")
    for f in sorted_frauds:
        print(f"  {f['transaction_id'][:8]} | {f['amount']:>8.0f}EUR | "
              f"conf:{f['confidence']:.2f} | [{f['agent']:20s}] | {f['reason'][:55]}")

    output_path = _output_filename(folder)
    with open(output_path, 'w') as f:
        for fraud in all_frauds:
            f.write(fraud['transaction_id'] + '\n')

    print(f"\n{len(all_frauds)} frauds written to {output_path}")
    print(f"Session ID: {session_id}")
    return session_id
