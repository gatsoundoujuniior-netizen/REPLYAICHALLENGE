"""Rule-based fraud scoring engine used as the pre-filter stage of the pipeline.

Every transaction is scored by a set of independent checks (missing description,
odd hours, unknown recipients, phishing exposure, GPS mismatch, spending vs.
salary, burst transactions). The combined score decides whether a transaction
is auto-classified, escalated straight to the senior agent, or sent through the
full agent chain (see agents.py).
"""

import json
import os
from collections import defaultdict

import pandas as pd

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
KNOWN_USERS_PATH = os.path.join(CONFIG_DIR, "known_users.json")
KNOWN_USERS_EXAMPLE_PATH = os.path.join(CONFIG_DIR, "known_users.example.json")


def load_known_users() -> dict:
    """Load user profiles (name, salary, city, phone, mail) from a local,
    gitignored config file. These come from the challenge dataset and are not
    shipped with the repository — see config/known_users.example.json."""
    if not os.path.exists(KNOWN_USERS_PATH):
        raise FileNotFoundError(
            f"{KNOWN_USERS_PATH} not found. Copy "
            f"{KNOWN_USERS_EXAMPLE_PATH} to known_users.json and fill it with "
            "the user profiles from the dataset's users.json."
        )
    with open(KNOWN_USERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_comment", None)
    return data


known_users = load_known_users()

regular_senders = ['EMP68176', 'EMP40508', 'EMP98299', 'EMP47291', 'EMP81634', 'EMP52067', 'EMP73915']

regular_keywords = [
    'salary', 'rent', 'insurance', 'health', 'subscription', 'phone bill', 'loan payment',
    'professional liability', 'professional membership', 'creative suite', 'professional body',
    'consulting software', 'professional indemnity', 'creative cloud', 'coworking', 'adobe',
    'stock photo', 'creative tools', 'freelance content', 'internet subscription',
    'retiree health', 'senior health', 'home insurance', 'monthly health', 'private health',
    'monthly professional', 'indemnity', 'membership fee', 'professional dues', 'professional association'
]

CASH_TYPES = {'withdrawal', 'in-person payment'}

phishing_keywords = [
    'paypa1', 'amaz0n', 'verify', 'netfl1x', 'renewal', 'update payment', 'update card',
    'failed', 'suspended', 'inps-verify', 'pension-verify', 'pensi0n', 'socs', 'ch4se',
    'natw3st', 'chase-secure', 'hmrc-secure', 'paypa1-secure', 'amaz0n-verify', 'bremenbank'
]

# Per-sender spending baseline (mean/std of historical transaction amounts),
# used to compute a z-score for how unusual a given amount is.
USER_AMOUNT_STATS = {
    'RTZO-JMIX-7D1-GAU-0': {'mean': 132, 'std': 158},
    'VLNT-SLVS-7DC-PIO-0': {'mean': 70, 'std': 41},
    'FCHN-VTSI-7FA-BRE-0': {'mean': 526, 'std': 703},
    'NNLL-BRTU-7D3-RAV-0': {'mean': 208, 'std': 178},
    'WTZL-TNAI-80B-PFA-0': {'mean': 160, 'std': 125},
    'JNSO-PLNA-7D0-FEL-1': {'mean': 207, 'std': 204},
    'CSPR-GRHL-800-ILM-1': {'mean': 173, 'std': 225}
}

# How often each recipient shows up in the transaction history — a
# first-time recipient is a stronger fraud signal than a recurring one.
RECIPIENT_FREQUENCY = {
    'PROP93151': 13, 'RENT75951': 13, 'HOME28625': 13, 'DOM58085': 12, 'HOME36805': 12,
    'HOME96755': 12, 'ABIT73300': 12, 'COMRB39809': 5, 'PAYPH11103': 5, 'VNDMR15938': 3,
    'ENTSR71727': 3, 'PAYBR30021': 3, 'PAYWT22183': 2, 'SHPRB44898': 2, 'TRDDN82305': 2,
    'PAYNV92614': 2, 'PAYDN65923': 2, 'PAYBR92660': 2, 'PAYCL92396': 2, 'SYSTR98866': 2,
    'ACCRN27991': 2, 'BILMN49918': 2, 'MKTSW37100': 2,
}

_tx_history = defaultdict(list)


def is_regular(tx):
    desc = str(tx.get('description', '')).lower()
    sender = str(tx.get('sender_id', ''))
    if sender in regular_senders:
        return True
    return any(k in desc for k in regular_keywords)


def is_cash(tx):
    return str(tx.get('transaction_type', '')) in CASH_TYPES


def get_zscore(tx):
    sender = tx.get('sender_id', '')
    if sender not in USER_AMOUNT_STATS:
        return 0.0
    s = USER_AMOUNT_STATS[sender]
    if s['std'] == 0:
        return 0.0
    return (float(tx.get('amount', 0)) - s['mean']) / s['std']


def pre_filter(tx, score_result):
    """First triage pass: decide whether a transaction is clearly normal,
    clearly fraud, needs immediate senior review, or is genuinely uncertain."""
    amount = float(tx.get('amount', 0))
    desc = str(tx.get('description', ''))
    tx_type = str(tx.get('transaction_type', ''))
    signals = score_result['signals']
    hint = score_result['hint']
    hour = pd.to_datetime(tx['timestamp']).hour
    zscore = get_zscore(tx)

    no_desc = desc in ('nan', 'None', '') or pd.isna(tx.get('description'))
    null_recip = not tx.get('recipient_id') or str(tx.get('recipient_id')) == 'nan'

    if is_regular(tx):
        return "normal", "regular"
    if is_cash(tx):
        if amount > 500 and hour < 6:
            return "uncertain", "cash_night_high"
        return "normal", "cash"

    if amount > 800 and not is_regular(tx) and not is_cash(tx):
        if no_desc or zscore > 2.0 or len(signals) >= 2:
            return "high_value_priority", f"priority_{int(amount)}EUR_z{zscore:.1f}"

    if tx_type == 'transfer' and no_desc and null_recip:
        return "fraud", "transfer_no_desc_null_recipient"
    if tx_type == 'transfer' and amount > 2000 and no_desc:
        return "fraud", f"critical_{int(amount)}EUR_no_desc"
    if zscore > 3.5 and not is_regular(tx):
        return "high_value_priority", f"zscore_extreme_{zscore:.1f}"

    if not signals and hint == 'low_risk':
        return "normal", "no_signals"
    return "uncertain", f"s:{len(signals)}_h:{hint}"


def check_rules(tx):
    signals, score = [], 0.0
    desc = str(tx.get('description', ''))
    tx_type = str(tx.get('transaction_type', ''))
    amount = float(tx.get('amount', 0))
    hour = pd.to_datetime(tx['timestamp']).hour
    recipient = str(tx.get('recipient_id', ''))
    zscore = get_zscore(tx)
    regular, cash = is_regular(tx), is_cash(tx)

    if not tx.get('description') or pd.isna(tx.get('description')):
        if tx_type == 'transfer':
            score += 0.5
            signals.append("missing_desc_transfer")
        elif not cash:
            score += 0.15
            signals.append("missing_desc")

    if hour < 6 and not cash and tx_type == 'transfer':
        score += 0.3
        signals.append("night_transfer")
    elif hour < 6 and not cash:
        score += 0.1
        signals.append("night_tx")

    if not tx.get('recipient_id') or str(tx.get('recipient_id')) == 'nan':
        if tx_type == 'transfer':
            score += 0.4
            signals.append("null_recip_transfer")

    if zscore > 3.0 and not regular:
        score += 0.4
        signals.append(f"zscore_outlier_{zscore:.1f}")
    elif zscore > 2.0 and not regular:
        score += 0.2
        signals.append(f"zscore_high_{zscore:.1f}")

    freq = RECIPIENT_FREQUENCY.get(recipient, 0)
    if freq == 0 and recipient not in ('nan', 'None', '') and recipient and tx_type == 'transfer' and not regular:
        score += 0.2
        signals.append("unknown_recip_first_time")
    elif freq == 1 and tx_type == 'transfer' and amount > 200:
        score += 0.1
        signals.append("rare_recip_high_amount")

    dl = desc.lower()
    if 'donation' in dl and not regular and amount > 200:
        score += 0.2
        signals.append("large_donation")
    if 'emergency fund' in dl:
        score += 0.3
        signals.append("emergency_fund")
    if 'savings transfer' in dl and amount > 1000:
        score += 0.3
        signals.append("large_savings_transfer")

    if amount > 2000 and not regular and not cash:
        score += 0.5
        signals.append(f"critical_{int(amount)}EUR")
    elif amount > 500 and not regular and not cash and tx_type == 'transfer':
        score += 0.15
        signals.append("high_amount_transfer")

    return {"score": min(round(score, 2), 1.0), "signals": signals,
            "hint": "high_risk" if score >= 0.5 else "suspicious" if score >= 0.2 else "low_risk"}


def check_phishing(tx, sms_list, mails_list):
    sender = tx.get('sender_id', '')
    if is_regular(tx) or is_cash(tx) or sender not in known_users:
        return {"score": 0, "signals": [], "hint": "low_risk"}
    user = known_users[sender]
    signals, score = [], 0.0
    tx_time = pd.to_datetime(tx['timestamp'])

    for item in sms_list:
        t = str(item) if not isinstance(item, dict) else str(item.get('sms', ''))
        if user['phone'] not in t:
            continue
        if any(k in t.lower() for k in phishing_keywords):
            try:
                ds = t.split('Date:')[1].split('\n')[0].strip()
                st = pd.to_datetime(ds)
                d = (tx_time - st).days
                if 0 <= d <= 30:
                    score += 0.4
                    signals.append(f"phishing_sms_{d}d")
                    break
            except Exception:
                continue

    for item in mails_list:
        t = str(item) if not isinstance(item, dict) else str(item.get('mail', ''))
        if user['mail'] not in t.lower():
            continue
        if any(k in t.lower() for k in phishing_keywords):
            score += 0.3
            signals.append("phishing_mail")
            break

    return {"score": min(round(score, 2), 1.0), "signals": signals,
            "hint": "high_risk" if score >= 0.4 else "suspicious" if score > 0 else "low_risk"}


def check_gps(tx, locations):
    sender = tx.get('sender_id', '')
    if sender not in known_users or is_regular(tx) or is_cash(tx):
        return {"score": 0, "signals": [], "hint": "low_risk"}
    tx_time = pd.to_datetime(tx['timestamp'])
    df_loc = pd.DataFrame(locations)
    df_loc['timestamp'] = pd.to_datetime(df_loc['timestamp'])
    diff = (df_loc['timestamp'] - tx_time).abs()
    nearby = df_loc[(df_loc['biotag'] == sender) & (diff < pd.Timedelta(hours=24))]
    signals, score = [], 0.0
    if len(nearby) == 0:
        score += 0.1
        signals.append("no_gps")
    else:
        cities = nearby['city'].unique()
        if known_users[sender]['city'] not in cities:
            score += 0.4
            signals.append(f"gps_mismatch_{cities[0]}")
    return {"score": min(round(score, 2), 1.0), "signals": signals,
            "hint": "high_risk" if score >= 0.4 else "suspicious" if score > 0 else "low_risk"}


def check_user_profile(tx):
    sender = tx.get('sender_id', '')
    if sender not in known_users or is_regular(tx) or is_cash(tx):
        return {"score": 0, "signals": [], "hint": "low_risk"}
    user = known_users[sender]
    amount = float(tx.get('amount', 0))
    signals, score = [], 0.0
    if amount > user['salary'] * 2:
        score += 0.4
        signals.append(f"exceeds_2x_{round(user['salary'])}EUR")
    elif amount > user['salary']:
        score += 0.2
        signals.append(f"exceeds_salary_{round(user['salary'])}EUR")
    return {"score": min(round(score, 2), 1.0), "signals": signals,
            "hint": "high_risk" if score >= 0.4 else "suspicious" if score > 0 else "low_risk"}


def check_burst(tx):
    sender = tx.get('sender_id', '')
    if sender not in known_users or is_cash(tx):
        return {"score": 0, "signals": [], "hint": "low_risk"}
    tx_time = pd.to_datetime(tx['timestamp'])
    signals, score = [], 0.0
    for pt in _tx_history[sender][-3:]:
        dm = abs((tx_time - pt).total_seconds() / 60)
        if dm < 10:
            score += 0.3
            signals.append(f"burst_{dm:.0f}min")
            break
    _tx_history[sender].append(tx_time)
    return {"score": min(round(score, 2), 1.0), "signals": signals,
            "hint": "high_risk" if score >= 0.3 else "low_risk"}


def score_transaction(tx, sms_list, mails_list, locations):
    """Combine all checks into a single weighted risk score."""
    r1 = check_rules(tx)
    r2 = check_phishing(tx, sms_list, mails_list)
    r3 = check_gps(tx, locations)
    r4 = check_user_profile(tx)
    r5 = check_burst(tx)
    all_signals = r1['signals'] + r2['signals'] + r3['signals'] + r4['signals'] + r5['signals']
    combined = round(
        r1['score'] * 0.40 + r2['score'] * 0.20 + r3['score'] * 0.15 + r4['score'] * 0.15 + r5['score'] * 0.10, 2
    )
    hint = "high_risk" if combined >= 0.4 else "suspicious" if combined >= 0.15 else "low_risk"
    amount = float(tx.get('amount', 0))
    if amount > 2000 and not is_regular(tx) and not is_cash(tx):
        hint = "high_risk"
    return {"combined_score": combined, "signal_count": len(all_signals),
            "signals": all_signals, "hint": hint, "amount": amount, "zscore": get_zscore(tx),
            "details": {"rules": r1, "phishing": r2, "gps": r3, "profile": r4, "burst": r5}}
