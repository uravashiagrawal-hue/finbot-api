from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import json
import os

# ─────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# LOAD DATA ONCE AT STARTUP
# ─────────────────────────────────────────────

all_transactions = pd.read_csv('transactions.csv')
all_cashflow     = pd.read_csv('daily_cashflow.csv')
available_biz    = sorted(all_transactions['business_id'].unique().tolist())

# ─────────────────────────────────────────────
# HUGGING FACE CONFIG
# ─────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
HF_URL     = "https://router.huggingface.co/v1/chat/completions"

chat_sessions = {}  # { business_id: [ {"role":..,"content":..} ] }


# ─────────────────────────────────────────────
# HELPER: Anomaly Detection
# ─────────────────────────────────────────────

def detect_anomalies(expenses_df):
    """Find transactions that are 3x above their category average."""
    if expenses_df.empty:
        return []

    anomalies = []
    cat_avg = expenses_df.groupby('category')['amount'].mean()

    for _, row in expenses_df.iterrows():
        avg = cat_avg.get(row['category'], 0)
        if avg > 0 and row['amount'] > 3 * avg:
            anomalies.append({
                "date":        row['date'],
                "description": row['description'],
                "amount":      round(row['amount'], 2),
                "category":    row['category'],
                "normal_avg":  round(avg, 2),
                "times_higher": round(row['amount'] / avg, 1)
            })

    # Sort by most extreme first
    anomalies.sort(key=lambda x: x['times_higher'], reverse=True)
    return anomalies[:5]


# ─────────────────────────────────────────────
# HELPER: Industry Benchmark (avg across all businesses)
# ─────────────────────────────────────────────

def build_industry_benchmark():
    """Compute average metrics across ALL businesses."""
    benchmark = {}

    for biz in available_biz:
        df = all_transactions[all_transactions['business_id'] == biz]
        expenses = df[df['type'] == 'expense']
        income   = df[df['type'] == 'income']

        total_expense = expenses['amount'].sum()
        total_income  = income['amount'].sum()

        if total_income > 0:
            cat_ratios = (expenses.groupby('category')['amount'].sum() / total_expense * 100).to_dict()
            benchmark[biz] = {
                "total_income":   total_income,
                "total_expense":  total_expense,
                "net_profit":     total_income - total_expense,
                "profit_margin":  round((total_income - total_expense) / total_income * 100, 1),
                "cat_ratios":     cat_ratios
            }

    # Industry averages
    all_margins   = [v['profit_margin'] for v in benchmark.values()]
    all_incomes   = [v['total_income']  for v in benchmark.values()]
    all_expenses  = [v['total_expense'] for v in benchmark.values()]

    # Category-wise average ratio across all businesses
    all_cats = {}
    for v in benchmark.values():
        for cat, ratio in v['cat_ratios'].items():
            all_cats.setdefault(cat, []).append(ratio)

    avg_cat_ratios = {cat: round(sum(vals)/len(vals), 1) for cat, vals in all_cats.items()}

    return {
        "per_business": benchmark,
        "avg_profit_margin": round(sum(all_margins)/len(all_margins), 1),
        "avg_total_income":  round(sum(all_incomes)/len(all_incomes), 2),
        "avg_total_expense": round(sum(all_expenses)/len(all_expenses), 2),
        "avg_category_ratios": avg_cat_ratios,
        "total_businesses": len(available_biz)
    }

INDUSTRY_BENCHMARK = build_industry_benchmark()  # computed once at startup


# ─────────────────────────────────────────────
# HELPER: Build comparison context for a business
# ─────────────────────────────────────────────

def build_comparison_context(biz_id: str) -> str:
    """Compare this business vs industry average."""
    biz_data = INDUSTRY_BENCHMARK['per_business'].get(biz_id)
    if not biz_data:
        return "No comparison data available."

    avg_margin  = INDUSTRY_BENCHMARK['avg_profit_margin']
    avg_income  = INDUSTRY_BENCHMARK['avg_total_income']
    avg_expense = INDUSTRY_BENCHMARK['avg_total_expense']
    avg_cats    = INDUSTRY_BENCHMARK['avg_category_ratios']

    biz_margin  = biz_data['profit_margin']
    biz_cats    = biz_data['cat_ratios']

    # Rank this business by profit margin
    all_margins = sorted(
        [(b, d['profit_margin']) for b, d in INDUSTRY_BENCHMARK['per_business'].items()],
        key=lambda x: x[1], reverse=True
    )
    rank = next((i+1 for i, (b, _) in enumerate(all_margins) if b == biz_id), None)

    # Category comparison
    cat_comparison = []
    for cat, biz_ratio in biz_cats.items():
        ind_ratio = avg_cats.get(cat, 0)
        diff = round(biz_ratio - ind_ratio, 1)
        status = "OVERSPENDING" if diff > 5 else ("🟢 EFFICIENT" if diff < -5 else "🟡 NORMAL")
        cat_comparison.append(
            f"  - {cat}: {biz_ratio:.1f}% vs industry avg {ind_ratio:.1f}% → {'+' if diff>0 else ''}{diff}% {status}"
        )

    comparison_str = f"""
=== PEER COMPARISON FOR {biz_id} ===
Profit Margin    : {biz_margin}% vs industry avg {avg_margin}% → {'🟢 Above Average' if biz_margin > avg_margin else '🔴 Below Average'}
Total Income     : ₹{biz_data['total_income']:,.0f} vs industry avg ₹{avg_income:,.0f}
Total Expense    : ₹{biz_data['total_expense']:,.0f} vs industry avg ₹{avg_expense:,.0f}
Industry Rank    : #{rank} out of {INDUSTRY_BENCHMARK['total_businesses']} businesses

Expense Category Breakdown vs Industry:
{chr(10).join(cat_comparison)}
"""
    return comparison_str

# HELPER: Build full financial context

def build_financial_context(biz_id: str):
    df = all_transactions[all_transactions['business_id'] == biz_id].copy()

    cashflow_df = (
        all_cashflow[all_cashflow['business_id'] == biz_id].copy()
        if 'business_id' in all_cashflow.columns
        else all_cashflow.copy()
    )

    expenses_df = df[df['type'] == 'expense']
    income_df   = df[df['type'] == 'income']

    biz_total_income  = income_df['amount'].sum()
    biz_total_expense = expenses_df['amount'].sum()
    biz_net_profit    = biz_total_income - biz_total_expense
    current_balance   = cashflow_df['cumulative_balance'].iloc[-1]
    avg_cashflow      = cashflow_df['net_cashflow'].mean()
    profit_margin     = round(biz_net_profit / biz_total_income * 100, 1) if biz_total_income > 0 else 0

    top_category   = expenses_df.groupby('category')['amount'].sum().idxmax()
    top_cat_amount = expenses_df.groupby('category')['amount'].sum().max()

    expense_breakdown = (
        expenses_df.groupby('category')['amount']
        .sum().sort_values(ascending=False).head(5)
    )

    recent_txns = df.sort_values('date', ascending=False).head(5)[
        ['date', 'description', 'amount', 'type', 'category']
    ]

    df['month'] = pd.to_datetime(df['date']).dt.strftime('%B %Y')
    monthly_summary = (
        df.groupby(['month', 'type'])['amount']
        .sum().unstack(fill_value=0).reset_index()
    )
    monthly_summary.columns = [str(c).lower() for c in monthly_summary.columns]

    # Anomalies
    anomalies = detect_anomalies(expenses_df)
    anomaly_lines = "\n".join([
        f"  {a['date']} | {a['description']} | ₹{a['amount']:,.0f} "
        f"({a['times_higher']}x above normal avg ₹{a['normal_avg']:,.0f})"
        for a in anomalies
    ]) or "No anomalies detected."

    expense_lines = "\n".join([f"  - {cat}: ₹{amt:,.0f}" for cat, amt in expense_breakdown.items()])
    monthly_lines = "\n".join(
        [f"  - {row['month']}: Income ₹{row.get('income',0):,.0f} | Expense ₹{row.get('expense',0):,.0f}"
         for _, row in monthly_summary.iterrows()]
    )
    recent_lines = "\n".join(
        [f"  - {row['date']} | {row['description']} | ₹{row['amount']:,.0f} ({row['type']} - {row['category']})"
         for _, row in recent_txns.iterrows()]
    )

    # Comparison vs industry
    comparison_str = build_comparison_context(biz_id)

    context_str = f"""
=== FINANCIAL SUMMARY FOR {biz_id} ===
Business Income  : ₹{biz_total_income:,.0f}
Business Expense : ₹{biz_total_expense:,.0f}
Net Profit       : ₹{biz_net_profit:,.0f}
Profit Margin    : {profit_margin}%
Current Balance  : ₹{current_balance:,.0f}
Avg Daily CF     : ₹{avg_cashflow:,.0f}
Top Expense Category: {top_category} (₹{top_cat_amount:,.0f})

Top 5 Expense Categories:
{expense_lines}

Monthly Breakdown:
{monthly_lines}

Recent Transactions:
{recent_lines}

🚨 Anomaly Alerts:
{anomaly_lines}

{comparison_str}

Total Transactions: {len(df):,}
Date Range: {df['date'].min()} to {df['date'].max()}
"""

    summary_dict = {
        "business_id":          biz_id,
        "total_income":         round(biz_total_income, 2),
        "total_expense":        round(biz_total_expense, 2),
        "net_profit":           round(biz_net_profit, 2),
        "profit_margin":        profit_margin,
        "current_balance":      round(float(current_balance), 2),
        "avg_daily_cashflow":   round(float(avg_cashflow), 2),
        "top_expense_category": top_category,
        "top_expense_amount":   round(top_cat_amount, 2),
        "expense_breakdown":    {cat: round(amt, 2) for cat, amt in expense_breakdown.items()},
        "anomalies":            anomalies,
        "total_transactions":   len(df),
        "date_range":           {"from": df['date'].min(), "to": df['date'].max()},
    }

    return summary_dict, context_str


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

# 1. GET /health
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"success": True, "status": "FinBot API is running 💹"})


# 2. GET /businesses
@app.route('/businesses', methods=['GET'])
def get_businesses():
    return jsonify({
        "success": True,
        "businesses": available_biz,
        "count": len(available_biz)
    })


# 3. GET /summary/<business_id>
@app.route('/summary/<business_id>', methods=['GET'])
def get_summary(business_id):
    biz_id = business_id.upper()
    if biz_id not in available_biz:
        return jsonify({"success": False, "error": f"Business '{biz_id}' not found."}), 404
    summary, _ = build_financial_context(biz_id)
    return jsonify({"success": True, "data": summary})


# 4. GET /compare/<business_id>  ← NEW
@app.route('/compare/<business_id>', methods=['GET'])
def compare_business(business_id):
    """
    Compare a business vs industry average.
    Returns structured comparison data for frontend charts.
    """
    biz_id = business_id.upper()
    if biz_id not in available_biz:
        return jsonify({"success": False, "error": f"Business '{biz_id}' not found."}), 404

    biz_data = INDUSTRY_BENCHMARK['per_business'].get(biz_id)
    avg_cats = INDUSTRY_BENCHMARK['avg_category_ratios']

    # Rank
    all_margins = sorted(
        [(b, d['profit_margin']) for b, d in INDUSTRY_BENCHMARK['per_business'].items()],
        key=lambda x: x[1], reverse=True
    )
    rank = next((i+1 for i, (b, _) in enumerate(all_margins) if b == biz_id), None)

    return jsonify({
        "success": True,
        "data": {
            "business_id":           biz_id,
            "profit_margin":         biz_data['profit_margin'],
            "industry_avg_margin":   INDUSTRY_BENCHMARK['avg_profit_margin'],
            "rank":                  rank,
            "total_businesses":      INDUSTRY_BENCHMARK['total_businesses'],
            "total_income":          biz_data['total_income'],
            "industry_avg_income":   INDUSTRY_BENCHMARK['avg_total_income'],
            "total_expense":         biz_data['total_expense'],
            "industry_avg_expense":  INDUSTRY_BENCHMARK['avg_total_expense'],
            "category_vs_industry":  {
                cat: {
                    "business_pct": round(biz_data['cat_ratios'].get(cat, 0), 1),
                    "industry_avg_pct": avg_cats.get(cat, 0),
                    "difference": round(biz_data['cat_ratios'].get(cat, 0) - avg_cats.get(cat, 0), 1)
                }
                for cat in avg_cats
            }
        }
    })


# 5. GET /anomalies/<business_id>  ← NEW
@app.route('/anomalies/<business_id>', methods=['GET'])
def get_anomalies(business_id):
    """Returns anomalous transactions for a business."""
    biz_id = business_id.upper()
    if biz_id not in available_biz:
        return jsonify({"success": False, "error": f"Business '{biz_id}' not found."}), 404

    df = all_transactions[all_transactions['business_id'] == biz_id]
    expenses_df = df[df['type'] == 'expense']
    anomalies = detect_anomalies(expenses_df)

    return jsonify({
        "success": True,
        "business_id": biz_id,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies
    })


# 6. POST /register  ← NEW — New business enrollment
@app.route('/register', methods=['POST'])
def register_business():
    """
    Register a new business and add their transactions.

    Request body:
    {
        "business_id": "BIZ_013",
        "transactions": [
            {
                "date": "2026-01-15",
                "description": "Office rent",
                "amount": 25000,
                "type": "expense",
                "category": "rent"
            },
            ...
        ]
    }
    """
    global all_transactions, available_biz

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    business_id  = data.get('business_id', '').strip().upper()
    transactions = data.get('transactions', [])

    if not business_id:
        return jsonify({"success": False, "error": "business_id is required"}), 400
    if not transactions:
        return jsonify({"success": False, "error": "transactions list is required"}), 400
    if business_id in available_biz:
        return jsonify({"success": False, "error": f"Business '{business_id}' already exists."}), 409

    # Build dataframe from new transactions
    new_df = pd.DataFrame(transactions)
    new_df['business_id'] = business_id
    new_df['user_id']     = f"U_{business_id}"
    new_df['transaction_id'] = [f"TXN_{business_id}_{i}" for i in range(len(new_df))]

    # Append to global dataset
    all_transactions = pd.concat([all_transactions, new_df], ignore_index=True)

    # Save back to CSV
    all_transactions.to_csv('transactions.csv', index=False)

    # Update available businesses
    available_biz = sorted(all_transactions['business_id'].unique().tolist())

    return jsonify({
        "success": True,
        "message": f"Business '{business_id}' registered successfully!",
        "business_id": business_id,
        "transactions_added": len(transactions)
    })


# 7. POST /chat  ← UPGRADED with comparison + anomaly awareness
@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chatbot endpoint.
    Request: { "business_id": "BIZ_001", "message": "...", "reset": false }
    Response: { "success": true, "reply": "...", "business_id": "BIZ_001" }
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    business_id  = data.get('business_id', '').strip().upper()
    user_message = data.get('message', '').strip()
    reset        = data.get('reset', False)

    if not business_id:
        return jsonify({"success": False, "error": "business_id is required"}), 400
    if not user_message:
        return jsonify({"success": False, "error": "message is required"}), 400
    if business_id not in available_biz:
        return jsonify({"success": False, "error": f"Business '{business_id}' not found."}), 404

    if reset or business_id not in chat_sessions:
        chat_sessions[business_id] = []

    _, context_str = build_financial_context(business_id)

    system_instruction = f"""You are FinBot, an expert AI financial assistant for small and medium-sized businesses.
You are currently assisting business: {business_id}

You have access to the following REAL financial data:
{context_str}

RULES — follow these strictly:
1. Use EXACT numbers from the data — never guess or make up figures
2. Format all amounts as ₹X,XX,XXX
3. When asked about anomalies or alerts → highlight the Anomaly Alerts section
4. When asked about comparison or peers → use the Peer Comparison section
5. Give 2-3 specific actionable tips, not generic advice
6. If data is not available for a question, say "I don't have that data"
7. Keep responses under 150 words — be direct and clear
8. Always give the answer first, then the explanation

You can answer questions about:
- Income, expenses, profit, balance, cash flow
- Category-wise spending breakdown
- Monthly and seasonal trends
- Anomaly alerts and unusual spending
- How this business compares to industry peers
- Cost reduction and financial improvement advice"""

    messages = [{"role": "system", "content": system_instruction}]
    messages += chat_sessions[business_id]
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            HF_URL,
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"].strip()

            chat_sessions[business_id].append({"role": "user",      "content": user_message})
            chat_sessions[business_id].append({"role": "assistant",  "content": reply})

            if len(chat_sessions[business_id]) > 20:
                chat_sessions[business_id] = chat_sessions[business_id][-20:]

            return jsonify({"success": True, "reply": reply, "business_id": business_id})

        else:
            return jsonify({"success": False, "error": f"HF API Error {response.status_code}: {response.text}"}), 500

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "Request timed out. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": "Connection failed. Check internet or VPN."}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 8. POST /chat/reset
@app.route('/chat/reset', methods=['POST'])
def reset_chat():
    data = request.get_json()
    business_id = data.get('business_id', '').strip().upper()
    if business_id in chat_sessions:
        chat_sessions[business_id] = []
    return jsonify({"success": True, "message": f"Chat history cleared for {business_id}"})


# ─────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  💹 FinBot API Server Starting...")
    print("="*50)
    print(f"  Loaded {len(available_biz)} businesses")
    print(f"  Industry benchmark computed ✅")
    print(f"  Anomaly detection active ✅")
    print(f"  Running at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000)
