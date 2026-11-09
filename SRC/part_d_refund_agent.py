
import sys
import json
import time
import sqlite3
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone

import mock_llm
from part_d_rag import retrieve


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"


# --------------------------------------------------
# TOOL DEFINITIONS
# --------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Get order details using an order ID.",
        "parameters": {
            "order_id": "string"
        }
    },
    {
        "name": "search_policy",
        "description": "Search the refund policy documents.",
        "parameters": {
            "query": "string"
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund. Full refund if delivery exceeds 45 minutes.",
        "parameters": {
            "order_amount": "number",
            "delivery_minutes": "number"
        }
    },
    {
        "name": "issue_refund",
        "description": "Issue an approved refund.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Send a refund request for human review.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        }
    }
]


# --------------------------------------------------
# DATABASE HELPER
# --------------------------------------------------



def get_order(order_id):
    import pandas as pd

    orders_path = ROOT / "data" / "raw" / "orders.csv"

    if not orders_path.exists():
        return {"error": "order not found"}

    df = pd.read_csv(orders_path)

    match = df[df["order_id"].astype(str).str.strip() == order_id]

    if match.empty:
        return {"error": "order not found"}

    order = match.iloc[0]

    amount = pd.to_numeric(order["order_amount"], errors="coerce")
    delivery = pd.to_numeric(order["delivery_minutes"], errors="coerce")

    if pd.isna(amount):
        return {"error": "order amount unavailable"}

    # Missing delivery time means the order cannot qualify
    # for a refund under the 45-minute rule.
    if pd.isna(delivery):
        delivery = 0

    return {
        "order_id": order_id,
        "status": str(order["status"]),
        "order_amount": float(amount),
        "delivery_minutes": float(delivery)
    }



def search_policy(query):
    results = retrieve(query, k=3)
    return {"results": results}


# --------------------------------------------------
# REFUND CALCULATION
# --------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
    amount = float(order_amount)
    minutes = float(delivery_minutes)

    if minutes > 45:
        refund = amount
    else:
        refund = 0

    return {
        "order_amount": amount,
        "delivery_minutes": minutes,
        "refund_inr": round(refund, 2)
    }


# --------------------------------------------------
# LEDGER HELPERS
# --------------------------------------------------

def write_ledger(tool_name, order_id, refund_inr):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with open(LEDGER_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {
        "status": "success",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }


def issue_refund(order_id, refund_inr):
    return write_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return write_ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# --------------------------------------------------
# CHECKPOINT HELPERS
# --------------------------------------------------

def save_checkpoint(run_id, scenario_id, messages,
                    tokens_used, step_number, executed_tools,
                    steps, llm_calls):

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "executed_tools": executed_tools,
        "steps": steps,
        "llm_calls": llm_calls
    }

    path = STATE_DIR / f"{run_id}.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2)

    return path


def load_checkpoint(run_id):
    path = STATE_DIR / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {run_id}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# RESPONSE HELPERS
# --------------------------------------------------

def get_message(response):
    return response.choices[0].message


def get_tokens(response, messages):
    usage = getattr(response, "usage", None)

    if usage is not None:
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            return int(total)

    # Fallback if usage is not provided
    return mock_llm.est_tokens(json.dumps(messages, default=str))


def get_tool_call_data(tool_call):
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        call_id = tool_call.get("id", str(uuid.uuid4()))
    else:
        name = getattr(tool_call, "name", None)
        arguments = getattr(tool_call, "arguments", {})
        call_id = getattr(tool_call, "id", str(uuid.uuid4()))

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    return call_id, name, arguments


def save_run(result):
    with open(RUNS_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(result, default=str) + "\n")


# --------------------------------------------------
# AGENT EXECUTION
# --------------------------------------------------

def continue_agent(
    run_id,
    checkpoint,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tool=None
):
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    steps = checkpoint["steps"]
    llm_calls = checkpoint["llm_calls"]
    executed_tools = checkpoint["executed_tools"]

    start_time = time.monotonic()

    def finish(status, final_answer):
        result = {
            "run_id": run_id,
            "scenario_id": checkpoint["scenario_id"],
            "status": status,
            "final_answer": final_answer,
            "steps": steps,
            "llm_calls": llm_calls,
            "tokens_used": tokens_used
        }
        return result

    while llm_calls < max_iterations:

        if time.monotonic() - start_time >= time_limit_seconds:
            return finish(
                "FALLBACK_TIMEOUT",
                "I could not complete this request within the time limit. "
                "A human agent should review it."
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1

        tokens_used += get_tokens(response, messages)

        if tokens_used > token_limit:
            return finish(
                "FALLBACK_TOKEN_LIMIT",
                "I could not complete this request within the token limit. "
                "A human agent should review it."
            )

        message = get_message(response)
        tool_calls = getattr(message, "tool_calls", None)
        content = getattr(message, "content", None)

        # No tool calls means the model has completed its answer.
        if not tool_calls:
            answer_text = content or "I could not find enough information."
            return finish("COMPLETED", answer_text)

        # Add assistant tool-call message to conversation.
        assistant_tool_calls = []

        for call in tool_calls:
            call_id, name, arguments = get_tool_call_data(call)

            assistant_tool_calls.append({
                "id": call_id,
                "name": name,
                "arguments": arguments
            })

        messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": assistant_tool_calls
        })

        for call in assistant_tool_calls:
            name = call["name"]
            arguments = call["arguments"]
            call_id = call["id"]

            if name not in TOOL_MAP:
                observation = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    observation = TOOL_MAP[name](**arguments)
                except Exception as error:
                    observation = {"error": str(error)}

            steps.append({
                "tool": name,
                "arguments": arguments,
                "observation": observation
            })

            executed_tools.append(name)

            messages.append({
                "role": "tool",
                "name": name,
                "tool_call_id": call_id,
                "content": json.dumps(observation, default=str)
            })

            # Save checkpoint after every tool observation.
            checkpoint.update({
                "messages": messages,
                "tokens_used": tokens_used,
                "step_number": len(steps),
                "executed_tools": executed_tools,
                "steps": steps,
                "llm_calls": llm_calls
            })

            save_checkpoint(
                run_id,
                checkpoint["scenario_id"],
                messages,
                tokens_used,
                len(steps),
                executed_tools,
                steps,
                llm_calls
            )

            # Simulate a crash after a specified tool observation.
            if crash_after_tool is not None and len(steps) == crash_after_tool:
                raise RuntimeError("Simulated crash. Checkpoint saved.")

    return finish(
        "FALLBACK_MAX_ITERATIONS",
        "I could not complete this request within the maximum "
        "number of steps. A human agent should review it."
    )


# --------------------------------------------------
# START A NEW AGENT RUN
# --------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tool=None
):
    run_id = str(uuid.uuid4())

    system_prompt = """
You are the Instamart customer refund support AGENT.

Follow these rules:
1. Ask for an order ID if the customer has not provided one.
2. Only completed orders are eligible for a refund.
3. Search the refund policy before making a refund decision.
4. Refund 100% of the order amount only when delivery took more than
   45 minutes. Otherwise refund is zero.
5. If the refund amount is greater than INR 1000, escalate to a human.
6. If the order cannot be found, explain that it was not found.
7. Never issue a refund for an ineligible order.
8. Use the available tools to complete the task.
"""

    messages = [
        {"role": "system", "content": "AGENT\n" + system_prompt},
        {"role": "user", "content": user_message}
    ]

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": 0,
        "step_number": 0,
        "executed_tools": [],
        "steps": [],
        "llm_calls": 0
    }

    save_checkpoint(
        run_id, scenario_id, messages, 0, 0, [], [], 0
    )

    result = continue_agent(
        run_id,
        checkpoint,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tool=crash_after_tool
    )

    return result


# --------------------------------------------------
# RESUME A SAVED RUN
# --------------------------------------------------

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    checkpoint = load_checkpoint(run_id)

    result = continue_agent(
        run_id,
        checkpoint,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )

    return result


# --------------------------------------------------
# SCENARIO FILE
# --------------------------------------------------

def load_scenarios():
    import pandas as pd

    path = ROOT / "data" / "raw" / "agent_scenarios.csv"

    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    df = pd.read_csv(path)

    id_column = next(
        (c for c in df.columns if c.lower() in
         ["scenario_id", "id"]),
        None
    )

    message_column = next(
        (c for c in df.columns if c.lower() in
         ["user_message", "message", "prompt", "customer_message"]),
        None
    )

    if id_column is None or message_column is None:
        raise ValueError(
            "Could not identify scenario ID and message columns. "
            f"Available columns: {list(df.columns)}"
        )

    return [
        (str(row[id_column]), str(row[message_column]))
        for _, row in df.iterrows()
    ]


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    print("\n========== PART D: REFUND AGENT ==========\n")

    scenarios = load_scenarios()

    print("SCENARIOS FOUND:", len(scenarios))

    # Run all seven scenarios.
    for scenario_id, user_message in scenarios:

        print(f"\nRunning {scenario_id}...")

        try:
            # Simulate crash after the second tool observation in S1.
            if scenario_id.upper() == "S1":
                try:
                    result = run_agent(
                        user_message,
                        scenario_id,
                        crash_after_tool=2
                    )
                except RuntimeError:
                    print("Simulated crash. Resuming from checkpoint.")

                    # Find the most recently saved checkpoint.
                    checkpoint_files = sorted(
                        STATE_DIR.glob("*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    )

                    checkpoint = json.loads(
                        checkpoint_files[0].read_text(encoding="utf-8")
                    )

                    result = resume_agent(checkpoint["run_id"])

            else:
                result = run_agent(user_message, scenario_id)

            print("STATUS:", result["status"])
            print("ANSWER:", result["final_answer"])
            print("TOOLS:", [s["tool"] for s in result["steps"]])
            print("LLM CALLS:", result["llm_calls"])
            print("TOKENS:", result["tokens_used"])

            save_run(result)

        except Exception as error:
            print("ERROR:", error)

    # Token limit demonstration.
    print("\n========== TOKEN LIMIT DEMO ==========")

    if scenarios:
        demo_id, demo_message = scenarios[0]
        token_result = run_agent(
            demo_message,
            "TOKEN_LIMIT_DEMO",
            token_limit=200
        )
        print("STATUS:", token_result["status"])

    # Timeout demonstration.
    print("\n========== TIMEOUT DEMO ==========")

    if scenarios:
        demo_id, demo_message = scenarios[0]
        timeout_result = run_agent(
            demo_message,
            "TIMEOUT_DEMO",
            time_limit_seconds=0
        )
        print("STATUS:", timeout_result["status"])

    print("\n========== PART D AGENT FINISHED ==========")
    print("Run log:", RUNS_FILE)
    print("Refund ledger:", LEDGER_FILE)
    print("Checkpoints:", STATE_DIR)
