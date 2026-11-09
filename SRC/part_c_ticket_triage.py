
import pandas as pd
import re
import json
import time
import sys
from pathlib import Path

# --------------------------------------------------
# 1. SET PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
import mock_llm


# --------------------------------------------------
# 2. LOAD TICKETS
# --------------------------------------------------

tickets = pd.read_csv(DATA_DIR / "tickets.csv")

print("\nTICKET DATA")
print("Columns:", list(tickets.columns))
print("\nMissing values:")
print(tickets.isnull().sum())
print("\nFirst five tickets:")
print(tickets.head())


# --------------------------------------------------
# 3. CHECK EMPTY AND LONG TICKETS
# --------------------------------------------------

tickets["text"] = tickets["text"].fillna("").astype(str)

tickets["token_count"] = tickets["text"].apply(
    mock_llm.est_tokens
)

tickets["status"] = "ready"

tickets.loc[
    tickets["text"].str.strip() == "",
    "status"
] = "skipped_empty"

tickets.loc[
    (tickets["token_count"] > 400) &
    (tickets["status"] == "ready"),
    "status"
] = "flagged_long"

print("\nTICKET STATUS SUMMARY")
print(tickets["status"].value_counts())

print("Empty tickets:", (tickets["status"] == "skipped_empty").sum())
print("Long tickets:", (tickets["status"] == "flagged_long").sum())
print("Ready tickets:", (tickets["status"] == "ready").sum())


# --------------------------------------------------
# 4. REDACT PERSONAL INFORMATION
# --------------------------------------------------

def redact_pii(text):
    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text
    )

    text = re.sub(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text
    )

    text = re.sub(
        r"ORD-\d{6}",
        "[ORDER_ID]",
        text
    )

    return text


original_text = tickets["text"]

email_count = original_text.apply(
    lambda x: len(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", x))
).sum()

phone_count = original_text.apply(
    lambda x: len(re.findall(r"\b[6-9]\d{9}\b", x))
).sum()

order_count = original_text.apply(
    lambda x: len(re.findall(r"ORD-\d{6}", x))
).sum()

tickets["redacted_text"] = ""

ready_mask = tickets["status"] == "ready"

tickets.loc[ready_mask, "redacted_text"] = (
    tickets.loc[ready_mask, "text"].apply(redact_pii)
)

print("\nPII REDACTION COMPLETED")
print("Emails found:", email_count)
print("Phone numbers found:", phone_count)
print("Order IDs found:", order_count)

print("\nSAMPLE REDACTED MESSAGES")
print(tickets[["text", "redacted_text"]].head(10))


# --------------------------------------------------
# 5. KEYWORD FALLBACK CLASSIFIER
# --------------------------------------------------

ALLOWED_CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]


def keyword_classify(text):
    text = text.lower()

    keywords = {
        "fraud": [
            "unauthorized", "fraud", "hacked", "stolen"
        ],
        "billing": [
            "charged", "invoice", "bill",
            "refund", "deducted", "payment"
        ],
        "technical": [
            "app", "crash", "login", "otp",
            "error", "freezes"
        ],
        "delivery": [
            "arrived", "rider", "delivery",
            "delivered", "late"
        ]
    }

    for category in [
        "fraud", "technical", "billing", "delivery"
    ]:
        if any(word in text for word in keywords[category]):
            return category

    return "technical"


# --------------------------------------------------
# 6. LLM CLASSIFICATION WITH RETRIES
# --------------------------------------------------

def classify_ticket(text, few_shot=False):

    system_prompt = (
        "You are a CLASSIFY support ticket assistant. "
        "Return only JSON with category and reason. "
        "Allowed categories: billing, delivery, technical, fraud."
    )

    if few_shot:
        system_prompt += """
        EXAMPLES:
        Payment deducted: billing
        App crashes: technical
        Unauthorized transaction: fraud
        Late rider: delivery
        """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": text
        }
    ]

    for attempt in range(3):
        try:
            response = mock_llm.chat(messages)

            answer = response.choices[0].message.content

            try:
                result = json.loads(answer)

                if result.get("category") in ALLOWED_CATEGORIES:
                    return result["category"], "llm_json"

                return keyword_classify(text), "keyword_fallback"

            except (json.JSONDecodeError, TypeError):
                return keyword_classify(text), "keyword_fallback"

        except mock_llm.TransientLLMError:

            if attempt == 2:
                return keyword_classify(text), "keyword_fallback"

            time.sleep(0.2)

    return keyword_classify(text), "keyword_fallback"


# --------------------------------------------------
# 7. ZERO-SHOT CLASSIFICATION
# --------------------------------------------------

mock_llm.reset()

tickets["zero_shot_category"] = ""
tickets["zero_shot_method"] = ""

ready_indices = tickets.index[
    tickets["status"] == "ready"
]

for index in ready_indices:

    category, method = classify_ticket(
        tickets.at[index, "redacted_text"],
        few_shot=False
    )

    tickets.at[index, "zero_shot_category"] = category
    tickets.at[index, "zero_shot_method"] = method

print("\nZERO-SHOT CLASSIFICATION")
print(tickets["zero_shot_method"].value_counts())


# --------------------------------------------------
# 8. FEW-SHOT CLASSIFICATION
# --------------------------------------------------

mock_llm.reset()

tickets["few_shot_category"] = ""
tickets["few_shot_method"] = ""

for index in ready_indices:

    category, method = classify_ticket(
        tickets.at[index, "redacted_text"],
        few_shot=True
    )

    tickets.at[index, "few_shot_category"] = category
    tickets.at[index, "few_shot_method"] = method

print("\nFEW-SHOT CLASSIFICATION")
print(tickets["few_shot_method"].value_counts())


# --------------------------------------------------
# 9. EVALUATE CLASSIFICATION ACCURACY
# --------------------------------------------------

ready_tickets = tickets[
    tickets["status"] == "ready"
].copy()

true_categories = ready_tickets["true_category"].str.lower()

zero_accuracy = (
    ready_tickets["zero_shot_category"] == true_categories
).mean() * 100

few_accuracy = (
    ready_tickets["few_shot_category"] == true_categories
).mean() * 100

print("\nCLASSIFICATION ACCURACY")
print(f"Zero-shot accuracy: {zero_accuracy:.2f}%")
print(f"Few-shot accuracy: {few_accuracy:.2f}%")


# --------------------------------------------------
# 10. ASSIGN SUPPORT TEAMS
# --------------------------------------------------

team_mapping = {
    "billing": "Payments Team",
    "delivery": "Delivery Team",
    "technical": "Technical Support Team",
    "fraud": "Trust and Safety Team"
}

ready_tickets["assigned_team"] = (
    ready_tickets["few_shot_category"].map(team_mapping)
)


# --------------------------------------------------
# 11. ADD CUSTOMER SEGMENTS
# --------------------------------------------------

segments_path = OUTPUT_DIR / "customer_segments.csv"

if segments_path.exists():

    segments = pd.read_csv(segments_path)

    ready_tickets = ready_tickets.merge(
        segments[["customer_id", "segment_name"]],
        on="customer_id",
        how="left"
    )

    ready_tickets["segment_name"] = (
        ready_tickets["segment_name"].fillna("Unknown")
    )

else:
    ready_tickets["segment_name"] = "Unknown"


# --------------------------------------------------
# 12. SAVE TRIAGE RESULTS
# --------------------------------------------------

result_columns = [
    "ticket_id",
    "customer_id",
    "true_category",
    "zero_shot_category",
    "few_shot_category",
    "assigned_team",
    "segment_name"
]

ready_tickets[result_columns].to_csv(
    OUTPUT_DIR / "triage_results.csv",
    index=False
)

print("\nTICKET RESULTS SAVED")
print("File: outputs/triage_results.csv")

print("\nASSIGNED TEAMS")
print(ready_tickets["assigned_team"].value_counts())


# --------------------------------------------------
# 13. BUDGET DEMONSTRATION
# --------------------------------------------------

budget_demo = ready_tickets[
    ["ticket_id", "few_shot_category", "few_shot_method"]
].copy()

budget_demo["estimated_tokens"] = (
    ready_tickets["redacted_text"].apply(mock_llm.est_tokens)
)

budget_demo["estimated_cost_inr"] = 0.0

budget_demo.to_csv(
    OUTPUT_DIR / "triage_budget_demo.csv",
    index=False
)

print("\nBUDGET DEMONSTRATION")
print("Tickets processed:", len(budget_demo))
print(
    "Estimated input tokens:",
    budget_demo["estimated_tokens"].sum()
)
print("Mock API cost: INR 0")
print("Saved: outputs/triage_budget_demo.csv")

print("\nPART C COMPLETED SUCCESSFULLY!")
