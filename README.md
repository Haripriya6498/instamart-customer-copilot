# Instamart Customer Intelligence & Support Copilot

## Project Overview

This capstone project builds a customer intelligence and support system for Instamart. It uses data cleaning, customer segmentation, machine learning, ticket classification, Retrieval-Augmented Generation (RAG), and an AI refund-support agent.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SQLite
- TF-IDF and Cosine Similarity
- Mock LLM
- JSON and CSV

## Project Structure

```text
instamart-customer-copilot/
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
├── src/
│   ├── part_a_cleaning.py
│   ├── part_a_validation.py
│   ├── part_b_segmentation.py
│   ├── part_b_lapse_model.py
│   ├── part_c_ticket_triage.py
│   ├── part_d_rag.py
│   └── part_d_refund_agent.py
├── sql/
│   └── queries.sql
├── outputs/
├── state/
├── generate_data.py
├── mock_llm.py
├── requirements.txt
└── README.md
```

## Part A — Data Cleaning and Validation

- Loaded and inspected customer, order, product, and ticket datasets.
- Handled missing values and duplicate records.
- Cleaned and validated the datasets.
- Stored the cleaned data in SQLite.
- Created SQL queries for business analysis.

## Part B — Customer Segmentation and Lapse Prediction

- Performed RFM-based customer segmentation.
- Built a customer lapse prediction model.
- Compared Logistic Regression and Random Forest.
- Evaluated models using precision, recall, F1-score, and confusion matrix.
- Selected Random Forest as the final model.
- Saved the trained model for future predictions.

### Model Results

| Metric | Result |
|---|---:|
| Accuracy | 72.4% |
| Precision | 64.0% |
| Recall | 64.0% |
| F1-score | 64.0% |

## Part C — Customer Ticket Triage

- Classified customer support tickets into categories.
- Used zero-shot and few-shot classification.
- Applied keyword fallback for invalid model responses.
- Detected and handled personally identifiable information.
- Assigned tickets to support teams.
- Generated triage results and budget reports.

### Ticket Results

| Metric | Result |
|---|---:|
| Ready tickets | 156 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Zero-shot accuracy | 86.54% |
| Few-shot accuracy | 100% |

## Part D — RAG and Refund Support Agent

### 1. Policy RAG

- Loaded five policy documents.
- Divided documents into 40-word chunks.
- Used 10-word overlap between chunks.
- Created TF-IDF vectors.
- Used cosine similarity to retrieve relevant policy chunks.
- Generated answers using the mock LLM.

**TF-IDF explanation:** Term Frequency measures how often a word appears in a document. Inverse Document Frequency gives less importance to words that appear in many documents and more importance to distinctive words.

**Limitation:** TF-IDF is based on word matching and may not understand the meaning of a question when different words express the same idea.

### 2. Refund Agent Tools

The agent uses five tools:

| Tool | Purpose |
|---|---|
| get_order | Retrieves order information |
| search_policy | Searches policy documents |
| calculate_refund | Calculates refund eligibility and amount |
| issue_refund | Records an approved refund |
| escalate_to_human | Records a request for human review |

**Refund rule:** A full refund is calculated when delivery exceeds 45 minutes. Otherwise, the refund is zero. Refunds above ₹1,000 are escalated to a human.

### 3. Scenario Evaluation

| Scenario | Expected Outcome | Actual Outcome |
|---|---|---|
| S1 | REFUND ₹128.25 | REFUND ₹128.25 |
| S2 | REFUND ₹135.00 | REFUND ₹135.00 |
| S3 | ESCALATE ₹1,999.00 | ESCALATE |
| S4 | DENIED | DENIED |
| S5 | DENIED | DENIED |
| S6 | NOT_FOUND | NOT_FOUND |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS |

All seven scenarios produced the expected outcomes.

### 4. Agent Safety Limits

| Test | Result |
|---|---|
| Maximum iterations | FALLBACK_MAX_ITERATIONS |
| Token limit | FALLBACK_TOKEN_LIMIT |
| Time limit | FALLBACK_TIMEOUT |
| Checkpoint and resume | Successful |

### 5. Checkpoint and Resume

The agent saves its execution state after every tool observation.

The S1 scenario was interrupted after its second tool observation and successfully resumed from its saved checkpoint.

Tool sequence across the interrupted and resumed run:

1. get_order
2. search_policy
3. calculate_refund
4. issue_refund

Each tool was executed once.

### 6. ReAct vs Plan-and-Execute

**ReAct:** The agent reasons, selects a tool, observes its result, and decides the next action. It can respond to unexpected observations, but may require more model calls and tokens.

**Plan-and-Execute:** The agent creates a plan first and then executes its steps. It can be more efficient for predictable tasks, but may need replanning when unexpected information appears.

**Preferred approach:** ReAct is preferred for this refund-support project because order details, policy results, and refund eligibility can affect the next action.

### 7. RAG Evaluation Results

The RAG system was evaluated using the policy evaluation dataset containing 14 questions: 12 answerable questions and 2 unanswerable questions.

| Metric | Overlap | Zero Overlap |
|---|---:|---:|
| Expected documents retrieved | 12/12 | 12/12 |
| Expected facts found | 12/12 | 11/12 |
| Unanswerable fallback answers | 2/2 | Not separately reported |

The zero-overlap experiment showed that chunk boundaries can affect retrieval. The question "How soon is money returned when a payment fails but is deducted?" was identified as a question that may fail without overlapping chunks.

The evaluation demonstrates the benefit of overlapping chunks, although manual testing also revealed some irrelevant retrieval results that could be improved.

## Output Files

Important generated files include:

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/lapse_model.joblib`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `data/instamart.db`

## How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Generate the project data if required:

```bash
python generate_data.py
```

3. Run the Part A, Part B, Part C, and Part D scripts from the `src` folder.

4. Check generated results in the `outputs` folder.

## Conclusion

This project demonstrates how data analytics, machine learning, text classification, RAG, and tool-using AI agents can work together to improve customer intelligence and automate customer support operations.

The refund agent successfully completed all seven evaluation scenarios and passed the token-limit, timeout, and checkpoint-resume demonstrations.

---

**Project:** Instamart Customer Intelligence & Support Copilot  
**Status:** Capstone implementation completed
