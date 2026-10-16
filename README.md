# Tabular Analysis Agent

A command-line agent that takes any CSV and answers plain-English questions about it. Pandas code is written by an LLM and executed by the agent's own tools. Answers are only returned when tool output (or a stored memory fact) supports them. The agent also remembers what it learns across sessions.

Built for the BYTE Applied ML recruitment task, Agentic [Basic], with the persistent-memory deep dive.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-..." > .env     # optional: MODEL=<openrouter model id>
uv run python main.py path/to/data.csv
```

Everything the program writes goes to `outputs/` (the memory database `outputs/memory.db` and charts in `outputs/plots/`). Put your datasets in `data/`. Both folders are git-ignored.

The first time a dataset schema is seen, the agent analyses it automatically (about a minute). After that, ask questions:

```
> What is the average target by sex?
> Do bmi, bp and s5 together predict target?
> Healthy patients are those with bmi below 0.03        # teaches a definition
> How many healthy patients are there?                  # applies it
> /memory      /forget <key>      /prep      exit
```

Every answer prints the solution, the tools (and LLM-written code) that produced it, and which memory facts were used. If the data cannot answer the question, the agent says so instead of guessing.

## Project layout

| File | Role |
|---|---|
| `main.py` | CLI loop and commands |
| `graph.py` | Preprocessing graph, query graph, analysis agent, answer rendering |
| `tools.py` | Data cleaning, safe code execution, the six tools |
| `memory.py` | Fact schema and the memory harness over a LangGraph store |
| `llm.py` | OpenRouter model factory |

## What I used

| Part | What |
|---|---|
| LLM | `qwen/qwen3.7-flash` through [OpenRouter](https://openrouter.ai) (about $0.03 in / $0.13 out per 1M tokens). Reasoning is switched off: it cut latency from 12.7 s to 1.1 s per call. Change it with `MODEL` in `.env`. |
| Orchestration | LangGraph (`StateGraph`) for the two pipelines, LangChain `create_agent` for the tool-calling loop |
| Memory store | LangGraph `SqliteStore` (`outputs/memory.db`) |
| Analysis | pandas, numpy, scipy, scikit-learn (RandomForest), matplotlib |

## How it works

Three cooperating units, all in [graph.py](graph.py):

**1. Preprocessing graph** (runs once per new schema, or on `/prep`)
`baseline -> propose -> check -> (propose again | end)`
- `clean` (in `tools.py`): drops empty and duplicate rows, imputes nulls (median for numeric, mode otherwise). The imputations are saved as a fact.
- `baseline`: runs the profile, outlier and correlation tools and stores the results as facts.
- `propose`: the LLM sees the schema and a 30-row sample and proposes up to 5 simple, testable assumptions, for example "price rises with rooms and size together".
- `check`: each assumption is verified on the **full** dataset by the analysis agent using the tools. A second LLM call judges whether the tool output supports it. Only confirmed claims are saved. At most 3 rounds, stopping early when a round confirms nothing new.

**2. Query graph** (every question)
`extract -> from_memory -> (done | analyse -> finalize)`
- `extract`: if the message teaches a definition, correction or preference, it is saved first so it applies to the same question.
- `from_memory`: answers only if the stored facts directly contain the answer.
- `analyse`: otherwise the tool-using agent works it out.
- `finalize`: `solved` can only be true if there is at least one tool step or one real memory fact behind it. Otherwise the user gets "I couldn't determine this from the data" plus the reason.

**3. Analysis agent**
`create_agent` with the six tools below. If it replies without calling a tool, it is re-run once with a stricter prompt. Tool errors come back to the model as `ERROR: ...` text so it can fix its own code. A `wrap_tool_call` middleware catches anything a tool raises.

## Tools

| Tool | What it does |
|---|---|
| `run_pandas(code)` | Runs **LLM-written** pandas/numpy/scipy code on a copy of `df`: any filter, group-by, aggregation or significance test (`stats` = `scipy.stats`) |
| `plot(code)` | Runs LLM-written matplotlib and saves the chart to `outputs/plots/`. The multimodal model then looks at the rendered image and the tool returns its description, so the agent can reason about what the chart actually shows |
| `profile()` | Dtypes, nulls, unique counts, distributions (mean/std/skew/quartiles, top values) |
| `correlations(k)` | Strongest column-pair associations across mixed types: Pearson, correlation ratio, Cramér's V |
| `outliers()` | IQR outliers and bounds for each non-binary numeric column |
| `fit_relation(target, features)` | RandomForest with 3-fold CV: do several columns *together* predict a target? Returns R² or accuracy plus feature importances |

Code execution (`_run` in [tools.py](tools.py)) rejects imports (except the already-provided modules), file and OS access, file-writing calls like `savefig`/`to_csv`, and dunder names; truncates output; and returns `ERROR: <type>: <message>` instead of raising. It is a guard, **not a security sandbox**: run it on your own data only.

## Memory design

Implemented in [memory.py](memory.py). Not a chat log and not a data dump: only structured facts.

- **What is kept:** `relation` (verified assumptions), `stat` (profile, outliers, correlations), `cleaning` (what was imputed), `definition` (e.g. "healthy patients" with a reusable pandas `code` filter), `correction`, `preference` (e.g. rounding).
- **Where:** a LangGraph store. Namespace `("facts", <schema fingerprint>)` per dataset, where the fingerprint is a hash of column names and dtypes, so the same schema with new rows reuses its memory and a different schema starts clean. Preferences live in a global `("prefs",)` namespace.
- **Updates and conflicts:** facts are keyed (`healthy_patients`). Saving an existing key replaces it, keeps the previous versions in `history`, and tells the user what was replaced.
- **Retrieval:** facts are injected into the prompt, and the agent is told to apply a fact's `code` verbatim. With tens of facts, all are included (`Memory.relevant` is the single place to swap in semantic search).
- **User control:** `/memory` lists everything, `/forget <key>` deletes. Teaching is done in plain language.
- **Effect on behaviour (tested):** a taught definition changed the counts returned later, even in a new process. Redefining it changed the answer. A rounding preference changed number formatting.

## My contribution vs libraries

Libraries do the plumbing: LangGraph (graph and store), LangChain (`create_agent`, structured output, `@tool`), OpenRouter (model access), pandas and scikit-learn (computation).

What I designed and wrote:
- The three-unit architecture, in particular the **assumption-then-verify** preprocessing: propose relations from a small sample, test them on all the data with tools, store only the confirmed ones.
- The six tools, including the mixed-type association measure and `fit_relation` for multi-column relationships.
- The safe-execution layer and the error-recovery loop (errors fed back to the model, tool-less replies retried).
- The **evidence gate**: an answer is "solved" only with tool steps or memory behind it, and every answer shows its steps.
- The memory harness: typed facts, schema fingerprints, replace-with-history, per-user preferences, inspection and deletion.
- Letting the model see its own charts (image to description) so plot questions are answered from what was rendered.
- Measuring and switching off the model's reasoning mode (about 10x faster, about 50x fewer output tokens).

## Limitations

- Assumption checks can use significance tests (`scipy.stats` is preloaded and the prompt asks for them), but a test is not enforced for every claim, so on noisy data a claim can be judged differently between runs.
- The judge sometimes lists a loosely related fact under "Memory used".
- Quality depends on the model; only `qwen/qwen3.7-flash` has been tested.
- Whole facts are put in the prompt, so very large amounts of memory would need the semantic-search swap.
- Tested on California housing (20,640 rows), IBM Telco churn (7,043 rows) and scikit-learn's diabetes data. Datasets are not committed: California housing comes from `sklearn.datasets.fetch_california_housing` and the churn file is IBM's `Telco-Customer-Churn.csv` (github.com/IBM/telco-customer-churn-on-icp4d); put them in `data/` to reproduce.
