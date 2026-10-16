import re
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from memory import Fact, Memory
from tools import make_tools, schema_text

SYSTEM = """You analyse one pandas DataFrame `df` (already cleaned). Schema:
{schema}

Rules:
- Compute everything with tools; write the pandas/plot code yourself via run_pandas / plot (df, pd, np, plt and stats=scipy.stats are predefined, no imports). Back comparisons with a significance test (stats) or fit_relation, not just a difference in means.
- If a tool returns ERROR, fix the code and retry.
- "Known facts" are the user's definitions and verified findings; apply a fact's `code` verbatim for its term.
- Never state a number you did not see in a tool result. If the data cannot answer, say so."""


class Assumption(BaseModel):
    claim: str = Field(description="one simple testable claim, e.g. 'price rises with rooms and sqft together'")
    check: str = Field(description="what to compute on the full data to test it")


class Proposal(BaseModel):
    assumptions: list[Assumption]


class Verdict(BaseModel):
    holds: bool = Field(description="true only if the tool outputs support the claim")
    evidence: str = Field(description="one or two sentences quoting the numbers from the tool outputs")


class Extraction(BaseModel):
    is_question: bool
    fact: Fact | None = None


class Answer(BaseModel):
    solved: bool
    answer: str = Field(description="one or two complete sentences stating the answer with its numbers and units, or the reason it could not be solved")
    keys: list[str] = Field(default=[], description="memory fact keys the answer relies on")


class PrepState(TypedDict):
    pending: list[Assumption]
    tried: list[str]
    confirmed: int
    new: int  # confirmed in the latest round
    round: int


class QueryState(TypedDict):
    question: str
    facts: str
    notice: str
    is_question: bool
    answer: str
    solved: bool
    steps: list[dict]  # [{"tool", "args", "result"}] in call order
    sources: list[str]  # memory fact keys used


@wrap_tool_call
def catch(request, handler):
    """Safety net: a raising tool becomes an ERROR message the agent can react to."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=f"ERROR: {type(e).__name__}: {e}", tool_call_id=request.tool_call["id"])


def _clip(x, n: int) -> str:
    return " ".join(str(x).split())[:n]


def fmt_steps(steps: list[dict], n: int = 200) -> str:
    return "\n".join(
        f"  {i}. {t['tool']}  {_clip(t['args'].get('code') or t['args'], n)}  -> {_clip(t['result'], n)}"
        + ("  (error)" if t["result"].startswith("ERROR") else "")
        for i, t in enumerate(steps, 1)
    )


def render(s: dict) -> str:
    """Solution plus the tools (and generated code) that produced it."""
    out = [f"[memory] {s['notice']}"] if s.get("notice") else []
    if s.get("is_question", True):
        out.append(("Answer: " if s["solved"] else "I couldn't determine this from the data. ") + s["answer"])
        if s.get("steps"):
            out.append("How I got it:\n" + fmt_steps(s["steps"]))
        if s.get("sources"):
            out.append("Memory used: " + ", ".join(s["sources"]))
    return "\n".join(out)


def build(df, memory: Memory, llm, cleaning_notes: list[str]):
    """Returns (preprocessing graph, query graph) bound to one cleaned DataFrame."""
    tools = make_tools(df, llm)
    by_name = {t.name: t for t in tools}
    schema = schema_text(df)
    agent = create_agent(llm, tools, system_prompt=SYSTEM.format(schema=schema), middleware=[catch])

    def judge(model, prompt: str):
        return llm.with_structured_output(model).invoke(prompt)

    def known(keys: list[str]) -> list[str]:
        have = {f.key for f in memory.all()}
        return [k for k in keys if k in have]

    def analyse_task(task: str) -> tuple[str, list[dict]]:
        """Run the tool-using agent; return its final text and the ordered tool steps."""
        for nudge in ("", "\n\nYou MUST call a tool to compute this; do not just describe or write code in text."):
            try:
                msgs = agent.invoke({"messages": [("user", task + nudge)]}, {"recursion_limit": 25})["messages"]
            except GraphRecursionError:
                return "stopped at the step limit", []
            results = {m.tool_call_id: str(m.content) for m in msgs if m.type == "tool"}
            steps = [{"tool": c["name"], "args": c["args"], "result": results.get(c["id"], "")}
                     for m in msgs if m.type == "ai" for c in m.tool_calls]
            if steps:
                break
        return str(msgs[-1].content), steps

    # ---- Unit 1: preprocessing ------------------------------------------------
    def baseline(_):
        memory.save(Fact(key="cleaning", kind="cleaning", text="; ".join(cleaning_notes) or "no nulls or duplicates"))
        for name in ("profile", "outliers", "correlations"):
            memory.save(Fact(key=name, kind="stat", text=by_name[name].invoke({})[:2000]))
        return {}

    def propose(s):
        prompt = (
            f"Schema:\n{schema}\n\nSample rows:\n{df.sample(min(30, len(df)), random_state=s['round']).to_string()}\n\n"
            f"Known facts:\n{memory.relevant('')}\n\nAlready tested, do not repeat or reword any of these: {s['tried']}\n\n"
            "Propose up to 5 SIMPLE, directly testable assumptions that establish the basic facts of this data: "
            "which columns drive an obvious target (alone or several together, e.g. rooms and size -> price), or "
            "how a group (demographic) differs in a categorical choice or a mean. Avoid causal or latent-factor "
            "claims. Each needs a concrete check using the available tools.")
        fresh = [a for a in judge(Proposal, prompt).assumptions if a.claim not in s["tried"]]
        return {"pending": fresh[:5], "round": s["round"] + 1}

    def verify(a: Assumption):
        """Test one assumption on the full data. Returns (confirmed, why, steps)."""
        try:
            found, steps = analyse_task(f"Verify on the full dataset using tools: {a.claim}. Suggested check: {a.check}. Report the numbers.")
            if not steps:
                return False, found or "agent produced no tool evidence", steps
            v = judge(Verdict, f"Claim: {a.claim}\nTool steps:\n{fmt_steps(steps, 800)}\nAgent findings: {found}\n"
                               "Does the claim hold? Answer true only if the tool outputs support it.")
            return v.holds, v.evidence, steps
        except Exception as e:  # one bad assumption must not abort preprocessing
            return False, f"{type(e).__name__}: {e}", []

    def check(s):
        with ThreadPoolExecutor(5) as pool:  # assumptions are independent
            results = list(pool.map(verify, s["pending"]))
        seen, new = {f.key for f in memory.all()}, 0
        for a, (ok, why, steps) in zip(s["pending"], results):
            if ok:
                key = "rel_" + re.sub(r"\W+", "_", a.claim.lower())[:40]
                memory.save(Fact(key=key, kind="relation", text=a.claim, evidence=(why + "\n" + fmt_steps(steps))[:800]))
                new += key not in seen  # a reworded repeat of a known fact is not progress
            print(f"  {'confirmed' if ok else 'rejected '}: {a.claim}\n      {_clip(why, 160)}")
        return {"tried": s["tried"] + [a.claim for a in s["pending"]], "confirmed": s["confirmed"] + new, "new": new}

    prep = StateGraph(PrepState)
    prep.add_node("baseline", baseline)
    prep.add_node("propose", propose)
    prep.add_node("check", check)
    prep.add_edge(START, "baseline")
    prep.add_edge("baseline", "propose")
    prep.add_edge("propose", "check")
    prep.add_conditional_edges(
        "check", lambda s: END if s["round"] >= 3 or s["new"] == 0 or s["confirmed"] >= 15 else "propose")

    # ---- Unit 2: query ---------------------------------------------------------
    def extract(s):
        e = judge(Extraction, (
            f"User message: {s['question']}\n"
            "If it states a durable definition, correction or preference about the data (e.g. 'active customers "
            "exclude cancelled accounts'), return it as `fact` (key: snake_case term; code: a pandas query filter "
            "if it defines a filter). Otherwise fact is null. is_question is true for any question or request to "
            "compute, plot or analyse something (even when phrased as a command), and false only for a pure statement "
            "that teaches a fact or preference."))
        return {"notice": memory.save(e.fact) if e.fact else "", "is_question": e.is_question,
                "facts": memory.relevant(s["question"])}

    def from_memory(s):
        a = judge(Answer, f"Known facts:\n{s['facts']}\n\nQuestion: {s['question']}\n"
                          "solved=true ONLY if a single fact literally states the answer; never combine or infer across facts. If the question asks for specific values (rates, counts, averages) the facts do not state, solved=false. Requests to plot, draw or visualise, "
                          "and questions needing a new computation, are NOT answerable from facts (solved=false). "
                          "answer = complete sentences quoting the specific numbers and columns from the facts. "
                          "keys = only the fact keys the answer actually relies on.")
        keys = known(a.keys)
        return {"solved": a.solved and bool(keys), "answer": a.answer, "sources": keys, "steps": []}

    def analyse(s):
        answer, steps = analyse_task(f"Known facts:\n{s['facts']}\n\nQuestion: {s['question']}")
        return {"answer": answer, "steps": steps}

    def finalize(s):
        a = judge(Answer, f"Question: {s['question']}\nAgent answer: {s['answer']}\nTool steps:\n{fmt_steps(s['steps'], 800)}\n"
                          f"Known facts:\n{s['facts']}\n"
                          "solved=true only if the question is clearly answered with numbers visible in the tool steps. "
                          "answer = the final answer, or a short reason if unsolved. keys = only the fact keys whose "
                          "definition or code the answer applied (empty list if none).")
        return {"solved": a.solved and bool(s["steps"]), "answer": a.answer, "sources": known(a.keys)}

    query = StateGraph(QueryState)
    for name, fn in (("extract", extract), ("from_memory", from_memory), ("analyse", analyse), ("finalize", finalize)):
        query.add_node(name, fn)
    query.add_edge(START, "extract")
    query.add_conditional_edges("extract", lambda s: "from_memory" if s["is_question"] else END)
    query.add_conditional_edges("from_memory", lambda s: END if s["solved"] else "analyse")
    query.add_edge("analyse", "finalize")
    query.add_edge("finalize", END)

    return prep.compile(), query.compile()
