import hashlib
from typing import Literal

import pandas as pd
from langgraph.store.base import BaseStore
from pydantic import BaseModel

PREFS = ("prefs",)


class Fact(BaseModel):
    key: str  # snake_case id, e.g. "active_customers"
    kind: Literal["relation", "stat", "cleaning", "definition", "correction", "preference"]
    text: str  # human-readable claim
    code: str = ""  # optional reusable pandas filter/expression
    evidence: str = ""  # tool output that confirmed it


def fingerprint(df: pd.DataFrame) -> str:
    """Same columns and dtypes -> same memory, whatever the rows."""
    return hashlib.md5(str(sorted((c, str(t)) for c, t in df.dtypes.items())).encode()).hexdigest()[:12]


class Memory:
    """Facts live in a LangGraph store: one namespace per dataset schema, one global one for preferences."""

    def __init__(self, store: BaseStore, schema_id: str):
        self.store = store
        self.facts_ns = ("facts", schema_id)

    def _ns(self, kind: str) -> tuple:
        return PREFS if kind == "preference" else self.facts_ns

    def save(self, fact: Fact) -> str:
        ns, old = self._ns(fact.kind), self.store.get(self._ns(fact.kind), fact.key)
        value = fact.model_dump()
        if old:  # same key = update; keep the superseded versions
            value["history"] = (old.value.get("history", []) + [{"text": old.value["text"], "code": old.value["code"]}])[-5:]
        self.store.put(ns, fact.key, value)
        if old and (old.value["text"], old.value["code"]) != (fact.text, fact.code):
            return f"replaced '{fact.key}': '{old.value['text']}' -> '{fact.text}'"
        return f"saved '{fact.key}': {fact.text}"

    def all(self) -> list[Fact]:
        items = self.store.search(self.facts_ns, limit=1000) + self.store.search(PREFS, limit=1000)
        return [Fact(**{k: v for k, v in i.value.items() if k in Fact.model_fields}) for i in items]

    def forget(self, key: str) -> None:
        for ns in (self.facts_ns, PREFS):
            self.store.delete(ns, key)

    def relevant(self, question: str) -> str:
        """Facts to put in a prompt. v1 returns all of them (tens at most); swap in semantic search here."""
        return "\n".join(f"- {f.key} [{f.kind}]: {f.text}" + (f" | code: {f.code}" if f.code else "") for f in self.all())
