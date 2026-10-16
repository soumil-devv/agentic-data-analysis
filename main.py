"""Usage: python main.py data.csv   (commands: /memory, /forget <key>, /prep, exit)"""
import sys

import pandas as pd
from langgraph.store.sqlite import SqliteStore

from graph import build, render
from llm import get_llm
from memory import Memory, fingerprint
from tools import OUTPUTS, clean

PREP_START = {"pending": [], "tried": [], "confirmed": 0, "new": 0, "round": 0}


def main(path: str) -> None:
    raw = pd.read_csv(path)
    df, notes = clean(raw)
    OUTPUTS.mkdir(exist_ok=True)
    with SqliteStore.from_conn_string(str(OUTPUTS / "memory.db")) as store:
        store.setup()
        memory = Memory(store, fingerprint(raw))
        prep, query = build(df, memory, get_llm(), notes)
        print(f"Loaded {path}: {len(df)} rows, {len(df.columns)} columns.")
        if not any(f.key == "profile" for f in memory.all()):
            print("New dataset schema: analysing it first...")
            prep.invoke(PREP_START)
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if q in ("exit", "quit"):
                break
            try:
                if q == "/memory":
                    print(memory.relevant(""))
                elif q.startswith("/forget "):
                    memory.forget(q.split(maxsplit=1)[1])
                elif q == "/prep":
                    prep.invoke(PREP_START)
                elif q:
                    print(render(query.invoke({"question": q})))
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1])
