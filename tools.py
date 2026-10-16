import ast
import base64
import functools
import io
import itertools
import re
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

PROVIDED = re.compile(  # imports of modules that are already in the namespace are harmless: drop them
    r"^\s*(import (pandas as pd|numpy as np|matplotlib\.pyplot as plt|scipy\.stats as stats)|from scipy import stats"
    r"|from matplotlib import pyplot as plt)\s*$", re.M)
BLOCKED = re.compile(r"\bimport\b|__|\bopen\(|\beval\(|\bexec\(|\bos\.|\bsys\.|subprocess"
                     r"|\.(savefig|to_csv|to_pickle|to_excel|to_parquet|to_sql|to_hdf|to_feather)\(")
MAX_CHARS = 3000
OUTPUTS = Path("outputs")  # everything the program writes (memory database, plots) goes here
PLOT_LOCK = threading.Lock()  # pyplot is global state
MAX_CATEGORIES = 50  # skip ID-like categorical columns in correlations


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop empty/duplicate rows and empty columns, impute nulls (median / mode). Returns (df, notes)."""
    n = len(df)
    df = df.dropna(how="all").dropna(axis=1, how="all").drop_duplicates().copy()
    notes = [f"dropped {n - len(df)} empty/duplicate rows"] if len(df) < n else []
    for c in df.select_dtypes(exclude=["number", "bool"]).columns:  # numbers stored as text, blanks included
        blank = df[c].astype(str).str.strip() == ""
        num = pd.to_numeric(df[c].where(~blank), errors="coerce")
        if num.notna().any() and num.notna().sum() >= 0.9 * (~blank).sum():
            df[c] = num
            notes.append(f"{c}: text -> numeric ({int(blank.sum())} blank values -> null)")
    for c in df.columns[df.isna().any()]:
        how, fill = ("median", df[c].median()) if _is_num(df[c]) else ("mode", df[c].mode().iloc[0])
        notes.append(f"{c}: {df[c].isna().sum()} nulls -> {how} {fill}")
        df[c] = df[c].fillna(fill)
    return df, notes


def schema_text(df: pd.DataFrame) -> str:
    cols = (f"- {c} ({df[c].dtype}, {df[c].nunique()} unique), e.g. {df[c].unique()[:3].tolist()}" for c in df.columns)
    return "\n".join(cols) + f"\n{len(df)} rows"


def _run(code: str, ns: dict) -> str:
    """Execute LLM-written code. Returns printed output + the last expression's value, or 'ERROR: ...'."""
    code = PROVIDED.sub("", code)
    if BLOCKED.search(code):
        return "ERROR: other imports, file/os access, file-writing calls and dunder names are not allowed; df, pd, np, plt, stats (scipy.stats) are predefined."
    out = io.StringIO()
    ns["print"] = functools.partial(print, file=out)  # per-call capture: redirect_stdout is global, not thread-safe
    try:
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, "<llm>", "exec"), ns)
        value = eval(compile(ast.Expression(last.value), "<llm>", "eval"), ns) if last else None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    if value is not None:
        out.write(value.to_string(max_rows=40) if hasattr(value, "to_string") else str(value))
    return out.getvalue()[:MAX_CHARS] or "(no output)"


def _assoc(a: pd.Series, b: pd.Series) -> float:
    """Association in [0, 1]: |Pearson| (num-num), correlation ratio (num-cat), Cramer's V (cat-cat)."""
    na, nb = _is_num(a), _is_num(b)
    if na and nb:
        return abs(a.corr(b))
    if na or nb:
        num, cat = (a, b) if na else (b, a)
        g, total = num.groupby(cat), ((num - num.mean()) ** 2).sum()
        return float(np.sqrt((g.count() * (g.mean() - num.mean()) ** 2).sum() / total)) if total else 0.0
    o = pd.crosstab(a, b).values
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return float(np.sqrt(((o - e) ** 2 / e).sum() / (o.sum() * (min(o.shape) - 1)))) if min(o.shape) > 1 else 0.0


def _describe(llm, path: str) -> str:
    """Let the (multimodal) LLM look at a saved chart. A failure only means no description."""
    try:
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        ask = "Describe this chart in 2-3 sentences: what is plotted, the main trend or pattern, notable features, approximate values."
        msg = HumanMessage([{"type": "text", "text": ask}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}])
        return str(llm.invoke([msg]).content)
    except Exception:
        return ""


def make_tools(df: pd.DataFrame, llm=None) -> list:
    """Six tools bound to one (already cleaned) DataFrame. Tools return 'ERROR: ...' strings instead of raising.
    If a multimodal `llm` is given, `plot` also returns the model's description of the rendered chart."""

    def namespace() -> dict:
        return {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "stats": stats}

    def unknown(cols: list[str]) -> str:
        bad = [c for c in cols if c not in df.columns]
        return f"ERROR: unknown columns {bad}; valid columns: {list(df.columns)}" if bad else ""

    @tool
    def run_pandas(code: str) -> str:
        """Run pandas/numpy/scipy code on the DataFrame `df` (pd, np and stats=scipy.stats are predefined, no imports, df is a copy).
        Use stats for significance tests, e.g. stats.ttest_ind, stats.chi2_contingency, stats.pearsonr.
        Use print() or end with an expression to return a value, e.g. df.groupby('city')['price'].mean()."""
        return _run(code, namespace())

    @tool
    def plot(code: str) -> str:
        """Run matplotlib code (df, pd, np, plt predefined, no imports, do not call savefig/show) to draw a chart.
        Returns the saved image path and a description of what the rendered chart shows."""
        with PLOT_LOCK:
            res, path = _run(code, namespace()), ""
            if not res.startswith("ERROR"):
                (OUTPUTS / "plots").mkdir(parents=True, exist_ok=True)
                path = str(OUTPUTS / "plots" / f"{len(list((OUTPUTS / 'plots').glob('*.png'))) + 1}.png")
                plt.savefig(path, bbox_inches="tight")
                res = f"saved {path}"
            plt.close("all")
        seen = _describe(llm, path) if path and llm else ""
        return f"{res}. The chart shows: {seen}" if seen else res

    @tool
    def profile() -> str:
        """Per-column dtype, null count, unique count and distribution (numeric: mean/std/skew/quartiles; otherwise top values)."""
        rows = []
        for c in df.columns:
            s = df[c]
            r = {"column": c, "dtype": str(s.dtype), "nulls": int(s.isna().sum()), "unique": int(s.nunique())}
            if _is_num(s):
                r |= {"mean": s.mean(), "std": s.std(), "skew": s.skew(), "min": s.min(), "q25": s.quantile(0.25),
                      "median": s.median(), "q75": s.quantile(0.75), "max": s.max()}
            else:
                r["top"] = s.value_counts().head(3).to_dict()
            rows.append(r)
        return pd.DataFrame(rows).round(3).to_string(index=False, na_rep="")

    @tool
    def correlations(k: int = 10) -> str:
        """Strongest associations between column pairs (0 to 1), numeric and categorical mixed:
        Pearson, correlation ratio, Cramer's V. Returns the top k pairs."""
        cols = [c for c in df.columns if _is_num(df[c]) or df[c].nunique() <= MAX_CATEGORIES][:40]
        pairs = []
        for a, b in itertools.combinations(cols, 2):
            d = df[[a, b]].dropna()
            if len(d) > 2:
                pairs.append((a, b, float(np.nan_to_num(_assoc(d[a], d[b])))))
        pairs.sort(key=lambda p: -p[2])
        return "\n".join(f"{a} ~ {b}: {v:.3f}" for a, b, v in pairs[:k]) or "no column pairs to compare"

    @tool
    def outliers() -> str:
        """IQR (1.5x) outlier count and bounds for every non-binary numeric column that has outliers."""
        lines = []
        for c in (c for c in df.columns if _is_num(df[c]) and df[c].nunique() > 2):  # IQR is meaningless on flags
            q1, q3 = df[c].quantile([0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            n = int(((df[c] < lo) | (df[c] > hi)).sum())
            if n:
                lines.append(f"{c}: {n} outliers ({n / len(df):.1%}), bounds [{lo:.4g}, {hi:.4g}]")
        return "\n".join(lines) or "no IQR outliers in any numeric column"

    @tool
    def fit_relation(target: str, features: list[str]) -> str:
        """Test whether several columns TOGETHER predict `target` (RandomForest, 3-fold CV).
        Returns R2 for a numeric target (accuracy otherwise) and feature importances,
        e.g. do rooms + size predict price, or do age + income predict brand."""
        if err := unknown([target, *features]):
            return err
        feats = [f for f in features if f != target] or [c for c in df.columns if c != target]
        d = df.sample(min(len(df), 20000), random_state=0)
        X = d[feats].apply(lambda s: s if _is_num(s) else s.astype("category").cat.codes)
        y = d[target]
        reg = _is_num(y) and y.nunique() > 10
        model = (RandomForestRegressor if reg else RandomForestClassifier)(n_estimators=100, random_state=0, n_jobs=-1)
        score = cross_val_score(model, X, y, cv=3).mean()
        imp = pd.Series(model.fit(X, y).feature_importances_, index=feats).sort_values(ascending=False)
        return f"{'R2' if reg else 'accuracy'} (3-fold CV) = {score:.3f}; importances: " + ", ".join(
            f"{k}={v:.2f}" for k, v in imp.head(8).items())

    return [run_pandas, plot, profile, correlations, outliers, fit_relation]
