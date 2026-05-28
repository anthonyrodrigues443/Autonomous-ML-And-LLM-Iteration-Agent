# Examples

Public-dataset demos that ship with `iterate`. Each is a self-contained `BenchmarkTarget` implementation that anyone can clone and run.

These show how to plug your own ML problem or LLM prompt into the framework. **Copy the closest example and adapt.**

| Example | Target family | Dataset | Demonstrates |
|---|---|---|---|
| `churn_tabular/` | `ModelTarget` | Public Kaggle churn dataset (Telco) | Tabular classification: sklearn/XGBoost/LightGBM iteration against a re-measured baseline |
| `toxicity_jigsaw/` | `PromptTarget` | Jigsaw Toxic Comment Classification (public) | LLM prompt iteration via LLM-as-judge eval |
| `intent_clinc150/` | `PromptTarget` | CLINC150 intent classification (public) | Multi-class prompt iteration; proves the framework generalizes beyond binary |

Each example ships a runnable entry point — `run.py` for `ModelTarget`, a `target.py` + `current_prompt.txt` for `PromptTarget` — example data or a `golden_set.jsonl`, and a per-example README.

---

## Adding your own target

```python
# my_company/spam_appeals/target.py
from iterate.targets import PromptTarget

class SpamAppealsTarget(PromptTarget):
    def get_current_prompt(self) -> str: ...
    def fetch_failures(self, window_days: int) -> list: ...
    def get_golden_set(self) -> list: ...
    def get_recent_sample(self, n: int) -> list: ...
    def score(self, prompt: str, sample) -> Metrics: ...
    def get_guards(self) -> list: ...
```

Then:

```bash
iterate run --target my_company/spam_appeals/target.py
```

The framework handles the rest.

---

**Examples land progressively:**

- Week 2: `churn_tabular/` skeleton + a working baseline notebook
- Week 3: `toxicity_jigsaw/` with current prompt + golden set
- Week 5: `intent_clinc150/` (pluggability proof)
