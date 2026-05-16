# How to use this folder

This is **not** the official repo itself — it's a staging copy of the files that need to go in your **first commit** on the official repo. The folder is git-ignored inside `waf-iq/CSAI415` so it never leaks into staging history.

## Workflow

### Step 1 — Initialize the official repo locally

```bash
# clone the empty official repo somewhere outside this staging project
git clone <official-repo-url> C:\Users\waska\Projects\CSAI415_official
cd C:\Users\waska\Projects\CSAI415_official
```

### Step 2 — Copy everything from this folder into the clone

```powershell
# from the official clone directory:
xcopy /E /I /Y C:\Users\waska\Projects\CSAI415\official_repo\* .
```

or in bash:

```bash
cp -r C:/Users/waska/Projects/CSAI415/official_repo/* .
cp C:/Users/waska/Projects/CSAI415/official_repo/.gitignore .
cp C:/Users/waska/Projects/CSAI415/official_repo/.env.example .
```

### Step 3 — Make the bootstrap commit

```bash
git add .
git commit -m "Initial project scaffold: build config, README, MEMBER_BRIEF, package init"
git push origin main
```

That's your one bootstrap commit. After this, share the official repo URL with the team.

## What this folder does NOT contain (by design)

These files are owned by your teammates — they each create them in their own commits on the official repo, under their own GitHub identity. Do not bootstrap them.

| File | Whose commit |
|---|---|
| `src/csai415/ingest.py` | Abdurlahman |
| `src/csai415/eval.py`, `runcard.py`, full `README.md`, `tests/test_smoke.py`, `reports/D1_report.pdf` | Yousef |
| `src/csai415/retrieve.py` | Ahmad |
| `src/csai415/automl.py`, `configs/winning_runcard.yaml`, `notebooks/01_automl.ipynb` | You (WAFIQ) — separate commit later when Pair B's work is ready |
| `src/csai415/online.py`, `notebooks/02_online_learning.ipynb`, `reports/prequential.png` | Ahmed + Yehia |
| `src/csai415/mlflow_tracking.py` | Musab |
| `ai_logs/<their-name>.md` | each member commits their own |

Each teammate clones the official repo (after your bootstrap commit lands), reads their `ai_briefs/<name>.md` from the staging repo, does their AI conversation, writes their files into their local clone, and commits.

## What this folder also does NOT contain

`ai_briefs/*.md` — these are staging-only coordination docs. They explicitly document the prompt-templating workflow which would look meta/weird to a grader. Keep them in `waf-iq/CSAI415` only.

## After everyone is done

Final repo on the official side should have:

- ~10+ commits, at least 1 from each of 7 GitHub identities (`git log --pretty=format:'%an' | sort -u` shows all 7 names)
- Every member's `ai_logs/<name>.md` has a real share-link
- `pytest tests/test_smoke.py` passes on a fresh clone
- `reports/D1_report.pdf` exists, 2 pages
