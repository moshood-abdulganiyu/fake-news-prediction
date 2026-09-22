# Project: Fake News Prediction (Rebuild, Streamlit)

Rebuilding an old Fake News Prediction notebook (TF-IDF + classic ML
classifiers) into a clean, light, end-to-end portfolio project, delivered
as a single Streamlit app.

## Repo structure (locked in, do not restructure)
data/
notebooks/
src/ (preprocess.py, model.py — plain Python, imported directly by the app,
      no API layer in between)
app/ (streamlit_app.py)
screenshots/
No server/ or client/ folders — this project does not use the FastAPI +
separate frontend pattern used in this author's other projects.

## Dataset
Fake.csv and True.csv (ISOT-style), combined into one dataframe with a
label column (1 = fake, 0 = real). Columns: Title, Text, Subject, Date.
No author column — this dataset is structurally different from the
original Kaggle train.csv the old notebook used.

## How to work
- Keep it light. No unnecessary abstractions or extra features.
- src/ holds cleaning, vectorizing, and inference logic. app/streamlit_app.py
  imports and calls it directly — don't duplicate logic inside the app file.
- Deployment target is Streamlit Community Cloud or a Streamlit-SDK
  Hugging Face Space — don't introduce Docker, a separate backend service,
  or a second deployment platform.

## CRITICAL — data leakage fix
The original notebook hit 99.4% test accuracy on a Decision Tree using only
author + title as features. This dataset (Fake.csv/True.csv) has no author
column, so the leakage risk here is different: `Subject` separates the two
classes almost perfectly (verified via crosstab in notebooks/01_eda.ipynb),
and True.csv text frequently starts with a Reuters dateline pattern like
"WASHINGTON (Reuters) -". When implementing the modeling code:
- Use cross-validation, not a single train/test split, for reported metrics.
- Do not use `Subject` as a model feature. Note its exclusion explicitly
  in code comments and the README, don't just omit it silently.
- Strip the Reuters/AP-style dateline prefix from `text` before cleaning.
- If asked to just "make accuracy higher," push back and ask whether that's
  actually the goal, given the leakage risks above.

## Style
- No em dashes.
- No AI-cliché phrasing in comments, docstrings, or commit messages.
- Concrete numbers over adjectives in any generated text (docs, comments,
  commit messages).
- Python: type hints on function signatures, docstrings on public functions.

## What NOT to do
- Don't silently change the repo structure or reintroduce a FastAPI backend.
- Don't add tooling beyond what a light Streamlit app needs.
- Don't fabricate metrics or claims not backed by actual code output.

## Context
The deeper reasoning (feature decisions, model comparison rationale,
README structure) happens in a separate Claude Project — this file exists
so code generated here doesn't drift from what's decided there. If a task
here seems to require a design decision (not just implementation), flag it
instead of guessing.