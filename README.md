# ML Projects Portfolio

A growing collection of machine learning projects built from the ground up — each one implemented, debugged, and understood end-to-end rather than just run once and forgotten.

Every project follows the same principle: build the full pipeline by hand, hit real problems along the way (overfitting, data leakage, broken assumptions), fix them, and document what actually happened. Where the theory isn't solid, I stop and fill the gap — linear algebra, probability, or whatever the model demands — before moving on.

This is a working portfolio, not a tutorial archive. The goal is to demonstrate practical ML engineering skill: clean pipelines, honest evaluation, and a clear record of the reasoning behind every decision.

---

## Philosophy

- **Build it, don't just read it** — every project is implemented and rerun independently, not copied
- **Debugging is the curriculum** — bugs like data leakage or inflated CV scores are treated as the main learning material, not noise to hide
- **Theory on demand** — math and ML theory are studied as needed to understand *why* a model works, not as a prerequisite checklist
- **Honest results** — metrics are reported as-is, including failed iterations and dead ends, because that's what a real ML workflow looks like

---

## Projects

### 01 — [Sonar Rock vs Mine Prediction](./01-sonar-rock-vs-mine-prediction)
Binary classification on the UCI Sonar dataset using SVM. Highlights include building a leak-free `Pipeline` with `StandardScaler`, tuning hyperparameters with `GridSearchCV`, and catching a data leakage bug from fitting on the full dataset instead of the training split alone. Final model: SVM (RBF kernel) — 90.5% test accuracy, 83.1% cross-validated accuracy.

More projects will be added here as they're completed, each with its own README detailing the approach, iterations, and results.

---

## Tech Stack

`Python 3` `scikit-learn` `pandas` `numpy`

Expanding into deep learning frameworks and additional tooling as the portfolio grows.

---

## Author

**Sai Srikar B**
B.Tech AI & Data Science | Rajalakshmi Engineering College
[LinkedIn](https://linkedin.com/in/sai-srikar-b-54b1b8359) · [GitHub](https://github.com/Saisrikar20)
