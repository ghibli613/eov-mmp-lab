# Studies

The pre-registration and one folder per batch. They were moved here from the repo root
by **A25b** (2026-10-10). **Paths cited before A25b omit the `studies/` prefix:**
`prereg/…` now means `studies/prereg/…`, and likewise for `rb1/…` and `d2/…`.

| folder | what | amendments | report | status |
|---|---|---|---|---|
| `prereg/` | the pre-registration (continues the EOV pilot file, frozen at A22) | A23 onward | — | append-only, 444 |
| `rb1/` | RB1: a clean EOV-MMP baseline, and what METOR's iterative-enhancement module adds | A23, A23a | `rb1/step1_report.md`, `rb1/fallback_check_report.md` | Step 1 done; stopped before Step 2 (GPU) |
| `d2/` | D2: at which front-end stage are D1's U1 object tracks lost? (test-label analysis, disclosed) | A25, A25a, A25b, A25c, A25d | `d2/step1_report.md`, `d2/d2_report.md` | closed (A25d) |
| `design_review/` | design reviews 1–3, the VidOR cost check, the Q7 headroom count, and the agreed plan | A25e | `design_review/agreed_plan.md` | agreed |
| `a26/` | A26: the protocol audit (R0u, R0 over the whole video, both), released checkpoint only | A26 | `a26/step0_report.md` | step 0 done; stopped before any GPU use |

Elsewhere in the chain:
- the EOV pilot (A1–A22, including D1 at A20) is in `~/Documents/thesis/source/ov-vidvrd-lab/pilot_analysis/`;
- M1 (A24–A24i, closed) is in `~/Documents/thesis/source/ov-pairtime/`.

The `d2` and `a26` commands run from this folder (`cd studies; python -m d2.…`, `python -m a26.…`). METOR's own tests
run from the repo root.
