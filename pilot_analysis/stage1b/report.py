#!/usr/bin/env python3
"""Stage 1 session 2 -- write slot_check.md, preregistration.md (+ sha256) and README.md
from slot_check_summary.json. No computation beyond formatting and hashing.

    python3 pilot_analysis/stage1b/report.py
"""
from __future__ import annotations

import hashlib, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATE = "2026-09-28"
S1 = json.load(open(os.path.join(ROOT, "pilot_analysis", "stage1", "axis_check_summary.json")))
S = json.load(open(os.path.join(HERE, "slot_check_summary.json")))
PARA = json.load(open(os.path.join(HERE, "paraphrases_frozen.json")))
PASS_AGREE, FIX_AUC, WEAK_AUC = 0.70, 0.80, 0.70


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pct(t):
    h, n = t
    return f"{h}/{n} ({h / n:.1%})" if n else "—"


def ci(t):
    return "—" if not t or t[0] != t[0] else f"{t[0]:.3f} ({t[1]:.3f}–{t[2]:.3f})"


sym = S["symmetry"]
tests = sorted({k[:-2] for k in sym if k.endswith("_n")})
sym_exact = all(sym[f"{t}_ok"] == sym[f"{t}_n"] for t in tests)


def verdict(agr, auc):
    if agr == agr and agr >= PASS_AGREE:
        return "PASS" if sym_exact else "FAIL (symmetry)"
    if auc is None:
        return "UNRESOLVED (<70%, no AUC)"
    if auc >= FIX_AUC:
        return "FIXABLE"
    if auc >= WEAK_AUC:
        return "WEAK"
    return "FAIL"


# ---- per-axis rows: session 1 vs slot level ----
S1V = {v["axis"]: v for v in S1["verdicts"]}
S1KEY = {"horizontal": "horizontal", "vertical": "vertical", "depth": "depth",
         "proximity": "proximity", "subject_state": "subject_state", "object_state": "object_state",
         "comparative=larger": "comparative=larger", "comparative=taller": "comparative=taller",
         "comparative=faster": "comparative=faster"}
rows = []
for axis in S1KEY:
    h = S["held"][axis]
    vals = {k: v for k, v in h["agree"].items() if k != "non-adjacent negatives"}
    agr = min(a / n for a, n in vals.values() if n)
    auc = h["auc"][0] if h["auc"] else None
    s1 = S1V[S1KEY[axis]]
    rows.append(dict(axis=axis, s1_agr=s1["agreement"], s1_auc=s1["auc"], s1_v=s1["verdict"],
                     w=h["window"], agr=agr, vals=vals, auc=h["auc"], v=verdict(agr, auc)))

TH = S["thresholds"]
ch = S["chosen"]
rm = S["rm"]
dec = S["decision"]
t3 = S["task3"]
frac = lambda t: t[0] / t[1] if t[1] else float("nan")

# ======================= slot_check.md =======================
L = ["# Stage 1 session 2 — slot-level axis check, relative-motion choice\n",
     "BASE labels from the TRAINING annotations only: novel instances are dropped when "
     "each file is parsed. Test annotations were not opened. Split: session 1's "
     "`train_split.json` (640 FIT / 160 HELD videos). Code: `slot_axes.py`, "
     "`slot_check.py`; raw numbers are in `slot_check_summary.json`.\n",
     "## Grid (interface_audit.md B2, on whole tracks from `anno/train`)\n",
     f"{S['grid']['pairs']} ordered pairs, {S['grid']['slots']} slots, "
     f"**{S['grid']['labelled_slots']} with at least one base label** (≥ 15 frames of a "
     f"base predicate's extent inside the slot, same ordered pair): "
     f"{S['n']['fit']} FIT, {S['n']['held']} HELD. {S['no_ecc_videos']} videos without "
     f"ECC are excluded. Slots whose labels assert conflicting values on an axis are "
     f"left out for that axis: " + ", ".join(f"{a} {n}" for a, n in S["conflicts"].items()) + ".\n",
     "## Task 1 — FIT balanced agreement per window, chosen window, HELD\n",
     "W0 = the slot; W1 = ± 1 slot; W2 = ± 2 slots; W3 = the pair's whole overlap, all "
     "clipped to the overlap. Thresholds are refit on FIT at each window, using session "
     "1's rules. Each axis takes the window with the best FIT balanced agreement "
     "(ties go to the smaller window). HELD is scored once, at that window.\n",
     "| axis | W0 | W1 | W2 | W3 | chosen | HELD agreement | HELD AUC (95% CI) |",
     "|---|---:|---:|---:|---:|---|---|---:|"]
for r in rows:
    c = S["curve"][r["axis"]]
    L.append(f"| {r['axis']} | " + " | ".join(
        (f"**{c[w]:.3f}**" if w == r["w"] else f"{c[w]:.3f}") for w in ("W0", "W1", "W2", "W3"))
        + f" | {r['w']} | " + "; ".join(f"{k} {pct(v)}" for k, v in r["vals"].items())
        + f" | {ci(r['auc'])} |")
prox = S["held"]["proximity"]["agree"]["non-adjacent negatives"]
L.append(f"\nProximity negatives measured as non-adjacent on HELD: {pct(prox)}.\n")
L.append("**Fitted thresholds per window (FIT):**\n")
L.append("| window | motion (box-diag/frame) | adjacency gap | D1 dead zone | D2 margin c |")
L.append("|---|---:|---:|---:|---:|")
for w in ("W0", "W1", "W2", "W3"):
    t = TH[w]
    L.append(f"| {w} | {t['motion']:.5f} | {t['gap']:.4f} | {t['dead_zone']:.4f} | {t['c']:.3f} |")
L.append("\n**subject_state by verb family, HELD, at " + ch["subject_state"] + ":** " + ", ".join(
    f"`{k}` {pct(v)}" for k, v in sorted(S["held"]["subject_state"]["by_verb"].items(),
                                      key=lambda kv: kv[1][0] / kv[1][1])) + "\n")
L.append("**Symmetry, at every window (fitted thresholds):**\n")
L.append("| window | D1 time reversal | D2 time reversal | swap (3 spatial axes) | swap (comparatives) | D1 = session-1 function |")
L.append("|---|---:|---:|---:|---:|---:|")
for w in ("W0", "W1", "W2", "W3"):
    g = lambda t: f"{sym[f'{w}_{t}_ok']}/{sym[f'{w}_{t}_n']}"
    L.append(f"| {w} | {g('timeD1')} | {g('timeD2')} | {g('swap')} | {g('comp')} | {g('d1consistency')} |")
L.append(f"\n**{'All exact.' if sym_exact else '⚠ NOT EXACT — code bug.'}**\n")

L.append("## Task 2 — relative_motion: D1 distance vs D2 heading\n")
L.append("Each definition takes its window by FIT: the mean of three agreements, namely "
         "chase/follow (D1: approach or co_move; D2: toward), co_move on `*_with`, and no "
         "subject-driven motion on stand/sit/lie subjects. The LOW-N sets (pass, approach) "
         "were not used for fitting. D2's margin c is fitted on the same objective.\n")
L.append("| HELD | D1 distance (" + rm["D1"]["HELD"]["window"] + ") | D2 heading (" + rm["D2"]["HELD"]["window"] + ") |")
L.append("|---|---:|---:|")
for k, lab in (("pass_", "**pass** ⚠ LOW-N"), ("chase_follow", "**chase/follow** (D1: approach or co_move · D2: toward)"),
               ("co_move", "**co_move** on `*_with` (floor 60%)"), ("approach", "approach ⚠ LOW-N (D2: toward)"),
               ("stationary_fp", "false positive: subject-driven motion on stand/sit/lie subjects"),
               ("with_toward_away_fp", "false positive: approach/recede (D1) or toward/away (D2) on `*_with`")):
    L.append(f"| {lab} | {pct(rm['D1']['HELD'][k])} | {pct(rm['D2']['HELD'][k])} |")
L.append("\nThe same, on FIT: " + "; ".join(
    f"{lab} D1 {pct(rm['D1']['FIT'][k])} vs D2 {pct(rm['D2']['FIT'][k])}"
    for k, lab in (("pass_", "pass"), ("chase_follow", "chase/follow"), ("co_move", "co_move"))) + ".\n")
L.append("What each definition outputs on HELD chase/follow slots: D1 " + ", ".join(
    f"{k} {v}" for k, v in sorted(rm["D1"]["HELD"]["dist_cf"].items(), key=lambda x: -x[1]))
    + "; D2 " + ", ".join(f"{k} {v}" for k, v in sorted(rm["D2"]["HELD"]["dist_cf"].items(), key=lambda x: -x[1])) + ".\n")
L.append(f"**Decision (rule fixed in the brief): {dec['choice']}.** {dec['why']}. {dec['floor']}.\n")
L.append("Two things the rule does not show, recorded so nobody mistakes them for evidence:\n")
L.append(f"- **The pass criterion was decided by a single instance.** On HELD, D1 detects "
         f"{pct(rm['D1']['HELD']['pass_'])} and D2 {pct(rm['D2']['HELD']['pass_'])}. On FIT "
         f"the order reverses: D1 {pct(rm['D1']['FIT']['pass_'])}, D2 {pct(rm['D2']['FIT']['pass_'])}. "
         f"**Neither definition detects pass.**")
L.append(f"- **The chase/follow criterion is asymmetric by construction.** D1 counts "
         f"approach *or* co_move as a hit, while D2 needs toward. "
         f"{rm['D1']['HELD']['dist_cf'].get('co_move', 0)} of D1's "
         f"{rm['D1']['HELD']['chase_follow'][0]} HELD hits are co_move, and only "
         f"{rm['D1']['HELD']['dist_cf'].get('approach', 0)} are approach. D1 wins because "
         f"chasers co-move at slot scale, not because it detects approach.\n")
L.append("**Approach and recede remain unvalidated on base data.** There are 12 base "
         "approach instances and 0 recede, and `pass` fails under both definitions. The "
         "novel test AP of the 12 `*_away` / `*_toward` predicates (and the `*_past` "
         "ones) is their test.\n")

L.append("## Task 3 — stopped vs moving vehicles\n")
L.append(f"motion_rate at the chosen subject_state window ({ch['subject_state']}); "
         f"move_* slots are positive, stop_* negative; vehicle subjects only "
         f"({', '.join(S['vehicles'])}):\n")
L.append("| slots | move | stop | AUC (95% CI, video-clustered) |")
L.append("|---|---:|---:|---:|")
for k, v in t3.items():
    L.append(f"| {k} | {v['n_move']} | {v['n_stop']} | {ci(v['auc'])} |")
open(os.path.join(HERE, "slot_check.md"), "w").write("\n".join(L) + "\n")

# ======================= preregistration.md =======================
files = {
    "schema": "pilot_analysis/stage0e/phi_frozen.json",
    "session-1 measured axes (definitions)": "pilot_analysis/stage1/measured_axes.py",
    "slot grid, windows, D1/D2 (definitions)": "pilot_analysis/stage1b/slot_axes.py",
    "per-window fitted thresholds": "pilot_analysis/stage1b/thresholds_slot.json",
    "FIT/HELD split": "pilot_analysis/stage1/train_split.json",
    "paraphrases": "pilot_analysis/stage1b/paraphrases_frozen.json",
}
w_rm = ch[f"relative_motion {dec['choice']}"]
P = [f"# Pre-registration — measured axes and paraphrases for stage 2\n",
     f"**Date:** {DATE}. **Status:** frozen. Any later change needs a dated amendment "
     f"appended below, saying what changed and why. This file's SHA256 is in "
     f"`preregistration.sha256`.\n",
     "## Frozen artefacts\n", "| artefact | path | SHA256 |", "|---|---|---|"]
for k, p in files.items():
    P.append(f"| {k} | `{p}` | `{sha(os.path.join(ROOT, p))}` |")
P += ["\n## Scoring grid\n",
      "One score vector per (ordered pair, 30-frame slot). A pair is two distinct tracks "
      "with ≥ 10 overlapping frames; the overlap is cut into 30-frame slots; a final "
      "partial slot is kept if ≥ 10 frames, else trimmed. Stage 2 builds these from "
      "**whole** tracks, not from `train_object_trajectories_gt.json`, which holds "
      "fragments. Windows: W0 = the slot; W1/W2 = ± 1/2 slots; W3 = the whole overlap; "
      "all clipped to the overlap.\n",
      "## Frozen measured rules\n",
      "| axis | rule (implementation) | window | threshold |", "|---|---|---|---|",
      f"| horizontal | sign of mean(subject − object centre x); < 0 = left (`M.horizontal`) | {ch['horizontal']} | none |",
      f"| vertical | sign of mean centre-y difference; subject higher = above (`M.vertical`) | {ch['vertical']} | none |",
      f"| depth | sign of mean bottom-edge y difference; lower = front (`M.depth`) | {ch['depth']} | none |",
      f"| proximity: adjacent | mean gap ÷ mean box diagonal < threshold, and not contained (`M.proximity`) | {ch['proximity']} | gap {TH[ch['proximity']]['gap']:.5f} |",
      f"| proximity: contained | in a majority of frames, subject-in-object fraction > object-in-subject fraction AND subject top edge not above object top edge (`M.contained`) | {ch['proximity']} | none |",
      f"| subject_state | `M.subject_motion` (net compensated displacement in own box diagonals per frame) > threshold = moving | {ch['subject_state']} | {TH[ch['subject_state']]['motion']:.6f} |",
      f"| object_state | `M.object_motion` > threshold = moving | {ch['object_state']} | {TH[ch['object_state']]['motion']:.6f} |",
      f"| relative_motion | **{dec['choice']}**, " + ("distance: `S.d1_decide(S.d1_stats(pair), dead_zone, motion)` — pass, then approach/recede on the change in normalised centre distance beyond the dead zone, then co_move (both moving, net directions agree), else none" if dec["choice"] == "D1" else "heading: `S.d2_decide(S.d2_stats(pair, heading), c, motion)`") + f" | {w_rm} | " + (f"dead zone {TH[w_rm]['dead_zone']:.5f}; motion {TH[w_rm]['motion']:.6f}" if dec["choice"] == "D1" else f"c {TH[w_rm]['c']:.3f}; motion {TH[w_rm]['motion']:.6f}") + " |",
      f"| comparative: larger | sign of mean log area ratio (`M.larger_score`) | {ch['comparative=larger']} | none |",
      f"| comparative: taller | sign of mean log height ratio (`M.taller_score`) | {ch['comparative=taller']} | none |",
      f"| comparative: faster | sign of log speed ratio, specified speed (`M.faster_score`) | {ch['comparative=faster']} | none |",
      "\nCamera compensation: `VidVRD_ECC_train.json`; `ecc[str(t+1)]` maps frame t−1 into "
      "frame t (0-based). Symmetry under time reversal and subject/object swap is exact "
      "at every window (slot_check.md).\n",
      "## Scoring rule (carried forward)\n",
      "Unset locomotion and posture assert absence **for compositional predicates only** "
      "(a bare spatial term, or verb + spatial term). Every other unset axis is "
      "marginalised. MEASURED axes have no trained parameters. LEARNED: locomotion, "
      "posture, contact, contact_action. subject_kind is supplied by the object "
      "classifier.\n",
      "## relative_motion — the decision, and what stays untested\n",
      f"Chosen: **{dec['choice']}** by the rule fixed in the session-2 brief: {dec['why']}; "
      f"{dec['floor']}. The HELD evidence is in slot_check.md. The pass criterion was "
      f"decided by one instance (D1 {pct(rm['D1']['HELD']['pass_'])} vs D2 "
      f"{pct(rm['D2']['HELD']['pass_'])}). **Approach and recede remain unvalidated on base "
      f"data (12 and 0 base instances), and pass is undetected by either definition. The "
      f"novel test AP of the `*_away`, `*_toward` and `*_past` predicates is their test.**\n",
      "## Paraphrases\n",
      f"`paraphrases_frozen.json`, SHA256 `{sha(os.path.join(HERE, 'paraphrases_frozen.json'))}`. "
      "It was written by a fresh subagent that saw only the 132 predicate strings and the "
      "three rules. It is saved verbatim and never edited. Known departures from the "
      "annotation meaning are recorded here, not fixed: 12 `*_front` predicates include "
      "\"ahead of\", which is heading-relative, whereas VidVRD front is camera-frame (stage "
      "0b). All 8 `stop_*` predicates include \"halts\" / \"comes to a stop\", an event "
      "reading, whereas VidVRD stop means a stationary vehicle (stage 0c).\n",
      "## Amendments\n", "_None._\n"]
open(os.path.join(HERE, "preregistration.md"), "w").write("\n".join(P) + "\n")
open(os.path.join(HERE, "preregistration.sha256"), "w").write(
    f"{sha(os.path.join(HERE, 'preregistration.md'))}  preregistration.md\n")

# ======================= README.md =======================
rnd = random.Random(20260928)
pick = sorted(rnd.sample(sorted(PARA), 10))
R = ["# Stage 1 session 2 — slot-level check, relative-motion rule, paraphrase freeze\n",
     "$0: CPU only. The repo is untouched: all output is in this directory. BASE training "
     "labels only, and no test annotations. Details: `slot_check.md`. Frozen record: "
     "`preregistration.md` (SHA256 in `preregistration.sha256`).\n",
     "```\npython3 pilot_analysis/stage1b/slot_check.py   # ~8 min CPU\n"
     "python3 pilot_analysis/stage1b/report.py       # writes the .md files\n```\n",
     "## Per-axis: session 1 (per instance) vs slot level (the classifier's grid)\n",
     "| axis | session 1: agreement / AUC / verdict | slot level: window, agreement (per value), AUC | slot verdict |",
     "|---|---|---|---|"]
for r in rows:
    s1a = f"{r['s1_auc'][0]:.3f}" if r["s1_auc"] else "—"
    R.append(f"| {r['axis']} | {r['s1_agr']:.1%} / {s1a} / {r['s1_v'].split(' (')[0]} | {r['w']}: "
             + "; ".join(f"{k} {v[0] / v[1]:.1%}" for k, v in r["vals"].items())
             + f", AUC {ci(r['auc'])} | **{r['v']}** |")
R.append(f"\nSame verdict rules as session 1. Symmetry: **{'exact at every window' if sym_exact else 'NOT EXACT'}**. "
         f"Grid: {S['grid']['labelled_slots']} labelled slots ({S['n']['fit']} FIT / {S['n']['held']} HELD).\n")
chg = [r for r in rows if r["v"].split(" (")[0] != r["s1_v"].split(" (")[0]]
adj = S["held"]["proximity"]["agree"]["adjacent"]
R.append("**Verdicts that changed at slot level:** " + (", ".join(
    f"{r['axis']} ({r['s1_v'].split(' (')[0]} → {r['v']})" for r in chg) if chg else "none") + ". "
    f"Proximity at {ch['proximity']}: adjacent {pct(adj)}, negatives {pct(prox)}. "
    f"subject_state for `stop`: {pct(S['held']['subject_state']['by_verb']['stop'])}.\n")
R.append("## relative_motion: D1 or D2\n")
R.append("| HELD | D1 distance | D2 heading |", )
R.append("|---|---:|---:|")
R.append(f"| window (chosen on FIT) | {rm['D1']['HELD']['window']} | {rm['D2']['HELD']['window']} |")
for k, lab in (("pass_", "pass ⚠ LOW-N"), ("chase_follow", "chase/follow"), ("co_move", "co_move (floor 60%)"),
               ("stationary_fp", "subject-driven motion on stand/sit/lie (false positive)")):
    R.append(f"| {lab} | {pct(rm['D1']['HELD'][k])} | {pct(rm['D2']['HELD'][k])} |")
R.append(f"\n**Frozen: {dec['choice']}.** {dec['why']}; {dec['floor']}. The rule was applied as "
         f"written, but its evidence is thinner than the table suggests:\n")
R.append(f"- **pass was decided by one instance** (HELD {rm['D1']['HELD']['pass_'][0]} vs "
         f"{rm['D2']['HELD']['pass_'][0]} of {rm['D1']['HELD']['pass_'][1]}; FIT reverses it, "
         f"{rm['D1']['FIT']['pass_'][0]} vs {rm['D2']['FIT']['pass_'][0]} of "
         f"{rm['D1']['FIT']['pass_'][1]}). Neither definition detects pass.")
R.append(f"- **The chase/follow criterion favours D1 by construction.** D1 counts co_move as "
         f"a hit, and {rm['D1']['HELD']['dist_cf'].get('co_move', 0)} of its "
         f"{rm['D1']['HELD']['chase_follow'][0]} hits are co_move; only "
         f"{rm['D1']['HELD']['dist_cf'].get('approach', 0)} are approach. D2 has the far lower "
         f"false-positive rate on stationary subjects.")
R.append("- **Approach and recede remain unvalidated on base data** (12 and 0 base instances). "
         "The novel test AP of `*_away`, `*_toward` and `*_past` is their test.\n")
a3 = t3["all training videos (nothing fitted)"]; h3 = t3["HELD only"]
R.append("## Stopped vs moving vehicles\n")
R.append(f"AUC of motion_rate, move_* vs stop_* slots, vehicle subjects, at {ch['subject_state']}: "
         f"**{ci(a3['auc'])}** over all training videos ({a3['n_move']} move / {a3['n_stop']} stop "
         f"slots; nothing is fitted in this comparison). HELD only: {ci(h3['auc'])} "
         f"({h3['n_move']} / {h3['n_stop']}). "
         + ("The all-videos CI excludes 0.5, so a separator exists, but it is weak. The "
            "limitations note should say *weak*, not *absent*. The HELD-only CI includes "
            "0.5 at this sample size."
            if a3["auc"][1] > 0.5 else
            "The CI includes 0.5: the four novel stop_* and five novel move_* predicates "
            "have no working separator.") + "\n")
R.append("## Paraphrases (frozen)\n")
R.append(f"`paraphrases_frozen.json`: 132 predicates × 4, SHA256 "
         f"`{sha(os.path.join(HERE, 'paraphrases_frozen.json'))}`. Both files are read-only. "
         f"It was produced by a fresh subagent whose prompt held only the 132 strings, the "
         f"three rules, the output format, and an instruction not to use tools. It made no "
         f"tool calls apart from returning its answer. 10 random examples (seed 20260928):\n")
R.append("| predicate | paraphrases |")
R.append("|---|---|")
for k in pick:
    R.append(f"| `{k}` | " + " · ".join(PARA[k]) + " |")
R.append("\n⚠ For the human read, two systematic meaning shifts. They are recorded, not "
         "edited: **\"ahead of\" appears in all 12 `*_front` predicates.** That is "
         "heading-relative, whereas VidVRD front is camera-frame (0b). **\"halts\" / \"comes "
         "to a stop\" appears in all 8 `stop_*` predicates.** That is an event, whereas VidVRD "
         "stop means a stationary vehicle (0c). Minor: `play` → \"plays\" drops the object; "
         "`run_above` → \"is running up above\" adds \"up\".\n")
open(os.path.join(HERE, "README.md"), "w").write("\n".join(R) + "\n")
print("symmetry exact:", sym_exact, "| decision:", dec["choice"])
