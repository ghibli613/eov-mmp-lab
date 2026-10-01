# Consequences of amendments A1–A3 (2026-09-28)

Schema: `stage0e/phi_frozen.json`, unchanged. The amendments change **tags**: A2 makes `pass` LEARNED and A3 makes `adjacent` LEARNED. A1 changes a **measurement rule** for approach/recede/co_move, not a tag or a phi value. Base support is counted from the training annotations, base predicates only.

## Novel predicates with no LEARNED positive ingredient: 14 → **10**

After the amendments: `above`, `away`, `beneath`, `move_above`, `move_away`, `move_toward`, `stop_above`, `stop_beneath`, `stop_with`, `toward`.

No longer in the set, because they now carry a LEARNED value: `move_next_to` (proximity=adjacent), `move_past` (relative_motion=pass), `past` (relative_motion=pass), `stop_next_to` (proximity=adjacent).

## Reachability: 5 → **5 unreachable** (unchanged)

`bite` (contact_action=bite (0)), `feed` (contact_action=feed (0)), `fight` (no ingredients), `hold` (contact_action=hold (0)), `kick` (contact_action=kick (0)).

The newly LEARNED values clear the ≥ 50 threshold: pass 54, adjacent 1342.

## Collisions

Identical phi: all-novel 0, base-novel 0, all-base 1 (chase, fall_off, follow). Re-tagging changes no phi value, so it **cannot** create a collision. Under the absence fill the groups are: chase, fall_off, follow. **No new novel collision.**

