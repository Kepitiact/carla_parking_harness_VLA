# Contributing Guide

Thank you for your interest in contributing to the CARLA Parking Dataset Generator! This guide explains how to develop, test, and submit changes.

---

## Development Setup

### 1. Clone & Install

```bash
git clone https://github.com/Kepitiact/carla_data_gen_VLA.git
cd carla_data_gen_VLA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Understand the Codebase

- **README.md** — High-level overview and quick-start
- **ARCHITECTURE.md** — Design decisions, coordinate conventions, data flow
- **scripts/generate_episodes.py** — Main collection pipeline
- **scripts/build_*.py** — Post-processing pipeline
- **CLAUDE.md** — Development philosophy (simplicity, surgical edits, no speculation)

### 3. Read the Code Philosophy

From `CLAUDE.md`:
- **Simplicity first**: minimum code that solves the problem, no speculative features
- **Surgical changes**: touch only what you must, match existing style
- **No half-finished implementations**: don't leave TODOs without addressing them

---

## Workflow for Changes

### Reporting Bugs

Create a GitHub issue with:
1. **Title**: concise summary
2. **Reproduction steps**: exactly how to trigger it
3. **Expected vs. actual behavior**
4. **Environment**: Python version, CARLA version, GPU type

Example:
```
Title: CARLA crashes after ~250 episodes
Steps:
  1. Run: python scripts/generate_episodes.py --num_episodes 250
  2. Wait ~2 hours
Expected: Episode 250 completes normally
Actual: "Invalid session" errors, process hangs
Environment: Python 3.8.20, CARLA 0.9.14, RTX 2000 Ada
```

### Proposing Features

Before implementing, open an issue or discussion:
1. **What problem does it solve?**
2. **Is it in scope?** (parking datasets, OpenDriveVLA compatibility)
3. **What's the minimal implementation?**

Examples of in-scope:
- New parking maneuver types (parallel, forward)
- New CARLA maps (Town10HD_Opt)
- Faster path planning
- Better NPC collision handling

Examples of out-of-scope:
- General CARLA tutorials (use CARLA docs)
- Unrelated RL research (fork and customize)
- Production deployment / DevOps (separate repo)

### Making Changes

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature
   # or: git checkout -b fix/issue-description
   ```

2. **Make minimal, focused changes:**
   - One feature per PR
   - No reformatting unrelated code
   - No dependency upgrades (unless necessary and tested)

3. **Test locally:**
   ```bash
   # For collection script changes:
   python scripts/generate_episodes.py --num_episodes 5  # quick smoke test
   
   # For post-processing changes:
   python scripts/validate_episodes.py --raw_dir data/raw
   python scripts/build_infos_pkl.py
   python scripts/verify_dataset.py
   ```

4. **Document in code:**
   - Docstrings for functions: describe parameters, returns, raises
   - Inline comments for non-obvious logic (the "why", not the "what")
   - No docstring bloat or multi-line comment blocks

5. **Commit with clear messages:**
   ```bash
   git commit -m "Fix MPC premature gear switch via rear-axle offset

   RS path computed from center, MPC receives rear-axle waypoints.
   Rear axle 1.4m ahead of first waypoint caused nearest_index to
   jump 2-3 steps, triggering mode change at tick 3 instead of 10+.
   
   Apply WB offset to start position, not goal, so cx[0] aligns with
   actual rear-axle position at episode start."
   ```

6. **Push & create a pull request:**
   ```bash
   git push origin feature/your-feature
   # Then open PR on GitHub
   ```

---

## Code Standards

### Style

- **Python version:** 3.8 (CARLA 0.9.14 constraint)
- **Formatter:** (no enforced formatter; match existing style)
  - 4-space indentation
  - 80–100 char line limits
  - snake_case for functions/variables, CamelCase for classes
- **Type hints:** Encouraged for function signatures (helps IDE autocomplete)

### Patterns to Avoid

❌ **Don't:**
- Add features beyond what's requested
- Refactor working code unless it's broken
- Import unused modules
- Write "error handling for impossible scenarios" (trust the framework)
- Create intermediate abstractions (one-off code doesn't need helpers)

✅ **Do:**
- Match existing code style
- Remove imports/variables YOUR change made unused
- Comment the "why" (constraints, workarounds, non-obvious logic)
- Test your changes before submitting

### Example: Good vs. Bad Change

**Bad:**
```python
# Randomly refactored argument parsing; unrelated to the fix
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Generate parking episodes"
)
# ... 10 new lines of prettification
```

**Good:**
```python
# Only the fix: add --fix flag to validate_episodes
parser.add_argument("--fix", action="store_true", 
                    help="Delete invalid episodes")
```

---

## Performance & Testing

### Minimal Test Suite (Manual)

For collection script changes:
```bash
# Run 5 episodes (should take ~10 minutes)
python scripts/generate_episodes.py --num_episodes 5 --seed 999

# Verify output
python scripts/validate_episodes.py --raw_dir data/raw
ls data/raw/episode_*/poses.json | wc -l  # should be 5 (or fewer if failures)
```

For post-processing changes:
```bash
# Use existing data or collect 10 episodes
python scripts/generate_episodes.py --num_episodes 10 --seed 99

# Run the full pipeline
python scripts/validate_episodes.py --raw_dir data/raw --fix
python scripts/build_infos_pkl.py
python scripts/build_cache_pkl.py
python scripts/verify_dataset.py
# Should complete without errors
```

### Benchmarking

For performance-sensitive code (path planning, trajectory computation):
```bash
# Profile a run
python -m cProfile -s cumtime scripts/build_cache_pkl.py > profile.txt
head -30 profile.txt
```

If adding a feature increases wall-clock time, document the trade-off in the PR.

---

## Common Contribution Types

### Adding Support for a New Map

1. **Audit the map geometry:**
   ```bash
   python -c "
   import xml.etree.ElementTree as ET
   tree = ET.parse('ParkingScenes/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/YourMap.xodr')
   root = tree.getroot()
   roads = root.findall('road')
   print(f'Roads: {len(roads)}')
   # Count lane types, check parking areas, etc.
   "
   ```

2. **Update constants in `generate_episodes.py`:**
   ```python
   SLOT_GEOMETRY = {
       "Town04_Opt": {...},
       "YourMap": {"width_m": 3.0, "length_m": 5.5, "type": "parallel"},  # ADD
   }
   GOAL_YAW = {
       "Town04_Opt": 180.0,
       "YourMap": 90.0,  # ADD
   }
   ```

3. **Update slot selection logic (near line 965):**
   ```python
   if args.map == 'Town04_Opt':
       slot_indices = list(range(17, 33))
   elif args.map == 'YourMap':  # ADD
       # Audit before using! Verify all slots have same goal_yaw.
       slot_indices = list(range(start_id, end_id))
   else:
       slot_indices = list(range(len(parking_position.parking_vehicle_locations)))
   ```

4. **Test collection:**
   ```bash
   python scripts/generate_episodes.py --map YourMap --num_episodes 5
   python scripts/validate_episodes.py --raw_dir data/raw
   ```

5. **Update README.md** with the new map's geometry.

### Improving Path Planning

If implementing Hybrid A* or a faster planner:

1. **Keep the interface:** `_cached_plan()` must output the same CSV format
2. **Validate output:** RS curves work as drop-in replacement; new planner must too
3. **Benchmark:** Compare planning time and collision rate
4. **Document trade-offs:** Why is it better? What scenarios does it handle?

### Fixing a Collision Issue

1. **Identify the pattern:**
   ```bash
   # Grep failed episodes for collision actor names
   grep "COLLISION" logs/collect_A.log | head -20
   ```

2. **Filter or adjust:**
   - If actor type: add to `--no_spawn_types` flag
   - If location: add spatial filter like `if sp.x < 285.0: continue`
   - If timing: adjust `GOAL_HOLD_FRAMES` or speed limits

3. **Test on 20–50 episodes** to measure impact (success rate before/after)

4. **Update README.md** with results

---

## Pull Request Checklist

Before submitting a PR:

- [ ] Code matches existing style (no random reformatting)
- [ ] Tested locally (smoke test for collection, full pipeline for post-processing)
- [ ] Docstrings added/updated for new functions
- [ ] Commits have clear messages (explain the "why")
- [ ] Changes are minimal and focused
- [ ] Updated README.md if user-facing behavior changed
- [ ] No unnecessary dependencies added
- [ ] No dead code or imports left behind

---

## Questions?

- **How do I set up a local CARLA instance?** → See CARLA docs + README.md quick-start
- **What's the output format?** → See ARCHITECTURE.md "Data Flow" section
- **How do I debug an episode failure?** → Run with `--verbose`, check `logs/collect.log`
- **Can I collect on multiple machines?** → Yes, use different `--seed` per machine, then merge `data/raw/` directories

---

## Code of Conduct

Be respectful in issues and PRs. Assume good intent. Help each other learn.

