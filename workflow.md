1. Paste prompt in Plan Mode
2. Review + approve architecture plan
3. Switch to Auto-accept for scaffolding
4. Switch to Plan Mode for complex files (4 key files)
5. Switch to Auto-accept for tests + scripts
6. Run all tests, verify everything works
7. Push to GitHub
8. Codex review on full codebase
9. Fix issues, final commit


The core tension is:

Auto-accept is fast but you lose visibility into WHY decisions were made
Plan mode is thorough but can be painfully slow for a 15+ file scaffolding day
Your preference says learning comes first, which means you need to understand the code, not just have it

Here's the workflow I'd recommend, broken into phases:

Phase 1: Plan Mode -- Architecture Review
Paste the Day 1 prompt into Claude Code in plan mode. Let it propose its implementation plan. Before it writes a single line of code, you should see:

How it plans to structure the abstract base class for data providers
What DuckDB schema it's proposing
How it handles the upsert logic
What the data validation approach looks like

Read this carefully. This is where 80% of the architectural mistakes happen. If something looks off or you don't understand a decision, ask Claude Code to explain before proceeding. This is your learning moment -- once you approve the plan, the code is just execution of decisions you already understood.
Phase 2: Auto-accept -- Scaffolding + Simple Files
Once the plan is approved, switch to auto-accept for:

Project structure (folders, pyproject.toml, Makefile, .gitignore)
Config files (settings.py, constants.py, universe.py)
Data models (models.py -- these are Pydantic schemas, straightforward)
Test fixtures (conftest.py)

These are low-risk files where there's basically one right way to do it. Watching Claude Code ask you to approve every folder creation is a waste of your time.
Phase 3: Plan Mode again -- Complex Implementation
Switch BACK to plan mode for the files that have real architectural decisions:

data/providers/base.py -- the abstract base class determines how every future data source works
data/storage/database.py -- DuckDB schema and upsert logic, get this wrong and you'll rebuild it on Day 4
data/validation/quality.py -- validation logic has edge cases that matter
data/ingest.py -- orchestration logic, error handling, resume capability

For each of these, review the plan, make sure you understand the approach, then let it execute.
Phase 4: Auto-accept -- Tests + Scripts
Switch back to auto-accept for tests and CLI scripts. These follow directly from the implementation above and are lower risk.
Phase 5: Codex Review -- AFTER everything is built
This is important -- don't use Codex during the build. Use it after. Here's why:
If you Codex-review every file as Claude Code produces it, you'll:

Break your flow constantly
Get conflicting suggestions between Claude Code's style and Codex's preferences
Spend the day context-switching instead of building

Instead, once Day 1 code is complete and tests pass, push to GitHub and run Codex review on the full codebase. Look for:

Inconsistencies across files (naming conventions, error handling patterns)
Missing edge cases in validation
Abstraction quality (will this be painful to extend on Day 2?)
Test coverage gaps

Fix what Codex flags, then commit your clean Day 1 codebase.