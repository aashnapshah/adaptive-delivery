---
description: Read demo/GOAL.md and build the pipeline demo per its build order
argument-hint: "[optional: which build-order step to start at, e.g. 'step 2']"
---

You are implementing the demo specified in the brief `GOAL.md` (the file lives in the `demo/`
folder — if it is not in the current directory, find it with a quick search before doing
anything else). That brief is the single source of truth.

Do this:

1. Read `GOAL.md` in full.
2. Implement it following its **"Build order"** section, top to bottom. If the user passed an
   argument ($ARGUMENTS), start at that step instead.
3. Respect the **"Hard constraints"** and treat **"Decisions (locked)"** as settled — do NOT
   re-open or re-ask them.
4. Check your work against each tab's **acceptance criteria** (the `✓` lines) before calling a
   stage done.
5. Work in slices: after each build-order milestone, make sure it actually runs
   (serve `demo/` and open the relevant page), then briefly report what's done and what's next.
6. Only stop to ask the user if you hit a genuine blocker not covered by the brief.

Keep everything offline / no-build-step as the brief requires.
