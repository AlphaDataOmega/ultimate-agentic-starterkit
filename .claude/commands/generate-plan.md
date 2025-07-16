# /generate-plan Command: Synthesize PLANNING.md from Context & Discussions

This command generates /docs/PLANNING.md once discussions are handled. No args needed.

Steps:
1. Read /docs/CONTEXT.md, PLANNING.md (assumptions/preferences).
2. Scan /docs/discussions/ for all DISCUSSION_*.md (full conversation history).
3. Analyze: Extract key decisions, phases, milestones, risks, and timelines.
4. Research: Use tools for realistic context according to the current date.
5. Synthesize: Create PLANNING.md organized by milestones/phases (not time); include deliverables, dependencies, validations, risks.
6. Structure: MD with tables for clarity; emphasize fast builds via Context Engineering.
7. Output: Write to /docs/PLANNING.md; summary of changes.
