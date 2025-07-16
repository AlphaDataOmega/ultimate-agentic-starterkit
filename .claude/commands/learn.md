# /learn Command: Iterative Context Learning for PROJECTS

This command processes CONTEXT.md and discussions to build/evolve understanding. The goal is to be able to create docs/PLANNING.md and docs/CONTEXT.md to the fullest extent possible. PLANNING.md must be a comprehensive plan from init to production deployment for beta testing. CONTEXT.md must be a comprehensive list of assumptions and preferences to validate against. 

$ARGUMENTS = The latest discussion reply. The discussion documents are like a conversation back and forth. You are odd numbered files and my responses to your questions and comments are even numbered files, we go on like this until my knowledge has transfered to you about the project and you have helped me round out the specific details. You will think of different user edge cases and backend scenarios to keep me on my toes, while being mindful of what's important for a production ready product that can be optimized later with user feedback. 

Steps:
1. Gather the following for context if you haven't already. CONTEXT.md, All /docs/project-overview/*.md files.
2. You have already had a discussion with the user about these files to catch up you must scan /docs/discussions/ for DISCUSSION_*.md files. Read any that are not currently in your context. 
3. If none: Generate DISCUSSION_001.md with basic questions on project (vision, tech, etc.).
4. Research gaps using tools (e.g., web_search for trends).
5. If exist: Analyze chain; generate NEXT DISCUSSION_N.md with responses/follow-ups/questions. (Do Not Edit it)
6. Edit CONTEXT.md: Append new assumptions/learnings (one-per-line).
7. Edit PLANNING.md in the root folder to align with new assumptions/learnings.
8. Output: Summary of updates; link to new DISCUSSION.md where you can ask further questions and grow your understanding of the entire project from init to production deployment. Also for edge use case scenarios so we can prepare for anything necessary. 