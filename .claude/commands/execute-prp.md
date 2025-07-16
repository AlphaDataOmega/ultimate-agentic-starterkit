# Execute BASE PRP

Implement a feature using using the PRP file.

## PRP File: $ARGUMENTS

## Execution Process

1. **Load PRP**
   - Read the specified PRP file
   - Understand all context and requirements
   - Follow all instructions in the PRP and extend the research if needed
   - Ensure you have all needed context to implement the PRP fully
   - Do more web searches and codebase exploration as needed

2. **ULTRATHINK**
   - Think hard before you execute the plan. Create a comprehensive plan addressing all requirements.
   - Break down complex tasks into smaller, manageable steps using your todos tools.
   - Use the TodoWrite tool to create and track your implementation plan.
   - Identify implementation patterns from existing code to follow.

3. **Execute the plan**
   - Execute the PRP
   - Implement all the code

4. **Validate**
   - Cross-check implementation against CONTEXT.md assumptions (e.g., no deletions ensured).
   - Ensure alignment with PLANNING.md (e.g., output ties to milestone deliverables).
   - If misalign: Auto-iterate or flag for discussion update.
   - Run each validation command
   - Fix any failures
   - Re-run until all pass

5. **Visual Testing** (if web interface)
   - See docs/VISUAL_TESTING.md for complete documentation
   - Run visual testing script to capture screenshots and logs
   - Use: `npm run visual-test <url>` or `npm run prp-validate <config-file>`
   - Analyze screenshot for visual correctness
   - Review console logs and network errors
   - Compare against expected behavior
   - Make adjustments if visual validation fails
   - Re-test until visual validation passes

6. **Documentation Update**
   - Update relevant documentation files /docs
   - Add/update README sections if needed
   - Document new features or changes
   - Ensure documentation reflects implementation

7. **Complete**
   - Ensure all checklist items done
   - Run final validation suite
   - Report completion status
   - Read the PRP again to ensure you have implemented everything

8. **Reference the PRP**
   - You can always reference the PRP again if needed

Note: If validation fails, use error patterns in PRP to fix and retry.