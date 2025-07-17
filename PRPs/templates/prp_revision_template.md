# PRP Revision Template - Retry Scenarios

## Overview
This template is used when a work order fails validation and needs to be retried. It references the original PRP but includes additional context about failures and revised approach.

## Template Structure

### 1. Revision Header
```markdown
# Project Revision: [PROJECT_NAME]
**Original PRP**: [ORIGINAL_PRP_FILE_PATH]
**Revision Number**: [1-5] (Max 5 attempts)
**Created**: [DATE]
**Work Order ID**: [WORK_ORDER_ID]
**Status**: RETRY
**Previous Failure**: [BRIEF_DESCRIPTION]
```

### 2. Failure Analysis
```markdown
## Previous Failure Analysis

### What Failed
- **Testing Issues**: [SPECIFIC_TEST_FAILURES]
- **Visual Issues**: [VISUAL_VALIDATION_FAILURES]
- **Runtime Errors**: [CONSOLE_ERRORS_LOGGED]
- **Integration Issues**: [INTEGRATION_PROBLEMS]

### Error Details
```
[FULL_ERROR_LOGS_AND_STACK_TRACES]
```

### Root Cause Analysis
[ANALYSIS_OF_WHY_THE_IMPLEMENTATION_FAILED]
```

### 3. Revised Approach
```markdown
## Revised Implementation Strategy

### Key Changes from Previous Attempt
- [CHANGE_1]: [DESCRIPTION_AND_RATIONALE]
- [CHANGE_2]: [DESCRIPTION_AND_RATIONALE]
- [CHANGE_3]: [DESCRIPTION_AND_RATIONALE]

### Additional Research Conducted
- [RESEARCH_FINDING_1]
- [RESEARCH_FINDING_2]

### New Technical Requirements
- [NEW_REQUIREMENT_1]
- [NEW_REQUIREMENT_2]
```

### 4. Claude Code Agent Instructions (Revised)
```markdown
## Claude Code Agent Instructions - Revision [NUMBER]

### Learn from Previous Failure
**DO NOT repeat these mistakes:**
- [MISTAKE_1]: [DESCRIPTION]
- [MISTAKE_2]: [DESCRIPTION]
- [MISTAKE_3]: [DESCRIPTION]

### Revised Implementation Focus
- **Primary Goal**: [REVISED_MAIN_OBJECTIVE]
- **Critical Success Factors**: [WHAT_MUST_WORK_THIS_TIME]
- **Alternative Approach**: [DIFFERENT_TECHNICAL_APPROACH]

### Enhanced Requirements
- [ADDITIONAL_REQUIREMENT_1]
- [ADDITIONAL_REQUIREMENT_2]
- [ADDITIONAL_REQUIREMENT_3]

### Testing Considerations
- **Test for**: [SPECIFIC_SCENARIOS_THAT_FAILED_BEFORE]
- **Validate**: [SPECIFIC_VALIDATION_POINTS]
- **Ensure**: [SPECIFIC_FUNCTIONALITY_REQUIREMENTS]
```

### 5. Expected Browser Appearance (Revised)
```xml
<expected_appearance>
  <description>Revised description based on previous failure analysis</description>
  <elements>
    <element>Corrected element descriptions</element>
    <element>Additional elements that were missing</element>
  </elements>
  <interactions>
    <interaction>Fixed interaction behaviors</interaction>
    <interaction>Additional interactions discovered</interaction>
  </interactions>
  <error_scenarios>
    <error_scenario>What should happen when error occurs</error_scenario>
    <error_scenario>Fallback behavior expectations</error_scenario>
  </error_scenarios>
</expected_appearance>
```

### 6. Validation Checkpoints
```markdown
## Enhanced Validation Framework

### Pre-Implementation Validation
- [ ] Failure analysis reviewed
- [ ] Alternative approach validated
- [ ] Technical requirements confirmed
- [ ] Dependencies verified

### Post-Implementation Validation
- [ ] Previous failure scenarios specifically tested
- [ ] New test cases pass
- [ ] Visual validation with corrected expectations
- [ ] Integration testing with existing code
- [ ] Performance requirements met (if applicable)

### Escalation Criteria
- [ ] If this attempt fails, escalate to BugBountyAgent
- [ ] If max retries (5) reached, create detailed failure report
- [ ] If blocking issue identified, pause work order
```

### 7. Retry History
```markdown
## Retry History

### Attempt 1
- **Date**: [DATE]
- **Failure**: [BRIEF_DESCRIPTION]
- **Duration**: [TIME_SPENT]

### Attempt 2
- **Date**: [DATE]
- **Failure**: [BRIEF_DESCRIPTION]
- **Duration**: [TIME_SPENT]

### Current Attempt: [NUMBER]
- **Date**: [DATE]
- **Strategy**: [CURRENT_STRATEGY]
- **Expected Outcome**: [WHAT_SUCCESS_LOOKS_LIKE]
```

### 8. Success Criteria
```markdown
## Revised Success Criteria

### Must Pass
- [ ] All previous failure scenarios resolved
- [ ] Core functionality works as specified
- [ ] Tests pass without flakiness
- [ ] Visual validation succeeds
- [ ] No console errors

### Should Pass
- [ ] Performance is acceptable
- [ ] Code quality is maintainable
- [ ] Integration is seamless
- [ ] Documentation is updated

### Nice to Have
- [ ] Additional features discovered during implementation
- [ ] Improved error handling
- [ ] Enhanced user experience
```

## Usage Instructions

### 1. Creating a Revision PRP
1. Copy this template to `PRPs/[work_order_id]_revision_[attempt_number].md`
2. Reference the original PRP file
3. Include detailed failure analysis
4. Provide revised approach

### 2. TestingValidationAgent Integration
- TestingValidationAgent creates revision PRPs automatically
- Includes captured error logs and analysis
- Provides technical recommendations for fixes

### 3. Escalation Process
- After 5 failed attempts, escalate to BugBountyAgent
- BugBountyAgent gets full revision history
- Creates comprehensive failure analysis report

### 4. Knowledge Base Updates
- Each revision teaches the system about common failure patterns
- Successful revisions update best practices
- Failure patterns inform future work order planning

---

*This revision template supports the agentic workflow's retry mechanism with structured failure analysis and progressive improvement.*