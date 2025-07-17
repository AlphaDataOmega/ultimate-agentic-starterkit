# Base PRP Template - Context-Rich for Agentic Workflow

## Overview
This template provides a comprehensive framework for Project Requirements & Planning (PRP) documents that integrate with the StarterKit's agentic workflow system. It provides clear, focused instructions for the Claude Code Agent while supporting the full validation pipeline.

## Template Structure

### 1. Project Header
```markdown
# Project: [PROJECT_NAME]
**Created**: [DATE]
**Work Order ID**: [WORK_ORDER_ID]
**Status**: [PENDING|IN_PROGRESS|COMPLETED|BLOCKED]
**Priority**: [HIGH|MEDIUM|LOW]
**Dependencies**: [LIST_OF_DEPENDENCIES]
```

### 2. Context Integration
```markdown
## Project Context

### Working Assumptions (Validated)
- [ ] [ASSUMPTION_1] - Status: [VERIFIED|PENDING|INVALID]
- [ ] [ASSUMPTION_2] - Status: [VERIFIED|PENDING|INVALID]
- [ ] [ASSUMPTION_3] - Status: [VERIFIED|PENDING|INVALID]

### Completed Work Orders
- [WO_ID]: [TITLE] - [STATUS] - [COMPLETION_DATE]
- [WO_ID]: [TITLE] - [STATUS] - [COMPLETION_DATE]

### Current Codebase Status
- **Files Created**: [NUMBER]
- **Key Components**: [LIST_OF_COMPONENTS]
- **Architecture Pattern**: [PATTERN_DESCRIPTION]
```

### 3. Work Order Specification
```markdown
## Work Order: [TITLE]

### Description
[DETAILED_DESCRIPTION_OF_WORK_TO_BE_DONE]

### Acceptance Criteria
- [ ] [CRITERION_1]
- [ ] [CRITERION_2]
- [ ] [CRITERION_3]

### Technical Requirements
- **Technology Stack**: [LANGUAGES/FRAMEWORKS]
- **Dependencies**: [LIBRARIES/SERVICES]
- **Performance**: [PERFORMANCE_REQUIREMENTS]
- **Security**: [SECURITY_REQUIREMENTS]

### Integration Points
- **Existing Code**: [HOW_TO_INTEGRATE_WITH_EXISTING]
- **APIs**: [API_ENDPOINTS_TO_CREATE_OR_MODIFY]
- **Database**: [DATABASE_SCHEMA_CHANGES]
```

### 4. Implementation Guidelines for Claude Code Agent
```markdown
## Claude Code Agent Instructions

### Your Primary Focus
- **ONLY implement the requested functionality**
- **Create minimal unit tests** for your code
- **Provide Expected Browser Appearance** description (if applicable)
- **Do NOT worry about comprehensive testing** - other agents handle validation

### File Organization
```
workspace/[PROJECT_NAME]/
├── src/
│   ├── [COMPONENT_FILES]
│   └── [FEATURE_FILES]
├── tests/
│   └── [BASIC_UNIT_TESTS]
└── [ADDITIONAL_FILES]
```

### Coding Standards
- Follow [LANGUAGE] conventions
- Include basic error handling
- Document public APIs with docstrings
- Follow security best practices

### Handoff Requirements
- [ ] Core functionality implemented
- [ ] Basic unit tests created
- [ ] Expected Browser Appearance described (if browser-based)
- [ ] Code documented with comments
```

### 5. Validation Loops
```markdown
## Validation Framework

### Pre-Implementation Validation
- [ ] Requirements review completed
- [ ] Architecture review completed
- [ ] Security review completed
- [ ] Dependency check completed

### Post-Implementation Validation
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Visual tests pass (if applicable)
- [ ] Performance requirements met
- [ ] Security requirements met
- [ ] Code review completed

### Retry Logic (Max 5 attempts)
**Attempt**: [1-5]
**Previous Failures**: 
- [FAILURE_1]: [DESCRIPTION] - [DATE]
- [FAILURE_2]: [DESCRIPTION] - [DATE]

**Revision Strategy**: [STRATEGY_FOR_CURRENT_ATTEMPT]
```

### 6. Deliverables for Claude Code Agent
```markdown
## Required Deliverables

### Source Code
- [ ] [FILE_1] - [DESCRIPTION]
- [ ] [FILE_2] - [DESCRIPTION]
- [ ] [FILE_3] - [DESCRIPTION]

### Basic Tests
- [ ] [TEST_FILE_1] - Basic unit tests for core functionality
- [ ] [TEST_FILE_2] - Integration tests if needed

### Expected Browser Appearance (if browser-based project)
```xml
<expected_appearance>
  <description>Brief description of what the browser should display</description>
  <elements>
    <element>Navigation bar with logo on left, user menu on right</element>
    <element>Main content area with data table showing 5 columns</element>
    <element>Footer with copyright and links</element>
  </elements>
  <interactions>
    <interaction>Clicking submit button should show success message</interaction>
    <interaction>Hovering over table rows should highlight them</interaction>
  </interactions>
</expected_appearance>
```

### Integration Notes
- [ ] How this integrates with existing code
- [ ] Configuration changes needed
- [ ] Dependencies added
```

### 7. Completion Tracking
```markdown
## Completion Tracking

### Progress Updates
- [DATE]: [PROGRESS_UPDATE]
- [DATE]: [PROGRESS_UPDATE]

### Blockers
- [DATE]: [BLOCKER_DESCRIPTION] - [RESOLUTION_STATUS]

### Lessons Learned
- [LESSON_1]
- [LESSON_2]

### Context Updates
- [NEW_ASSUMPTION_1] - Now verified
- [NEW_TECHNICAL_DECISION] - Impacts future work orders

### Next Steps
- [NEXT_STEP_1]
- [NEXT_STEP_2]
```

## Usage Instructions

### 1. Creating a New PRP
1. Copy this template to `PRPs/[work_order_id]_prp.md`
2. Fill in project-specific information
3. Update context from knowledge base
4. Define clear acceptance criteria

### 2. During Implementation
1. Update progress regularly
2. Document any blockers immediately
3. Update validation checkboxes as completed
4. Record lessons learned

### 3. Validation Loops
1. If validation fails, create revision document
2. Reference original PRP but don't modify it
3. Document failure reasons and new strategy
4. Maximum 5 retry attempts before escalation

### 4. Integration with Agents
- **ParserAgent**: Extracts tasks from PRP
- **CoderAgent**: Implements based on PRP requirements
- **TesterAgent**: Validates against acceptance criteria
- **TestingValidationAgent**: Handles retry logic
- **DocumentationAgent**: Updates final documentation

## Claude Code Integration

### Context Preparation
The PRP template is designed to work with Claude's large context window:
- Full project context included
- Working assumptions clearly stated
- Existing code integration points defined
- Clear implementation guidelines provided

### Visual Testing Support
For visual work orders:
- Screenshot requirements specified
- Visual acceptance criteria defined
- UI/UX guidelines included
- Component integration defined

### Error Handling
Built-in retry mechanism:
- Original PRP remains unchanged
- Revision documents created for retries
- Error analysis included in revisions
- Escalation path defined (BugBountyAgent)

## Template Validation

### Required Sections
- [ ] Project Header with all metadata
- [ ] Context Integration with current state
- [ ] Work Order Specification with clear criteria
- [ ] Implementation Guidelines with file organization
- [ ] Validation Framework with checkboxes
- [ ] Deliverables with specific artifacts
- [ ] Completion Tracking with progress updates

### Quality Checks
- [ ] Acceptance criteria are testable
- [ ] Technical requirements are specific
- [ ] Integration points are clearly defined
- [ ] File organization follows project structure
- [ ] Validation loops are properly configured

---

*This template is version 2.0 and supports the complete agentic workflow with context-rich planning and validation loops.*