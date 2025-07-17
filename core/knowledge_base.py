"""
Knowledge Base Management for the Ultimate Agentic StarterKit.

This module manages the multi-document project knowledge base that builds up
progressively as work orders are completed.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime
from enum import Enum

from core.models import ProjectTask, AgentResult
from core.logger import get_logger
from core.config import get_config


class ProjectType(str, Enum):
    """Types of projects that can be created."""
    GAME = "game"
    WEB_APP = "web_app"
    CLI_TOOL = "cli_tool"
    API_SERVICE = "api_service"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    ML_PROJECT = "ml_project"
    GENERIC = "generic"


class DocumentType(str, Enum):
    """Types of project documents in the knowledge base."""
    OVERVIEW = "OVERVIEW.md"
    CONTEXT = "CONTEXT.md"
    REQUIREMENTS = "REQUIREMENTS.md"
    ARCHITECTURE = "ARCHITECTURE.md"
    USER_STORIES = "USER_STORIES.md"
    DATA_MODELS = "DATA_MODELS.md"
    API_SPEC = "API_SPEC.md"
    SECURITY = "SECURITY.md"
    DEPLOYMENT = "DEPLOYMENT.md"
    BUSINESS_RULES = "BUSINESS_RULES.md"
    WORK_ORDERS = "work_orders/"
    COMPLETIONS = "completions/"


class WorkOrderStatus(str, Enum):
    """Status of work orders."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ProjectKnowledgeBase:
    """
    Manages the comprehensive project knowledge base with multiple specialized documents
    and progressive work order tracking.
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize the knowledge base."""
        self.project_root = Path(project_root)
        self.logger = get_logger("knowledge_base")
        self.config = get_config()
        
        # Knowledge base structure
        self.docs_dir = self.project_root / "docs"
        self.work_orders_dir = self.project_root / "work_orders"
        self.completions_dir = self.project_root / "completions"
        
        # Create directories
        for dir_path in [self.docs_dir, self.work_orders_dir, self.completions_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Document templates
        self.document_templates = {
            DocumentType.CONTEXT: self._get_context_template(),
            DocumentType.REQUIREMENTS: self._get_requirements_template(),
            DocumentType.ARCHITECTURE: self._get_architecture_template(),
            DocumentType.USER_STORIES: self._get_user_stories_template(),
            DocumentType.DATA_MODELS: self._get_data_models_template(),
            DocumentType.API_SPEC: self._get_api_spec_template(),
            DocumentType.SECURITY: self._get_security_template(),
            DocumentType.DEPLOYMENT: self._get_deployment_template(),
            DocumentType.BUSINESS_RULES: self._get_business_rules_template()
        }
        
        # Work order tracking
        self.work_order_index = self._load_work_order_index()
    
    async def detect_project_type(self, overview_content: str) -> ProjectType:
        """Detect project type from overview content."""
        content_lower = overview_content.lower()
        
        # Game indicators
        game_keywords = ['game', 'player', 'score', 'level', 'paddle', 'ball', 'pygame', 'unity', 'godot']
        if any(keyword in content_lower for keyword in game_keywords):
            return ProjectType.GAME
        
        # CLI tool indicators
        cli_keywords = ['cli', 'command', 'terminal', 'script', 'tool', 'argparse', 'click', 'typer']
        if any(keyword in content_lower for keyword in cli_keywords):
            return ProjectType.CLI_TOOL
        
        # API service indicators
        api_keywords = ['api', 'service', 'endpoint', 'microservice', 'rest', 'graphql', 'fastapi', 'flask']
        if any(keyword in content_lower for keyword in api_keywords):
            return ProjectType.API_SERVICE
        
        # Mobile app indicators
        mobile_keywords = ['mobile', 'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin']
        if any(keyword in content_lower for keyword in mobile_keywords):
            return ProjectType.MOBILE_APP
        
        # Desktop app indicators
        desktop_keywords = ['desktop', 'electron', 'tkinter', 'pyqt', 'wxpython', 'kivy']
        if any(keyword in content_lower for keyword in desktop_keywords):
            return ProjectType.DESKTOP_APP
        
        # ML project indicators
        ml_keywords = ['machine learning', 'ml', 'ai', 'neural network', 'tensorflow', 'pytorch', 'sklearn', 'model']
        if any(keyword in content_lower for keyword in ml_keywords):
            return ProjectType.ML_PROJECT
        
        # Web app indicators (more comprehensive check)
        web_keywords = ['web', 'frontend', 'backend', 'database', 'react', 'vue', 'angular', 'django', 'express']
        if any(keyword in content_lower for keyword in web_keywords):
            return ProjectType.WEB_APP
        
        # Default to web app if uncertain
        return ProjectType.WEB_APP
    
    def _get_context_template(self) -> str:
        """Get template for CONTEXT.md - working assumptions."""
        return """# Project Context & Working Assumptions

## Core Assumptions
*These are fundamental truths that will be true when the project is built*

### User Interaction Assumptions
- [ ] Users can navigate using mouse/keyboard/touch
- [ ] Users have basic computer literacy
- [ ] Users understand common UI patterns (buttons, forms, menus)

### Technical Environment Assumptions  
- [ ] Target browsers support modern JavaScript (ES6+)
- [ ] Users have stable internet connection for web apps
- [ ] Server environment supports chosen technology stack
- [ ] Database can handle expected concurrent users

### Business Logic Assumptions
- [ ] User authentication is required for protected features
- [ ] Data validation happens on both client and server
- [ ] Error states are handled gracefully
- [ ] System maintains data consistency

### Integration Assumptions
- [ ] Third-party APIs are available and stable
- [ ] Payment processing (if applicable) follows security standards
- [ ] External services have documented APIs
- [ ] Backup/recovery procedures are in place

### Performance Assumptions
- [ ] Response times under 200ms for critical operations
- [ ] System can handle [X] concurrent users
- [ ] Database queries are optimized
- [ ] Caching strategies are implemented where needed

### Security Assumptions
- [ ] All user inputs are validated and sanitized
- [ ] Authentication tokens are secure and time-limited
- [ ] Sensitive data is encrypted in transit and at rest
- [ ] Access controls are properly implemented

## Project-Specific Context

### Domain Knowledge
*Add domain-specific assumptions here*

### Integration Context  
*External systems, APIs, services this project depends on*

### User Personas & Behaviors
*How users will interact with the system*

### Success Metrics
*How we'll know the project is working correctly*

---
*This file is updated as work orders are completed and assumptions are validated*
"""
    
    def _get_requirements_template(self) -> str:
        """Get template for detailed functional requirements."""
        return """# Detailed Requirements Specification

## Functional Requirements

### Core Features
1. **Feature 1**: [Name]
   - **Purpose**: What it does and why
   - **Acceptance Criteria**: 
     - [ ] Criterion 1
     - [ ] Criterion 2
   - **Edge Cases**: What could go wrong
   - **Dependencies**: Other features this relies on

### User Requirements
- **User Role 1**: Can do X, Y, Z
- **User Role 2**: Can do A, B, C

### System Requirements
- **Performance**: Response time, throughput, capacity
- **Scalability**: Growth expectations
- **Reliability**: Uptime, error rates
- **Security**: Authentication, authorization, data protection

### Integration Requirements
- **External APIs**: What we need to integrate with
- **Data Import/Export**: File formats, frequencies
- **Third-party Services**: Payment, email, storage

### Compliance Requirements
- **Regulatory**: GDPR, HIPAA, PCI-DSS, etc.
- **Accessibility**: WCAG guidelines
- **Browser Support**: Which browsers/versions

---
*Generated and refined through learning phase*
"""
    
    def _get_architecture_template(self) -> str:
        """Get template for system architecture."""
        return """# System Architecture

## Technology Stack
- **Frontend**: Framework, libraries, build tools
- **Backend**: Language, framework, database
- **Infrastructure**: Hosting, CDN, monitoring
- **DevOps**: CI/CD, testing, deployment

## System Design

### High-Level Architecture
```
[Frontend] ←→ [API Gateway] ←→ [Backend Services] ←→ [Database]
```

### Component Architecture
- **Presentation Layer**: UI components, state management
- **Business Logic Layer**: Services, domain models
- **Data Access Layer**: Repositories, ORM
- **Infrastructure Layer**: External APIs, file storage

### Data Flow
1. User interaction → Frontend
2. API calls → Backend
3. Business logic processing
4. Database operations
5. Response back to frontend

### Security Architecture
- **Authentication**: Method, token management
- **Authorization**: Role-based access control
- **Data Protection**: Encryption, validation

### Deployment Architecture
- **Environments**: Development, staging, production
- **Scalability**: Load balancing, auto-scaling
- **Monitoring**: Logging, metrics, alerting

---
*Evolves as technical decisions are made during development*
"""
    
    def _get_user_stories_template(self) -> str:
        """Get template for detailed user stories."""
        return """# User Stories & Acceptance Criteria

## Epic 1: [Epic Name]

### Story 1.1: [Story Name]
**As a** [user type]
**I want** [functionality]
**So that** [benefit/value]

**Acceptance Criteria:**
- [ ] Given [context], when [action], then [result]
- [ ] Given [context], when [action], then [result]

**Definition of Done:**
- [ ] Code written and reviewed
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] UI/UX review completed
- [ ] Accessibility tested
- [ ] Performance validated

### Story 1.2: [Story Name]
[Same format as above]

## Edge Cases & Error Scenarios
- **No network connection**: How system behaves
- **Invalid user input**: Validation and feedback
- **System overload**: Graceful degradation
- **Data corruption**: Recovery procedures

## User Journey Maps
1. **New User Journey**: Registration → Setup → First use
2. **Power User Journey**: Advanced features and workflows
3. **Error Recovery Journey**: How users recover from problems

---
*Stories refined as user needs become clearer*
"""
    
    def _get_data_models_template(self) -> str:
        """Get template for data models and schemas."""
        return """# Data Models & Schema Design

## Entity Relationship Diagram
```
[User] ──< [UserSession] >── [ActivityLog]
   │
   └──< [UserProfile]
```

## Core Entities

### User
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Business Rules:**
- Email must be unique and valid
- Password must meet security requirements
- Soft delete for data retention

### [Other Entities]
[Define each entity with schema and business rules]

## Data Relationships
- **One-to-Many**: User → Orders
- **Many-to-Many**: Users ↔ Roles
- **One-to-One**: User → Profile

## Data Validation Rules
- **Required Fields**: What cannot be null
- **Format Validation**: Email, phone, dates
- **Business Rules**: Age limits, quantity limits
- **Referential Integrity**: Foreign key constraints

## Data Migration Strategy
- **Version Control**: How schema changes are managed
- **Rollback Plan**: How to undo changes safely
- **Seed Data**: Initial data for development/testing

---
*Schema evolves as data requirements become clear*
"""
    
    def _get_api_spec_template(self) -> str:
        """Get template for API specification."""
        return """# API Specification

## Base URL
- **Development**: `https://api-dev.example.com/v1`
- **Production**: `https://api.example.com/v1`

## Authentication
```http
Authorization: Bearer <jwt_token>
```

## Core Endpoints

### User Management

#### POST /auth/login
**Purpose**: Authenticate user and return JWT token

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "role": "user"
  }
}
```

**Error Responses:**
- `400`: Invalid credentials
- `429`: Too many login attempts

### [Other Endpoint Categories]

## Data Formats
- **Dates**: ISO 8601 format (2023-12-01T10:30:00Z)
- **IDs**: UUIDs for all entities
- **Pagination**: Cursor-based with limit/offset

## Error Handling
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email format",
    "details": {
      "field": "email",
      "value": "invalid-email"
    }
  }
}
```

## Rate Limiting
- **Authenticated**: 1000 requests/hour
- **Unauthenticated**: 100 requests/hour

---
*API evolves as endpoints are implemented*
"""
    
    def _get_security_template(self) -> str:
        """Get template for security specifications."""
        return """# Security Specification

## Authentication Strategy
- **Method**: JWT tokens with refresh tokens
- **Session Management**: Stateless with token expiration
- **Multi-factor Authentication**: TOTP for sensitive operations

## Authorization Framework
- **Role-Based Access Control (RBAC)**
  - Admin: Full system access
  - User: Limited to own data
  - Guest: Read-only public data

## Data Protection
- **Encryption in Transit**: TLS 1.3 for all connections
- **Encryption at Rest**: AES-256 for sensitive data
- **Key Management**: Separate key service, rotation policy

## Input Validation
- **SQL Injection**: Parameterized queries only
- **XSS Prevention**: Content Security Policy, input sanitization
- **CSRF Protection**: Token-based validation

## Security Headers
```http
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
```

## Audit & Logging
- **User Actions**: Login, data changes, permission changes
- **System Events**: Errors, performance issues, security events
- **Log Retention**: 90 days for compliance

## Vulnerability Management
- **Dependency Scanning**: Automated security scans
- **Penetration Testing**: Quarterly assessments
- **Security Updates**: Patch management process

---
*Security measures implemented progressively*
"""
    
    def _get_deployment_template(self) -> str:
        """Get template for deployment specifications."""
        return """# Deployment Specification

## Environment Strategy
- **Development**: Local development, hot reloading
- **Staging**: Production-like for testing
- **Production**: Live system with monitoring

## Infrastructure Requirements
- **Compute**: CPU, memory, storage needs
- **Database**: Connection limits, backup strategy
- **Networking**: Load balancer, CDN, DNS
- **Monitoring**: Health checks, metrics, alerting

## CI/CD Pipeline
1. **Code Commit** → Git repository
2. **Automated Tests** → Unit, integration, e2e
3. **Build** → Docker images, static assets
4. **Deploy to Staging** → Automated deployment
5. **Smoke Tests** → Basic functionality verification
6. **Deploy to Production** → Blue-green deployment

## Rollback Strategy
- **Database Migrations**: Reversible changes only
- **Application Code**: Previous version available
- **Static Assets**: CDN cache invalidation
- **Recovery Time**: Target 5 minutes for rollback

## Monitoring & Alerting
- **Application Metrics**: Response time, error rate, throughput
- **Infrastructure Metrics**: CPU, memory, disk, network
- **Business Metrics**: User activity, conversion rates
- **Alert Thresholds**: When to notify on-call team

---
*Deployment process refined through iterations*
"""
    
    def _get_business_rules_template(self) -> str:
        """Get template for business rules and domain logic."""
        return """# Business Rules & Domain Logic

## Core Business Rules

### User Account Management
- **Rule 1**: Email addresses must be unique across the system
- **Rule 2**: User accounts are soft-deleted to preserve data integrity
- **Rule 3**: Password changes require current password verification

### Data Validation Rules
- **Input Validation**: All user inputs sanitized and validated
- **Business Logic Validation**: Rules specific to domain
- **Cross-field Validation**: Dependencies between form fields

### Workflow Rules
- **State Transitions**: Valid state changes for entities
- **Approval Processes**: Who can approve what actions
- **Timing Constraints**: When actions can be performed

### Integration Rules
- **External API Limits**: Rate limiting, retry logic
- **Data Synchronization**: How external data is kept current
- **Error Handling**: What to do when external systems fail

## Domain-Specific Logic

### [Domain Area 1]
- Business rules specific to this domain
- Calculations, formulas, algorithms
- Compliance requirements

### [Domain Area 2]
- Additional business logic
- Industry-specific requirements
- Regulatory compliance

## Exception Handling
- **Business Rule Violations**: How to handle and communicate
- **Data Inconsistency**: Detection and resolution
- **System Errors**: Graceful degradation strategies

---
*Business rules captured and refined through development*
"""
    
    def _load_work_order_index(self) -> Dict[str, Any]:
        """Load work order tracking index."""
        index_file = self.work_orders_dir / "index.json"
        if index_file.exists():
            return json.loads(index_file.read_text())
        return {
            "next_id": 1,
            "work_orders": {},
            "completion_history": []
        }
    
    def _save_work_order_index(self):
        """Save work order tracking index."""
        index_file = self.work_orders_dir / "index.json"
        index_file.write_text(json.dumps(self.work_order_index, indent=2))
    
    async def determine_needed_documents(self, overview_content: str, project_type: ProjectType) -> List[DocumentType]:
        """Determine which documents are needed based on project type."""
        needed_docs = [DocumentType.CONTEXT]  # Always need context
        
        # Core documents for all projects
        core_docs = [
            DocumentType.REQUIREMENTS,
            DocumentType.ARCHITECTURE
        ]
        needed_docs.extend(core_docs)
        
        # Project-type-specific document selection
        if project_type == ProjectType.GAME:
            # Games: Simple structure, no complex business logic
            needed_docs.append(DocumentType.USER_STORIES)
            # Skip: API_SPEC, DATA_MODELS, SECURITY, BUSINESS_RULES
            # Only add deployment if specifically mentioned
            if any(keyword in overview_content.lower() for keyword in ['deploy', 'hosting', 'distribution']):
                needed_docs.append(DocumentType.DEPLOYMENT)
        
        elif project_type == ProjectType.CLI_TOOL:
            # CLI tools: Focus on commands and configuration
            needed_docs.append(DocumentType.USER_STORIES)
            # Add deployment for distribution
            needed_docs.append(DocumentType.DEPLOYMENT)
            # Add security only if auth/permissions mentioned
            if any(keyword in overview_content.lower() for keyword in ['auth', 'permission', 'secure']):
                needed_docs.append(DocumentType.SECURITY)
        
        elif project_type == ProjectType.WEB_APP:
            # Web apps: Comprehensive documentation
            needed_docs.extend([
                DocumentType.USER_STORIES,
                DocumentType.DATA_MODELS,
                DocumentType.API_SPEC,
                DocumentType.SECURITY,
                DocumentType.DEPLOYMENT,
                DocumentType.BUSINESS_RULES
            ])
        
        elif project_type == ProjectType.API_SERVICE:
            # API services: Focus on API spec and security
            needed_docs.extend([
                DocumentType.API_SPEC,
                DocumentType.DATA_MODELS,
                DocumentType.SECURITY,
                DocumentType.DEPLOYMENT,
                DocumentType.BUSINESS_RULES
            ])
        
        elif project_type == ProjectType.MOBILE_APP:
            # Mobile apps: Similar to web apps but different deployment
            needed_docs.extend([
                DocumentType.USER_STORIES,
                DocumentType.DATA_MODELS,
                DocumentType.SECURITY,
                DocumentType.DEPLOYMENT
            ])
            # Add API spec if backend integration
            if any(keyword in overview_content.lower() for keyword in ['api', 'backend', 'server']):
                needed_docs.append(DocumentType.API_SPEC)
        
        elif project_type == ProjectType.DESKTOP_APP:
            # Desktop apps: Similar to mobile but different deployment
            needed_docs.extend([
                DocumentType.USER_STORIES,
                DocumentType.DEPLOYMENT
            ])
            # Add data models if local storage
            if any(keyword in overview_content.lower() for keyword in ['database', 'data', 'storage']):
                needed_docs.append(DocumentType.DATA_MODELS)
        
        elif project_type == ProjectType.ML_PROJECT:
            # ML projects: Focus on data and models
            needed_docs.extend([
                DocumentType.DATA_MODELS,
                DocumentType.DEPLOYMENT
            ])
            # Add API spec if serving model
            if any(keyword in overview_content.lower() for keyword in ['api', 'serve', 'endpoint']):
                needed_docs.append(DocumentType.API_SPEC)
        
        else:  # GENERIC or unknown
            # Default to web app approach
            needed_docs.extend([
                DocumentType.USER_STORIES,
                DocumentType.DATA_MODELS,
                DocumentType.SECURITY,
                DocumentType.DEPLOYMENT
            ])
        
        self.logger.info(f"Project type: {project_type}, needed documents: {[doc.value for doc in needed_docs]}")
        return needed_docs
    
    async def initialize_document(self, doc_type: DocumentType, project_type: ProjectType = ProjectType.WEB_APP, project_context: Dict[str, Any] = None) -> Path:
        """Initialize a document with AI-generated content based on project context."""
        doc_path = self.docs_dir / doc_type.value
        
        if not doc_path.exists():
            # Generate document content using AI
            document_content = await self._ai_generate_document_content(doc_type, project_type, project_context or {})
            
            if document_content:
                doc_path.write_text(document_content)
                self.logger.info(f"Initialized AI-generated {project_type} document: {doc_path}")
            else:
                # Fallback to basic template
                template_content = self._get_project_specific_template(doc_type, project_type)
                if template_content:
                    doc_path.write_text(template_content)
                    self.logger.info(f"Initialized template-based {project_type} document: {doc_path}")
                else:
                    raise ValueError(f"No template available for document type: {doc_type} with project type: {project_type}")
        
        return doc_path
    
    async def _ai_generate_document_content(self, doc_type: DocumentType, project_type: ProjectType, project_context: Dict[str, Any]) -> str:
        """Generate document content using AI based on project context."""
        
        # Build comprehensive context for AI generation
        context_summary = self._build_generation_context(project_context)
        
        # Create generation prompt
        generation_prompt = f"""
# Document Generation Task

## Document Type: {doc_type.value}
## Project Type: {project_type}

## Project Context
{context_summary}

## Task
Generate a comprehensive {doc_type.value} document specifically tailored for this {project_type} project.

## Requirements
1. **Be specific to this project** - Use the actual project details provided
2. **Match project complexity** - Simple projects get simple docs, complex projects get detailed docs
3. **Include relevant sections only** - Don't add unnecessary sections for this project type
4. **Use markdown format** - Well-structured with headers, lists, and code blocks where appropriate
5. **Be actionable** - Include specific, implementable details

## Document Purpose
- **CONTEXT.md**: Working assumptions and project context
- **REQUIREMENTS.md**: Functional and technical requirements
- **ARCHITECTURE.md**: System design and technical decisions
- **USER_STORIES.md**: User stories and acceptance criteria
- **DATA_MODELS.md**: Data structures and relationships
- **API_SPEC.md**: API endpoints and specifications
- **SECURITY.md**: Security requirements and implementation
- **DEPLOYMENT.md**: Deployment strategy and infrastructure
- **BUSINESS_RULES.md**: Business logic and domain rules

## Response Format
Provide only the markdown content for the document, no additional text or explanations.
"""
        
        # Use enhanced project manager for generation
        from agents.enhanced_project_manager import EnhancedProjectManager
        from core.models import ProjectTask, AgentType
        
        pm = EnhancedProjectManager()
        generation_task = ProjectTask(
            id="document-generation",
            title=f"Generate {doc_type.value}",
            description=generation_prompt,
            type="CREATE",
            agent_type=AgentType.ADVISOR
        )
        
        try:
            result = await pm.execute(generation_task)
            
            if result.success:
                # Extract generated content
                generated_content = result.output.get("document_content", "")
                
                if generated_content:
                    return generated_content
                else:
                    self.logger.warning(f"AI generation produced empty content for {doc_type.value}")
                    return ""
            else:
                self.logger.error(f"AI document generation failed: {result.error}")
                return ""
        
        except Exception as e:
            self.logger.error(f"AI document generation error: {e}")
            return ""
    
    def _build_generation_context(self, project_context: Dict[str, Any]) -> str:
        """Build comprehensive context for AI document generation."""
        context_parts = []
        
        # Add project overview
        overview = project_context.get("overview_content", "")
        if overview:
            context_parts.append(f"**Project Overview**:\n{overview}")
        
        # Add refined scope if available
        refined_scope = project_context.get("refined_scope", "")
        if refined_scope and refined_scope != overview:
            context_parts.append(f"**Refined Scope**:\n{refined_scope}")
        
        # Add user responses/clarifications
        user_responses = project_context.get("user_responses", {})
        if user_responses:
            context_parts.append("**User Clarifications**:")
            for key, value in user_responses.items():
                context_parts.append(f"- {key}: {value}")
        
        # Add existing documents for context
        existing_docs = project_context.get("documents", {})
        if existing_docs:
            context_parts.append("**Existing Documents**:")
            for doc_name, doc_content in existing_docs.items():
                if doc_content:
                    # Include summary of existing docs
                    summary = doc_content[:200] + "..." if len(doc_content) > 200 else doc_content
                    context_parts.append(f"- {doc_name}: {summary}")
        
        # Add completion history for context
        completion_history = project_context.get("completion_history", [])
        if completion_history:
            context_parts.append("**Implementation Progress**:")
            for completion in completion_history[-3:]:  # Last 3 completions
                context_parts.append(f"- {completion.get('title', 'Unknown')}: {completion.get('status', 'completed')}")
        
        return "\n\n".join(context_parts) if context_parts else "No additional context available"
    
    def _get_project_specific_template(self, doc_type: DocumentType, project_type: ProjectType) -> str:
        """Get project-type-specific template content."""
        # For games, use simplified templates
        if project_type == ProjectType.GAME:
            if doc_type == DocumentType.CONTEXT:
                return self._get_game_context_template()
            elif doc_type == DocumentType.REQUIREMENTS:
                return self._get_game_requirements_template()
            elif doc_type == DocumentType.ARCHITECTURE:
                return self._get_game_architecture_template()
            elif doc_type == DocumentType.USER_STORIES:
                return self._get_game_user_stories_template()
            elif doc_type == DocumentType.DEPLOYMENT:
                return self._get_game_deployment_template()
        
        # For CLI tools, use command-focused templates
        elif project_type == ProjectType.CLI_TOOL:
            if doc_type == DocumentType.CONTEXT:
                return self._get_cli_context_template()
            elif doc_type == DocumentType.REQUIREMENTS:
                return self._get_cli_requirements_template()
            elif doc_type == DocumentType.ARCHITECTURE:
                return self._get_cli_architecture_template()
            elif doc_type == DocumentType.USER_STORIES:
                return self._get_cli_user_stories_template()
            elif doc_type == DocumentType.DEPLOYMENT:
                return self._get_cli_deployment_template()
        
        # For web apps and other complex projects, use full templates
        if doc_type in self.document_templates:
            return self.document_templates[doc_type]
        
        return ""
    
    async def get_project_context(self, work_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive project context for work order execution."""
        context = {
            "documents": {},
            "work_orders": {
                "completed": [],
                "pending": [],
                "current": work_order_id
            },
            "assumptions": [],
            "technical_decisions": [],
            "completion_history": self.work_order_index.get("completion_history", [])
        }
        
        # Load all document content
        for doc_type in DocumentType:
            if doc_type.value.endswith('.md'):
                doc_path = self.docs_dir / doc_type.value
                if doc_path.exists():
                    context["documents"][doc_type.value] = doc_path.read_text()
        
        # Load work order status
        for wo_id, wo_data in self.work_order_index.get("work_orders", {}).items():
            if wo_data["status"] == WorkOrderStatus.COMPLETED:
                context["work_orders"]["completed"].append(wo_data)
            elif wo_data["status"] == WorkOrderStatus.PENDING:
                context["work_orders"]["pending"].append(wo_data)
        
        # Extract assumptions from CONTEXT.md
        context_doc = context["documents"].get("CONTEXT.md", "")
        assumptions = self._extract_assumptions(context_doc)
        context["assumptions"] = assumptions
        
        return context
    
    def _extract_assumptions(self, context_content: str) -> List[str]:
        """Extract checked assumptions from CONTEXT.md."""
        assumptions = []
        lines = context_content.split('\n')
        
        for line in lines:
            if '- [x]' in line:
                assumption = line.replace('- [x]', '').strip()
                if assumption:
                    assumptions.append(assumption)
        
        return assumptions
    
    async def create_work_order(self, title: str, description: str, 
                              dependencies: List[str] = None) -> str:
        """Create a new work order."""
        work_order_id = f"WO-{self.work_order_index['next_id']:04d}"
        self.work_order_index['next_id'] += 1
        
        work_order = {
            "id": work_order_id,
            "title": title,
            "description": description,
            "status": WorkOrderStatus.PENDING,
            "dependencies": dependencies or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "context_snapshot": await self.get_project_context(),
            "completion_data": None
        }
        
        self.work_order_index["work_orders"][work_order_id] = work_order
        self._save_work_order_index()
        
        # Save work order file
        wo_file = self.work_orders_dir / f"{work_order_id}.json"
        wo_file.write_text(json.dumps(work_order, indent=2))
        
        self.logger.info(f"Created work order: {work_order_id}")
        return work_order_id
    
    async def complete_work_order(self, work_order_id: str, 
                                completion_data: Dict[str, Any]):
        """Mark work order as complete and update knowledge base."""
        if work_order_id not in self.work_order_index["work_orders"]:
            raise ValueError(f"Work order not found: {work_order_id}")
        
        work_order = self.work_order_index["work_orders"][work_order_id]
        work_order["status"] = WorkOrderStatus.COMPLETED
        work_order["updated_at"] = datetime.now().isoformat()
        work_order["completion_data"] = completion_data
        
        # Save completion document
        completion_file = self.completions_dir / f"{work_order_id}_completion.md"
        completion_content = self._format_completion_document(work_order, completion_data)
        completion_file.write_text(completion_content)
        
        # Update completion history
        self.work_order_index["completion_history"].append({
            "work_order_id": work_order_id,
            "completed_at": datetime.now().isoformat(),
            "title": work_order["title"],
            "artifacts": completion_data.get("artifacts", []),
            "lessons_learned": completion_data.get("lessons_learned", []),
            "context_updates": completion_data.get("context_updates", [])
        })
        
        self._save_work_order_index()
        
        # Update context document with new assumptions/learnings
        await self._update_context_from_completion(completion_data)
        
        self.logger.info(f"Completed work order: {work_order_id}")
    
    def _format_completion_document(self, work_order: Dict[str, Any], 
                                  completion_data: Dict[str, Any]) -> str:
        """Format work order completion document."""
        return f"""# Work Order Completion: {work_order['id']}

## Work Order Details
- **Title**: {work_order['title']}
- **Created**: {work_order['created_at']}
- **Completed**: {work_order['updated_at']}
- **Status**: {work_order['status']}

## Description
{work_order['description']}

## Completion Results

### Artifacts Created
{chr(10).join(f"- {artifact}" for artifact in completion_data.get('artifacts', []))}

### Code Changes
{completion_data.get('code_changes', 'No code changes documented')}

### Tests Added
{completion_data.get('tests_added', 'No tests documented')}

### Documentation Updates
{completion_data.get('documentation_updates', 'No documentation updates')}

## Lessons Learned
{chr(10).join(f"- {lesson}" for lesson in completion_data.get('lessons_learned', []))}

## Context Updates
{chr(10).join(f"- {update}" for update in completion_data.get('context_updates', []))}

## Next Steps
{completion_data.get('next_steps', 'No next steps identified')}

---
*Generated automatically on work order completion*
"""
    
    async def _update_context_from_completion(self, completion_data: Dict[str, Any]):
        """Update CONTEXT.md with learnings from work order completion."""
        context_path = self.docs_dir / "CONTEXT.md"
        if not context_path.exists():
            await self.initialize_document(DocumentType.CONTEXT)
        
        context_content = context_path.read_text()
        
        # Add new assumptions if any
        new_assumptions = completion_data.get('context_updates', [])
        if new_assumptions:
            assumptions_section = "\n### Validated Assumptions from Development\n"
            for assumption in new_assumptions:
                assumptions_section += f"- [x] {assumption}\n"
            
            context_content += f"\n{assumptions_section}"
            context_path.write_text(context_content)
            self.logger.info(f"Updated CONTEXT.md with {len(new_assumptions)} new assumptions")
    
    async def get_next_work_order(self) -> Optional[str]:
        """Get the next work order ready for execution."""
        for wo_id, work_order in self.work_order_index["work_orders"].items():
            if work_order["status"] == WorkOrderStatus.PENDING:
                # Check if dependencies are completed
                dependencies_met = all(
                    self.work_order_index["work_orders"][dep_id]["status"] == WorkOrderStatus.COMPLETED
                    for dep_id in work_order["dependencies"]
                    if dep_id in self.work_order_index["work_orders"]
                )
                
                if dependencies_met:
                    return wo_id
        
        return None
    
    async def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of current knowledge base state."""
        documents = []
        for doc_type in DocumentType:
            if doc_type.value.endswith('.md'):
                doc_path = self.docs_dir / doc_type.value
                if doc_path.exists():
                    documents.append({
                        "type": doc_type.value,
                        "exists": True,
                        "size": len(doc_path.read_text()),
                        "last_modified": datetime.fromtimestamp(doc_path.stat().st_mtime).isoformat()
                    })
                else:
                    documents.append({
                        "type": doc_type.value,
                        "exists": False
                    })
        
        work_orders = self.work_order_index["work_orders"]
        wo_summary = {
            "total": len(work_orders),
            "completed": len([wo for wo in work_orders.values() if wo["status"] == WorkOrderStatus.COMPLETED]),
            "pending": len([wo for wo in work_orders.values() if wo["status"] == WorkOrderStatus.PENDING]),
            "in_progress": len([wo for wo in work_orders.values() if wo["status"] == WorkOrderStatus.IN_PROGRESS])
        }
        
        return {
            "documents": documents,
            "work_orders": wo_summary,
            "completion_history": len(self.work_order_index["completion_history"]),
            "project_root": str(self.project_root)
        }
    
    async def update_from_completion(self, completion_result: Dict[str, Any]):
        """Update knowledge base from work order completion."""
        # This method is called by the orchestrator after each work order completion
        # to update the knowledge base with new learnings
        
        # Extract any new assumptions or technical decisions
        context_updates = completion_result.get("context_updates", [])
        if context_updates:
            await self._update_context_from_completion(completion_result)
        
        # Update architecture document if technical decisions were made
        if "architecture" in completion_result.get("lessons_learned", []):
            # Could update ARCHITECTURE.md with new details
            pass
        
        self.logger.info("Knowledge base updated from work order completion")
    
    # AI-Enhanced Features (Future Automation Opportunities)
    # Templates are now only used as fallbacks - AI generates most content
    
    async def _ai_generate_test_strategy(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-driven test strategy based on project complexity and type."""
        # TODO: Implement AI-driven test strategy generation
        # - Analyze code complexity
        # - Determine appropriate testing levels
        # - Generate test plans
        pass
    
    async def _ai_code_review(self, code_content: str) -> Dict[str, Any]:
        """AI-powered code review for quality, security, and performance."""
        # TODO: Implement AI code review
        # - Analyze code for best practices
        # - Check security vulnerabilities
        # - Suggest performance improvements
        pass
    
    async def _ai_deployment_strategy(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-driven deployment strategy based on project requirements."""
        # TODO: Implement AI deployment strategy
        # - Analyze project requirements
        # - Suggest optimal deployment configuration
        # - Generate environment-specific settings
        pass
    
    async def _ai_error_recovery(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-based error recovery and self-healing suggestions."""
        # TODO: Implement AI error recovery
        # - Analyze failure patterns
        # - Suggest intelligent recovery strategies
        # - Learn from previous failures
        pass
    
    async def _ai_architecture_advisor(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """AI-driven architecture recommendations based on requirements."""
        # TODO: Implement AI architecture advisor
        # - Analyze project requirements
        # - Suggest optimal architecture patterns
        # - Recommend technology stack
        pass