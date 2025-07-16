-- Initialize PostgreSQL database for StarterKit
-- This script runs when the database container starts for the first time

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS starterkit;

-- Connect to the starterkit database
\c starterkit;

-- Create tables for project management
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    project_type VARCHAR(50) NOT NULL CHECK (project_type IN ('web', 'blockchain', 'ai', 'general')),
    requirements JSONB DEFAULT '{}',
    validation_criteria JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for tasks
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    task_type VARCHAR(50) NOT NULL CHECK (task_type IN ('CREATE', 'MODIFY', 'TEST', 'VALIDATE')),
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'blocked')),
    confidence DECIMAL(3,2) DEFAULT 0.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    agent_type VARCHAR(50) NOT NULL CHECK (agent_type IN ('parser', 'coder', 'tester', 'advisor', 'orchestrator')),
    dependencies JSONB DEFAULT '[]',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    output_files JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for agent results
CREATE TABLE IF NOT EXISTS agent_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    success BOOLEAN NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    output JSONB,
    error_message TEXT,
    execution_time DECIMAL(10,3) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for workflow states
CREATE TABLE IF NOT EXISTS workflow_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    current_task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    completed_tasks JSONB DEFAULT '[]',
    failed_tasks JSONB DEFAULT '[]',
    overall_confidence DECIMAL(3,2) DEFAULT 0.0 CHECK (overall_confidence >= 0.0 AND overall_confidence <= 1.0),
    workflow_status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (workflow_status IN ('pending', 'in_progress', 'completed', 'failed', 'blocked')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_agent_type ON tasks(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_results_task_id ON agent_results(task_id);
CREATE INDEX IF NOT EXISTS idx_agent_results_agent_id ON agent_results(agent_id);
CREATE INDEX IF NOT EXISTS idx_workflow_states_project_id ON workflow_states(project_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updating updated_at columns
CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_states_updated_at BEFORE UPDATE ON workflow_states
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for development
INSERT INTO projects (title, description, project_type) VALUES
    ('Sample Web Project', 'A sample web application project', 'web'),
    ('Sample AI Project', 'A sample AI/ML project', 'ai')
ON CONFLICT DO NOTHING;

-- Grant permissions to starterkit user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO starterkit;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO starterkit;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO starterkit;