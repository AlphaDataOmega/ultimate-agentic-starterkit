"""
Tests for State Management System.
"""

import pytest
import asyncio
import json
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from workflows.state_manager import (
    WorkflowStateManager,
    SQLiteStateStore,
    StoredState,
    StateValidationError,
    StateStorageError,
    StateVersion,
    serialize_state,
    deserialize_state,
    validate_state_schema,
    get_state_manager,
    save_workflow_state,
    load_workflow_state,
    delete_workflow_state
)


class TestStoredState:
    """Test cases for StoredState."""
    
    def test_stored_state_initialization(self):
        """Test stored state initialization."""
        state = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"key": "value"},
            version=1,
            checksum="abc123",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert state.workflow_id == "test-workflow"
        assert state.project_id == "test-project"
        assert state.state_data == {"key": "value"}
        assert state.version == 1
        assert state.checksum == "abc123"
        assert state.created_at is not None
        assert state.updated_at is not None
    
    def test_stored_state_auto_timestamps(self):
        """Test automatic timestamp generation."""
        state = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"key": "value"}
        )
        
        assert state.created_at is not None
        assert state.updated_at is not None
        assert state.created_at <= state.updated_at
    
    def test_stored_state_to_dict(self):
        """Test stored state serialization."""
        state = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"key": "value"},
            version=1,
            checksum="abc123"
        )
        
        data = state.to_dict()
        
        assert data["workflow_id"] == "test-workflow"
        assert data["project_id"] == "test-project"
        assert data["state_data"] == {"key": "value"}
        assert data["version"] == 1
        assert data["checksum"] == "abc123"
        assert "created_at" in data
        assert "updated_at" in data


class TestStateVersion:
    """Test cases for StateVersion."""
    
    def test_state_version_initialization(self):
        """Test state version initialization."""
        version = StateVersion(
            workflow_id="test-workflow",
            version=1,
            checksum="abc123",
            created_at=datetime.now(),
            metadata={"changes": "initial"}
        )
        
        assert version.workflow_id == "test-workflow"
        assert version.version == 1
        assert version.checksum == "abc123"
        assert version.metadata == {"changes": "initial"}
    
    def test_state_version_to_dict(self):
        """Test state version serialization."""
        version = StateVersion(
            workflow_id="test-workflow",
            version=1,
            checksum="abc123",
            metadata={"changes": "initial"}
        )
        
        data = version.to_dict()
        
        assert data["workflow_id"] == "test-workflow"
        assert data["version"] == 1
        assert data["checksum"] == "abc123"
        assert data["metadata"] == {"changes": "initial"}
        assert "created_at" in data


class TestSQLiteStateStore:
    """Test cases for SQLiteStateStore."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def state_store(self, temp_db_path):
        """Create SQLite state store instance."""
        return SQLiteStateStore(temp_db_path)
    
    def test_sqlite_state_store_initialization(self, state_store):
        """Test SQLite state store initialization."""
        assert state_store.db_path is not None
        assert state_store.max_versions == 10
        assert state_store.compression_enabled is True
    
    def test_save_state_success(self, state_store):
        """Test successful state save."""
        state = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"key": "value"}
        )
        
        result = state_store.save_state(state)
        
        assert result is True
    
    def test_save_state_duplicate_version(self, state_store):
        """Test saving state with duplicate version."""
        state = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"key": "value"},
            version=1
        )
        
        # Save once
        state_store.save_state(state)
        
        # Try to save again with same version
        result = state_store.save_state(state)
        
        assert result is False  # Should fail due to duplicate version
    
    def test_load_state_success(self, state_store):
        """Test successful state load."""
        state = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"key": "value"}
        )
        
        state_store.save_state(state)
        loaded_state = state_store.load_state("test-workflow")
        
        assert loaded_state is not None
        assert loaded_state.workflow_id == "test-workflow"
        assert loaded_state.project_id == "test-project"
        assert loaded_state.state_data == {"key": "value"}
    
    def test_load_state_not_found(self, state_store):
        """Test loading non-existent state."""
        loaded_state = state_store.load_state("non-existent-workflow")
        
        assert loaded_state is None
    
    def test_load_state_specific_version(self, state_store):
        """Test loading specific version of state."""
        state_v1 = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"version": 1},
            version=1
        )
        
        state_v2 = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"version": 2},
            version=2
        )
        
        state_store.save_state(state_v1)
        state_store.save_state(state_v2)
        
        loaded_state = state_store.load_state("test-workflow", version=1)
        
        assert loaded_state is not None
        assert loaded_state.version == 1
        assert loaded_state.state_data == {"version": 1}
    
    def test_delete_state_success(self, state_store):
        """Test successful state deletion."""
        state = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"key": "value"}
        )
        
        state_store.save_state(state)
        result = state_store.delete_state("test-workflow")
        
        assert result is True
        
        # Verify state is gone
        loaded_state = state_store.load_state("test-workflow")
        assert loaded_state is None
    
    def test_delete_state_not_found(self, state_store):
        """Test deleting non-existent state."""
        result = state_store.delete_state("non-existent-workflow")
        
        assert result is False
    
    def test_list_states(self, state_store):
        """Test listing all states."""
        state1 = StoredState(
            workflow_id="workflow-1",
            project_id="project-1",
            state_data={"key": "value1"}
        )
        
        state2 = StoredState(
            workflow_id="workflow-2",
            project_id="project-2",
            state_data={"key": "value2"}
        )
        
        state_store.save_state(state1)
        state_store.save_state(state2)
        
        states = state_store.list_states()
        
        assert len(states) == 2
        workflow_ids = {state["workflow_id"] for state in states}
        assert "workflow-1" in workflow_ids
        assert "workflow-2" in workflow_ids
    
    def test_get_state_versions(self, state_store):
        """Test getting state versions."""
        state_v1 = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"version": 1},
            version=1
        )
        
        state_v2 = StoredState(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data={"version": 2},
            version=2
        )
        
        state_store.save_state(state_v1)
        state_store.save_state(state_v2)
        
        versions = state_store.get_state_versions("test-workflow")
        
        assert len(versions) == 2
        version_nums = {v["version"] for v in versions}
        assert 1 in version_nums
        assert 2 in version_nums
    
    def test_cleanup_old_versions(self, state_store):
        """Test cleanup of old versions."""
        workflow_id = "test-workflow"
        
        # Create more versions than max_versions
        for i in range(15):
            state = StoredState(
                workflow_id=workflow_id,
                project_id="test-project",
                state_data={"version": i},
                version=i
            )
            state_store.save_state(state)
        
        # Cleanup should keep only max_versions
        state_store.cleanup_old_versions(workflow_id)
        
        versions = state_store.get_state_versions(workflow_id)
        assert len(versions) <= state_store.max_versions
        
        # Should keep the latest versions
        version_nums = {v["version"] for v in versions}
        assert max(version_nums) == 14  # Latest version should be kept
    
    def test_close(self, state_store):
        """Test closing state store."""
        state_store.close()
        
        # Should be able to close multiple times without error
        state_store.close()


class TestWorkflowStateManager:
    """Test cases for WorkflowStateManager."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def mock_config(self, temp_db_path):
        """Mock configuration."""
        return {
            'state_storage_path': temp_db_path,
            'max_versions_per_workflow': 5,
            'enable_compression': True,
            'auto_cleanup_enabled': True,
            'cleanup_interval_hours': 24
        }
    
    @pytest.fixture
    def state_manager(self, mock_config):
        """Create state manager instance."""
        with patch('workflows.state_manager.get_logger'):
            return WorkflowStateManager(mock_config)
    
    @pytest.mark.asyncio
    async def test_save_state_success(self, state_manager):
        """Test successful state save."""
        state_data = {"key": "value", "progress": 0.5}
        
        result = await state_manager.save_state(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data=state_data
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_save_state_with_validation_error(self, state_manager):
        """Test state save with validation error."""
        # Invalid state data (missing required fields)
        state_data = {"invalid": "data"}
        
        with patch('workflows.state_manager.validate_state_schema', side_effect=StateValidationError("Invalid schema")):
            with pytest.raises(StateValidationError):
                await state_manager.save_state(
                    workflow_id="test-workflow",
                    project_id="test-project",
                    state_data=state_data
                )
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, state_manager):
        """Test successful state load."""
        state_data = {"key": "value", "progress": 0.5}
        
        # Save first
        await state_manager.save_state(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data=state_data
        )
        
        # Load
        loaded_state = await state_manager.load_state("test-workflow")
        
        assert loaded_state is not None
        assert loaded_state["workflow_id"] == "test-workflow"
        assert loaded_state["project_id"] == "test-project"
        assert loaded_state["state_data"] == state_data
    
    @pytest.mark.asyncio
    async def test_load_state_not_found(self, state_manager):
        """Test loading non-existent state."""
        loaded_state = await state_manager.load_state("non-existent-workflow")
        
        assert loaded_state is None
    
    @pytest.mark.asyncio
    async def test_delete_state_success(self, state_manager):
        """Test successful state deletion."""
        state_data = {"key": "value"}
        
        # Save first
        await state_manager.save_state(
            workflow_id="test-workflow",
            project_id="test-project",
            state_data=state_data
        )
        
        # Delete
        result = await state_manager.delete_state("test-workflow")
        
        assert result is True
        
        # Verify deletion
        loaded_state = await state_manager.load_state("test-workflow")
        assert loaded_state is None
    
    @pytest.mark.asyncio
    async def test_list_workflows(self, state_manager):
        """Test listing all workflows."""
        # Save multiple states
        await state_manager.save_state("workflow-1", "project-1", {"key": "value1"})
        await state_manager.save_state("workflow-2", "project-2", {"key": "value2"})
        
        workflows = await state_manager.list_workflows()
        
        assert len(workflows) == 2
        workflow_ids = {w["workflow_id"] for w in workflows}
        assert "workflow-1" in workflow_ids
        assert "workflow-2" in workflow_ids
    
    @pytest.mark.asyncio
    async def test_get_workflow_history(self, state_manager):
        """Test getting workflow history."""
        workflow_id = "test-workflow"
        
        # Save multiple versions
        await state_manager.save_state(workflow_id, "project-1", {"version": 1})
        await state_manager.save_state(workflow_id, "project-1", {"version": 2})
        await state_manager.save_state(workflow_id, "project-1", {"version": 3})
        
        history = await state_manager.get_workflow_history(workflow_id)
        
        assert len(history) == 3
        versions = {h["version"] for h in history}
        assert 1 in versions
        assert 2 in versions
        assert 3 in versions
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(self, state_manager):
        """Test creating checkpoint."""
        workflow_id = "test-workflow"
        state_data = {"key": "value", "progress": 0.5}
        
        # Save initial state
        await state_manager.save_state(workflow_id, "project-1", state_data)
        
        # Create checkpoint
        checkpoint_id = await state_manager.create_checkpoint(
            workflow_id,
            description="Test checkpoint"
        )
        
        assert checkpoint_id is not None
        assert len(checkpoint_id) > 0
    
    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, state_manager):
        """Test restoring from checkpoint."""
        workflow_id = "test-workflow"
        original_state = {"key": "value", "progress": 0.5}
        
        # Save initial state
        await state_manager.save_state(workflow_id, "project-1", original_state)
        
        # Create checkpoint
        checkpoint_id = await state_manager.create_checkpoint(workflow_id, "Test checkpoint")
        
        # Modify state
        await state_manager.save_state(workflow_id, "project-1", {"key": "modified"})
        
        # Restore from checkpoint
        result = await state_manager.restore_from_checkpoint(workflow_id, checkpoint_id)
        
        assert result is True
        
        # Verify restoration
        loaded_state = await state_manager.load_state(workflow_id)
        assert loaded_state["state_data"] == original_state
    
    @pytest.mark.asyncio
    async def test_validate_state_integrity(self, state_manager):
        """Test state integrity validation."""
        workflow_id = "test-workflow"
        state_data = {"key": "value", "progress": 0.5}
        
        # Save state
        await state_manager.save_state(workflow_id, "project-1", state_data)
        
        # Validate integrity
        result = await state_manager.validate_state_integrity(workflow_id)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cleanup_old_states(self, state_manager):
        """Test cleanup of old states."""
        workflow_id = "test-workflow"
        
        # Create multiple versions
        for i in range(10):
            await state_manager.save_state(workflow_id, "project-1", {"version": i})
        
        # Cleanup
        await state_manager.cleanup_old_states(workflow_id)
        
        # Should keep only max_versions
        history = await state_manager.get_workflow_history(workflow_id)
        assert len(history) <= state_manager.max_versions_per_workflow
    
    @pytest.mark.asyncio
    async def test_get_storage_stats(self, state_manager):
        """Test getting storage statistics."""
        # Save some states
        await state_manager.save_state("workflow-1", "project-1", {"key": "value1"})
        await state_manager.save_state("workflow-2", "project-2", {"key": "value2"})
        
        stats = await state_manager.get_storage_stats()
        
        assert "total_workflows" in stats
        assert "total_states" in stats
        assert "storage_size_bytes" in stats
        assert stats["total_workflows"] >= 2
    
    @pytest.mark.asyncio
    async def test_close(self, state_manager):
        """Test closing state manager."""
        await state_manager.close()
        
        # Should be able to close multiple times
        await state_manager.close()


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_serialize_state(self):
        """Test state serialization."""
        state_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        serialized = serialize_state(state_data)
        
        assert isinstance(serialized, (str, bytes))
        
        # Should be deserializable
        deserialized = deserialize_state(serialized)
        assert deserialized == state_data
    
    def test_serialize_state_with_datetime(self):
        """Test state serialization with datetime objects."""
        now = datetime.now()
        state_data = {"timestamp": now, "key": "value"}
        
        serialized = serialize_state(state_data)
        deserialized = deserialize_state(serialized)
        
        # Datetime should be preserved (or converted to string)
        assert "timestamp" in deserialized
        assert deserialized["key"] == "value"
    
    def test_deserialize_state_invalid_data(self):
        """Test deserializing invalid data."""
        with pytest.raises(ValueError):
            deserialize_state("invalid json data")
    
    def test_validate_state_schema_valid(self):
        """Test state schema validation with valid data."""
        state_data = {
            "workflow_id": "test-workflow",
            "project_id": "test-project",
            "current_task": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "agent_results": [],
            "overall_confidence": 0.0,
            "workflow_status": "running",
            "retry_count": 0,
            "metrics": {}
        }
        
        # Should not raise exception
        validate_state_schema(state_data)
    
    def test_validate_state_schema_invalid(self):
        """Test state schema validation with invalid data."""
        state_data = {"invalid": "schema"}
        
        with pytest.raises(StateValidationError):
            validate_state_schema(state_data)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_get_state_manager(self):
        """Test getting global state manager."""
        with patch('workflows.state_manager.WorkflowStateManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            manager = get_state_manager()
            
            assert manager is not None
            mock_manager_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_workflow_state(self):
        """Test convenience function for saving workflow state."""
        with patch('workflows.state_manager.get_state_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.save_state = AsyncMock(return_value=True)
            mock_get_manager.return_value = mock_manager
            
            result = await save_workflow_state("test-workflow", "test-project", {"key": "value"})
            
            assert result is True
            mock_manager.save_state.assert_called_once_with(
                "test-workflow", "test-project", {"key": "value"}
            )
    
    @pytest.mark.asyncio
    async def test_load_workflow_state(self):
        """Test convenience function for loading workflow state."""
        with patch('workflows.state_manager.get_state_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.load_state = AsyncMock(return_value={"key": "value"})
            mock_get_manager.return_value = mock_manager
            
            state = await load_workflow_state("test-workflow")
            
            assert state == {"key": "value"}
            mock_manager.load_state.assert_called_once_with("test-workflow", None)
    
    @pytest.mark.asyncio
    async def test_delete_workflow_state(self):
        """Test convenience function for deleting workflow state."""
        with patch('workflows.state_manager.get_state_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.delete_state = AsyncMock(return_value=True)
            mock_get_manager.return_value = mock_manager
            
            result = await delete_workflow_state("test-workflow")
            
            assert result is True
            mock_manager.delete_state.assert_called_once_with("test-workflow")