"""
State Management System for the Ultimate Agentic StarterKit.

This module provides persistent workflow state storage, serialization,
resumption capabilities, and comprehensive state validation.
"""

import asyncio
import json
import sqlite3
import pickle
import threading
import hashlib
import gzip
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import uuid

from core.logger import get_logger
from core.config import get_config
from core.voice_alerts import get_voice_alerts


T = TypeVar('T')


class StateStorageBackend(str, Enum):
    """State storage backend types."""
    SQLITE = "sqlite"
    FILE = "file"
    MEMORY = "memory"
    REDIS = "redis"  # Future implementation


class StateCompressionType(str, Enum):
    """State compression types."""
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"


@dataclass
class StateMetadata:
    """Metadata for workflow state."""
    state_id: str
    workflow_id: str
    project_id: str
    created_at: datetime
    updated_at: datetime
    version: int = 1
    checksum: str = ""
    size_bytes: int = 0
    compression: StateCompressionType = StateCompressionType.NONE
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'state_id': self.state_id,
            'workflow_id': self.workflow_id,
            'project_id': self.project_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'checksum': self.checksum,
            'size_bytes': self.size_bytes,
            'compression': self.compression.value,
            'tags': self.tags,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateMetadata':
        """Create from dictionary."""
        return cls(
            state_id=data['state_id'],
            workflow_id=data['workflow_id'],
            project_id=data['project_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            version=data.get('version', 1),
            checksum=data.get('checksum', ''),
            size_bytes=data.get('size_bytes', 0),
            compression=StateCompressionType(data.get('compression', 'none')),
            tags=data.get('tags', []),
            description=data.get('description', '')
        )


@dataclass
class StoredState:
    """Container for stored workflow state."""
    metadata: StateMetadata
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metadata': self.metadata.to_dict(),
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredState':
        """Create from dictionary."""
        return cls(
            metadata=StateMetadata.from_dict(data['metadata']),
            data=data['data']
        )


class StateValidator:
    """Validator for workflow state data."""
    
    def __init__(self):
        self.logger = get_logger("state_validator")
        self.required_fields = [
            'project_spec',
            'workflow_status',
            'completed_tasks',
            'failed_tasks',
            'agent_results'
        ]
    
    def validate_state(self, state_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate workflow state data.
        
        Args:
            state_data: State data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in state_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate project_spec
        if 'project_spec' in state_data:
            project_spec = state_data['project_spec']
            if not isinstance(project_spec, dict):
                errors.append("project_spec must be a dictionary")
            elif 'tasks' not in project_spec:
                errors.append("project_spec must contain 'tasks' field")
        
        # Validate workflow_status
        if 'workflow_status' in state_data:
            status = state_data['workflow_status']
            valid_statuses = ['initializing', 'running', 'paused', 'completed', 'failed', 'cancelled']
            if status not in valid_statuses:
                errors.append(f"Invalid workflow_status: {status}")
        
        # Validate task lists
        for field in ['completed_tasks', 'failed_tasks']:
            if field in state_data:
                if not isinstance(state_data[field], list):
                    errors.append(f"{field} must be a list")
        
        # Validate agent_results
        if 'agent_results' in state_data:
            agent_results = state_data['agent_results']
            if not isinstance(agent_results, list):
                errors.append("agent_results must be a list")
            else:
                for i, result in enumerate(agent_results):
                    if not isinstance(result, dict):
                        errors.append(f"agent_results[{i}] must be a dictionary")
                    elif 'task_id' not in result:
                        errors.append(f"agent_results[{i}] missing task_id")
        
        # Validate execution_plan if present
        if 'execution_plan' in state_data:
            execution_plan = state_data['execution_plan']
            if isinstance(execution_plan, dict) and 'execution_order' in execution_plan:
                execution_order = execution_plan['execution_order']
                if not isinstance(execution_order, list):
                    errors.append("execution_plan.execution_order must be a list")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.logger.warning(f"State validation failed: {errors}")
        
        return is_valid, errors
    
    def sanitize_state(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize state data by removing invalid fields and fixing issues.
        
        Args:
            state_data: State data to sanitize
            
        Returns:
            Sanitized state data
        """
        sanitized = state_data.copy()
        
        # Ensure required fields exist
        for field in self.required_fields:
            if field not in sanitized:
                if field in ['completed_tasks', 'failed_tasks', 'agent_results']:
                    sanitized[field] = []
                elif field == 'workflow_status':
                    sanitized[field] = 'initializing'
                elif field == 'project_spec':
                    sanitized[field] = {'tasks': []}
        
        # Fix data types
        for field in ['completed_tasks', 'failed_tasks', 'agent_results']:
            if field in sanitized and not isinstance(sanitized[field], list):
                sanitized[field] = []
        
        # Fix workflow status
        if 'workflow_status' in sanitized:
            status = sanitized['workflow_status']
            valid_statuses = ['initializing', 'running', 'paused', 'completed', 'failed', 'cancelled']
            if status not in valid_statuses:
                sanitized['workflow_status'] = 'initializing'
        
        return sanitized


class SQLiteStateStore:
    """SQLite-based state storage backend."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("sqlite_state_store")
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS workflow_states (
                    state_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    compression TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    description TEXT NOT NULL,
                    state_data BLOB NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_workflow_id ON workflow_states(workflow_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_project_id ON workflow_states(project_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON workflow_states(created_at)
            ''')
            
            conn.commit()
    
    def save_state(self, state: StoredState) -> bool:
        """
        Save state to SQLite database.
        
        Args:
            state: State to save
            
        Returns:
            True if successful
        """
        with self.lock:
            try:
                # Serialize state data
                state_data = json.dumps(state.data, default=str).encode('utf-8')
                
                # Compress if needed
                if state.metadata.compression == StateCompressionType.GZIP:
                    state_data = gzip.compress(state_data)
                
                # Calculate checksum
                checksum = hashlib.md5(state_data).hexdigest()
                state.metadata.checksum = checksum
                state.metadata.size_bytes = len(state_data)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO workflow_states (
                            state_id, workflow_id, project_id, created_at, updated_at,
                            version, checksum, size_bytes, compression, tags, description,
                            state_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        state.metadata.state_id,
                        state.metadata.workflow_id,
                        state.metadata.project_id,
                        state.metadata.created_at.isoformat(),
                        state.metadata.updated_at.isoformat(),
                        state.metadata.version,
                        state.metadata.checksum,
                        state.metadata.size_bytes,
                        state.metadata.compression.value,
                        json.dumps(state.metadata.tags),
                        state.metadata.description,
                        state_data
                    ))
                    
                    conn.commit()
                
                self.logger.debug(f"Saved state {state.metadata.state_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save state: {e}")
                return False
    
    def load_state(self, state_id: str) -> Optional[StoredState]:
        """
        Load state from SQLite database.
        
        Args:
            state_id: State ID to load
            
        Returns:
            StoredState if found, None otherwise
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT state_id, workflow_id, project_id, created_at, updated_at,
                               version, checksum, size_bytes, compression, tags, description,
                               state_data
                        FROM workflow_states
                        WHERE state_id = ?
                    ''', (state_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    # Parse metadata
                    metadata = StateMetadata(
                        state_id=row[0],
                        workflow_id=row[1],
                        project_id=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        updated_at=datetime.fromisoformat(row[4]),
                        version=row[5],
                        checksum=row[6],
                        size_bytes=row[7],
                        compression=StateCompressionType(row[8]),
                        tags=json.loads(row[9]),
                        description=row[10]
                    )
                    
                    # Deserialize state data
                    state_data = row[11]
                    
                    # Decompress if needed
                    if metadata.compression == StateCompressionType.GZIP:
                        state_data = gzip.decompress(state_data)
                    
                    # Verify checksum
                    actual_checksum = hashlib.md5(state_data).hexdigest()
                    if actual_checksum != metadata.checksum:
                        self.logger.error(f"Checksum mismatch for state {state_id}")
                        return None
                    
                    # Parse JSON
                    data = json.loads(state_data.decode('utf-8'))
                    
                    return StoredState(metadata=metadata, data=data)
                    
            except Exception as e:
                self.logger.error(f"Failed to load state {state_id}: {e}")
                return None
    
    def list_states(self, workflow_id: Optional[str] = None,
                   project_id: Optional[str] = None,
                   limit: int = 100) -> List[StateMetadata]:
        """
        List states with optional filtering.
        
        Args:
            workflow_id: Optional workflow ID filter
            project_id: Optional project ID filter
            limit: Maximum number of states to return
            
        Returns:
            List of StateMetadata objects
        """
        with self.lock:
            try:
                query = '''
                    SELECT state_id, workflow_id, project_id, created_at, updated_at,
                           version, checksum, size_bytes, compression, tags, description
                    FROM workflow_states
                '''
                params = []
                
                conditions = []
                if workflow_id:
                    conditions.append('workflow_id = ?')
                    params.append(workflow_id)
                
                if project_id:
                    conditions.append('project_id = ?')
                    params.append(project_id)
                
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
                
                query += ' ORDER BY updated_at DESC LIMIT ?'
                params.append(limit)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(query, params)
                    
                    states = []
                    for row in cursor.fetchall():
                        metadata = StateMetadata(
                            state_id=row[0],
                            workflow_id=row[1],
                            project_id=row[2],
                            created_at=datetime.fromisoformat(row[3]),
                            updated_at=datetime.fromisoformat(row[4]),
                            version=row[5],
                            checksum=row[6],
                            size_bytes=row[7],
                            compression=StateCompressionType(row[8]),
                            tags=json.loads(row[9]),
                            description=row[10]
                        )
                        states.append(metadata)
                    
                    return states
                    
            except Exception as e:
                self.logger.error(f"Failed to list states: {e}")
                return []
    
    def delete_state(self, state_id: str) -> bool:
        """
        Delete a state from the database.
        
        Args:
            state_id: State ID to delete
            
        Returns:
            True if successful
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'DELETE FROM workflow_states WHERE state_id = ?',
                        (state_id,)
                    )
                    
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        self.logger.info(f"Deleted state {state_id}")
                        return True
                    else:
                        self.logger.warning(f"State {state_id} not found for deletion")
                        return False
                        
            except Exception as e:
                self.logger.error(f"Failed to delete state {state_id}: {e}")
                return False
    
    def cleanup_old_states(self, older_than: timedelta = timedelta(days=30)) -> int:
        """
        Clean up states older than specified time.
        
        Args:
            older_than: Delete states older than this
            
        Returns:
            Number of states deleted
        """
        with self.lock:
            try:
                cutoff_time = datetime.now() - older_than
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'DELETE FROM workflow_states WHERE updated_at < ?',
                        (cutoff_time.isoformat(),)
                    )
                    
                    conn.commit()
                    
                    deleted_count = cursor.rowcount
                    self.logger.info(f"Cleaned up {deleted_count} old states")
                    return deleted_count
                    
            except Exception as e:
                self.logger.error(f"Failed to cleanup old states: {e}")
                return 0


class WorkflowStateManager:
    """
    Main workflow state management system.
    
    This class provides high-level state management functionality including
    persistence, recovery, validation, and cleanup.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the state manager.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.system_config = get_config()
        self.logger = get_logger("workflow_state_manager")
        self.voice = get_voice_alerts()
        
        # Configuration
        self.backend_type = StateStorageBackend(
            self.config.get('backend', 'sqlite')
        )
        self.compression_type = StateCompressionType(
            self.config.get('compression', 'gzip')
        )
        self.auto_cleanup = self.config.get('auto_cleanup', True)
        self.cleanup_interval = self.config.get('cleanup_interval', 24)  # hours
        self.max_state_age = self.config.get('max_state_age', 30)  # days
        
        # Initialize backend
        self.backend = self._create_backend()
        
        # Initialize validator
        self.validator = StateValidator()
        
        # State cache
        self.state_cache: Dict[str, StoredState] = {}
        self.cache_lock = threading.Lock()
        
        # Cleanup timer
        self.cleanup_timer: Optional[threading.Timer] = None
        if self.auto_cleanup:
            self._schedule_cleanup()
        
        self.logger.info("Workflow State Manager initialized")
    
    def _create_backend(self) -> Union[SQLiteStateStore]:
        """Create storage backend based on configuration."""
        if self.backend_type == StateStorageBackend.SQLITE:
            db_path = self.config.get('db_path', 'workflow_states.db')
            return SQLiteStateStore(db_path)
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")
    
    def _schedule_cleanup(self):
        """Schedule automatic cleanup of old states."""
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        # Schedule next cleanup
        self.cleanup_timer = threading.Timer(
            self.cleanup_interval * 3600,  # Convert hours to seconds
            self._perform_cleanup
        )
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()
    
    def _perform_cleanup(self):
        """Perform automatic cleanup of old states."""
        try:
            older_than = timedelta(days=self.max_state_age)
            deleted_count = self.backend.cleanup_old_states(older_than)
            
            if deleted_count > 0:
                self.logger.info(f"Auto-cleanup removed {deleted_count} old states")
            
            # Clear cache of deleted states
            with self.cache_lock:
                self.state_cache.clear()
                
        except Exception as e:
            self.logger.error(f"Auto-cleanup failed: {e}")
        
        # Schedule next cleanup
        if self.auto_cleanup:
            self._schedule_cleanup()
    
    def save_workflow_state(self, workflow_id: str, project_id: str,
                           state_data: Dict[str, Any],
                           description: str = "",
                           tags: List[str] = None) -> Optional[str]:
        """
        Save workflow state with validation.
        
        Args:
            workflow_id: Workflow identifier
            project_id: Project identifier
            state_data: State data to save
            description: Optional description
            tags: Optional tags
            
        Returns:
            State ID if successful, None otherwise
        """
        # Validate state data
        is_valid, errors = self.validator.validate_state(state_data)
        if not is_valid:
            self.logger.error(f"State validation failed: {errors}")
            
            # Try to sanitize
            sanitized_data = self.validator.sanitize_state(state_data)
            is_valid, errors = self.validator.validate_state(sanitized_data)
            
            if not is_valid:
                self.logger.error(f"State sanitization failed: {errors}")
                return None
            
            state_data = sanitized_data
            self.logger.warning("State was sanitized before saving")
        
        # Create state metadata
        state_id = str(uuid.uuid4())
        metadata = StateMetadata(
            state_id=state_id,
            workflow_id=workflow_id,
            project_id=project_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            compression=self.compression_type,
            tags=tags or [],
            description=description
        )
        
        # Create stored state
        stored_state = StoredState(metadata=metadata, data=state_data)
        
        # Save to backend
        if self.backend.save_state(stored_state):
            # Cache the state
            with self.cache_lock:
                self.state_cache[state_id] = stored_state
            
            self.logger.info(f"Saved workflow state {state_id}")
            return state_id
        else:
            self.logger.error(f"Failed to save workflow state")
            return None
    
    def load_workflow_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """
        Load workflow state by ID.
        
        Args:
            state_id: State ID to load
            
        Returns:
            State data if found, None otherwise
        """
        # Check cache first
        with self.cache_lock:
            if state_id in self.state_cache:
                cached_state = self.state_cache[state_id]
                self.logger.debug(f"Loaded state {state_id} from cache")
                return cached_state.data
        
        # Load from backend
        stored_state = self.backend.load_state(state_id)
        if stored_state:
            # Validate loaded state
            is_valid, errors = self.validator.validate_state(stored_state.data)
            if not is_valid:
                self.logger.error(f"Loaded state validation failed: {errors}")
                return None
            
            # Cache the state
            with self.cache_lock:
                self.state_cache[state_id] = stored_state
            
            self.logger.info(f"Loaded workflow state {state_id}")
            return stored_state.data
        else:
            self.logger.warning(f"State {state_id} not found")
            return None
    
    def list_workflow_states(self, workflow_id: Optional[str] = None,
                           project_id: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        List workflow states with optional filtering.
        
        Args:
            workflow_id: Optional workflow ID filter
            project_id: Optional project ID filter
            limit: Maximum number of states to return
            
        Returns:
            List of state metadata dictionaries
        """
        states = self.backend.list_states(workflow_id, project_id, limit)
        return [state.to_dict() for state in states]
    
    def delete_workflow_state(self, state_id: str) -> bool:
        """
        Delete a workflow state.
        
        Args:
            state_id: State ID to delete
            
        Returns:
            True if successful
        """
        # Remove from cache
        with self.cache_lock:
            if state_id in self.state_cache:
                del self.state_cache[state_id]
        
        # Delete from backend
        success = self.backend.delete_state(state_id)
        
        if success:
            self.logger.info(f"Deleted workflow state {state_id}")
        
        return success
    
    def get_latest_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest state for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Latest state data if found, None otherwise
        """
        states = self.backend.list_states(workflow_id=workflow_id, limit=1)
        if states:
            return self.load_workflow_state(states[0].state_id)
        return None
    
    def backup_states(self, backup_path: str) -> bool:
        """
        Create a backup of all states.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful
        """
        try:
            states = self.backend.list_states(limit=10000)  # Large limit
            
            backup_data = {
                'created_at': datetime.now().isoformat(),
                'state_count': len(states),
                'states': []
            }
            
            for state_meta in states:
                stored_state = self.backend.load_state(state_meta.state_id)
                if stored_state:
                    backup_data['states'].append(stored_state.to_dict())
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            self.logger.info(f"Created backup with {len(backup_data['states'])} states")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def restore_states(self, backup_path: str) -> bool:
        """
        Restore states from a backup file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful
        """
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            restored_count = 0
            
            for state_dict in backup_data.get('states', []):
                stored_state = StoredState.from_dict(state_dict)
                
                # Update metadata
                stored_state.metadata.updated_at = datetime.now()
                
                if self.backend.save_state(stored_state):
                    restored_count += 1
            
            self.logger.info(f"Restored {restored_count} states from backup")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored states.
        
        Returns:
            Dictionary with state statistics
        """
        try:
            all_states = self.backend.list_states(limit=10000)
            
            stats = {
                'total_states': len(all_states),
                'total_size_bytes': sum(state.size_bytes for state in all_states),
                'workflows': len(set(state.workflow_id for state in all_states)),
                'projects': len(set(state.project_id for state in all_states)),
                'compression_types': {},
                'age_distribution': {
                    'today': 0,
                    'this_week': 0,
                    'this_month': 0,
                    'older': 0
                }
            }
            
            # Count compression types
            for state in all_states:
                comp_type = state.compression.value
                stats['compression_types'][comp_type] = stats['compression_types'].get(comp_type, 0) + 1
            
            # Calculate age distribution
            now = datetime.now()
            today = now.date()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            for state in all_states:
                if state.updated_at.date() == today:
                    stats['age_distribution']['today'] += 1
                elif state.updated_at >= week_ago:
                    stats['age_distribution']['this_week'] += 1
                elif state.updated_at >= month_ago:
                    stats['age_distribution']['this_month'] += 1
                else:
                    stats['age_distribution']['older'] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the state cache."""
        with self.cache_lock:
            self.state_cache.clear()
        self.logger.info("State cache cleared")
    
    def shutdown(self):
        """Shutdown the state manager."""
        self.logger.info("Shutting down state manager")
        
        # Cancel cleanup timer
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        # Clear cache
        self.clear_cache()
        
        self.logger.info("State manager shutdown complete")


# Global state manager instance
_state_manager_instance: Optional[WorkflowStateManager] = None


def get_state_manager() -> WorkflowStateManager:
    """
    Get the global state manager instance.
    
    Returns:
        WorkflowStateManager instance
    """
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = WorkflowStateManager()
    return _state_manager_instance


def save_workflow_state(workflow_id: str, project_id: str,
                       state_data: Dict[str, Any],
                       description: str = "",
                       tags: List[str] = None) -> Optional[str]:
    """
    Convenience function to save workflow state.
    
    Args:
        workflow_id: Workflow identifier
        project_id: Project identifier
        state_data: State data to save
        description: Optional description
        tags: Optional tags
        
    Returns:
        State ID if successful, None otherwise
    """
    return get_state_manager().save_workflow_state(
        workflow_id, project_id, state_data, description, tags
    )


def load_workflow_state(state_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load workflow state.
    
    Args:
        state_id: State ID to load
        
    Returns:
        State data if found, None otherwise
    """
    return get_state_manager().load_workflow_state(state_id)


def get_latest_workflow_state(workflow_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get latest workflow state.
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        Latest state data if found, None otherwise
    """
    return get_state_manager().get_latest_state(workflow_id)