"""
Unit tests for Git Manager.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import subprocess

from StarterKit.integrations.git_manager import GitManager, GitValidationError


class TestGitManager:
    """Test suite for Git Manager."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=temp_dir, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_dir, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_dir, check=True)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def git_manager(self, temp_repo):
        """Create Git Manager instance."""
        return GitManager(temp_repo)
    
    def test_initialization(self, git_manager, temp_repo):
        """Test Git Manager initialization."""
        assert git_manager.repo_path == Path(temp_repo)
        assert git_manager.git_available is True
        assert git_manager.validation_gates["linting"] is True
        assert git_manager.validation_gates["type_checking"] is True
        assert git_manager.validation_gates["unit_tests"] is True
    
    def test_initialization_without_git(self):
        """Test initialization without Git available."""
        with patch('StarterKit.integrations.git_manager.GIT_PYTHON_AVAILABLE', False):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 1  # Git not available
                
                git_manager = GitManager("/tmp/test")
                assert git_manager.git_available is False
    
    @pytest.mark.asyncio
    async def test_create_feature_branch_new(self, git_manager):
        """Test creating a new feature branch."""
        with patch.object(git_manager, '_branch_exists', return_value=False):
            with patch.object(git_manager, '_switch_branch', return_value=True):
                with patch.object(git_manager, '_pull_latest', return_value=True):
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value.returncode = 0
                        
                        result = await git_manager.create_feature_branch("feature/test")
                        
                        assert result["success"] is True
                        assert result["branch_name"] == "feature/test"
                        assert result["created_new"] is True
    
    @pytest.mark.asyncio
    async def test_create_feature_branch_existing(self, git_manager):
        """Test switching to existing feature branch."""
        with patch.object(git_manager, '_branch_exists', return_value=True):
            with patch.object(git_manager, '_switch_branch', return_value=True):
                
                result = await git_manager.create_feature_branch("feature/existing")
                
                assert result["success"] is True
                assert result["branch_name"] == "feature/existing"
                assert result["created_new"] is False
    
    @pytest.mark.asyncio
    async def test_create_feature_branch_failure(self, git_manager):
        """Test feature branch creation failure."""
        with patch.object(git_manager, '_branch_exists', return_value=False):
            with patch.object(git_manager, '_switch_branch', return_value=True):
                with patch.object(git_manager, '_pull_latest', return_value=True):
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value.returncode = 1
                        mock_run.return_value.stderr = "Branch creation failed"
                        
                        result = await git_manager.create_feature_branch("feature/fail")
                        
                        assert result["success"] is False
                        assert "Branch creation failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_commit_with_validation_success(self, git_manager):
        """Test successful commit with validation."""
        with patch.object(git_manager, '_stage_files', return_value={"success": True, "files_staged": ["test.py"]}):
            with patch.object(git_manager, '_has_staged_changes', return_value=True):
                with patch.object(git_manager, '_run_validation_gates', return_value={"passed": True, "failures": []}):
                    with patch.object(git_manager, '_generate_commit_message', return_value="feat: add test feature"):
                        with patch.object(git_manager, '_create_commit', return_value={"success": True, "commit_hash": "abc123"}):
                            
                            result = await git_manager.commit_with_validation("Test commit")
                            
                            assert result["success"] is True
                            assert result["commit_hash"] == "abc123"
                            assert result["message"] == "Test commit"
    
    @pytest.mark.asyncio
    async def test_commit_with_validation_failure(self, git_manager):
        """Test commit with validation failure."""
        with patch.object(git_manager, '_stage_files', return_value={"success": True, "files_staged": ["test.py"]}):
            with patch.object(git_manager, '_has_staged_changes', return_value=True):
                with patch.object(git_manager, '_run_validation_gates', return_value={"passed": False, "failures": ["Linting failed"]}):
                    
                    result = await git_manager.commit_with_validation("Test commit")
                    
                    assert result["success"] is False
                    assert result["error"] == "Validation gates failed"
                    assert "Linting failed" in result["details"]
    
    @pytest.mark.asyncio
    async def test_commit_no_changes(self, git_manager):
        """Test commit with no changes."""
        with patch.object(git_manager, '_stage_files', return_value={"success": True, "files_staged": []}):
            with patch.object(git_manager, '_has_staged_changes', return_value=False):
                
                result = await git_manager.commit_with_validation("Test commit")
                
                assert result["success"] is False
                assert result["error"] == "No changes to commit"
    
    @pytest.mark.asyncio
    async def test_commit_skip_validation(self, git_manager):
        """Test commit with validation skipped."""
        with patch.object(git_manager, '_stage_files', return_value={"success": True, "files_staged": ["test.py"]}):
            with patch.object(git_manager, '_has_staged_changes', return_value=True):
                with patch.object(git_manager, '_generate_commit_message', return_value="feat: add test feature"):
                    with patch.object(git_manager, '_create_commit', return_value={"success": True, "commit_hash": "abc123"}):
                        
                        result = await git_manager.commit_with_validation("Test commit", skip_validation=True)
                        
                        assert result["success"] is True
                        assert result["commit_hash"] == "abc123"
    
    @pytest.mark.asyncio
    async def test_push_branch_success(self, git_manager):
        """Test successful branch push."""
        with patch.object(git_manager, '_get_current_branch', return_value="feature/test"):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                
                result = await git_manager.push_branch("feature/test")
                
                assert result["success"] is True
                assert result["branch_name"] == "feature/test"
                assert result["remote"] == "origin"
    
    @pytest.mark.asyncio
    async def test_push_branch_failure(self, git_manager):
        """Test branch push failure."""
        with patch.object(git_manager, '_get_current_branch', return_value="feature/test"):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 1
                mock_run.return_value.stderr = "Push failed"
                
                result = await git_manager.push_branch("feature/test")
                
                assert result["success"] is False
                assert "Push failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_create_pull_request_success(self, git_manager):
        """Test successful pull request creation."""
        with patch.object(git_manager, '_get_current_branch', return_value="feature/test"):
            with patch('subprocess.run') as mock_run:
                # Mock gh --version check
                mock_run.side_effect = [
                    Mock(returncode=0),  # gh available
                    Mock(returncode=0, stdout="https://github.com/user/repo/pull/123")  # PR creation
                ]
                
                result = await git_manager.create_pull_request("Test PR", "Test description")
                
                assert result["success"] is True
                assert result["pr_url"] == "https://github.com/user/repo/pull/123"
                assert result["title"] == "Test PR"
    
    @pytest.mark.asyncio
    async def test_create_pull_request_no_gh(self, git_manager):
        """Test pull request creation without GitHub CLI."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1  # gh not available
            
            result = await git_manager.create_pull_request("Test PR", "Test description")
            
            assert result["success"] is False
            assert result["error"] == "GitHub CLI not available"
    
    @pytest.mark.asyncio
    async def test_run_validation_gates_success(self, git_manager):
        """Test successful validation gates."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0  # All checks pass
            
            result = await git_manager._run_validation_gates()
            
            assert result["passed"] is True
            assert len(result["failures"]) == 0
    
    @pytest.mark.asyncio
    async def test_run_validation_gates_failure(self, git_manager):
        """Test validation gates failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1  # Checks fail
            mock_run.return_value.stdout = "Linting errors found"
            
            result = await git_manager._run_validation_gates()
            
            assert result["passed"] is False
            assert len(result["failures"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_commit_message_single_file(self, git_manager):
        """Test commit message generation for single file."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "test.py"
            
            message = await git_manager._generate_commit_message()
            
            assert message == "feat: update test"
    
    @pytest.mark.asyncio
    async def test_generate_commit_message_multiple_files(self, git_manager):
        """Test commit message generation for multiple files."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "test1.py\ntest2.py\ntest3.py"
            
            message = await git_manager._generate_commit_message()
            
            assert message == "feat: update 3 files"
    
    @pytest.mark.asyncio
    async def test_generate_commit_message_no_files(self, git_manager):
        """Test commit message generation with no files."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = ""
            
            message = await git_manager._generate_commit_message()
            
            assert message == "chore: minor updates"
    
    def test_determine_commit_type_python(self, git_manager):
        """Test commit type determination for Python files."""
        files = ["test.py", "main.py"]
        commit_type = git_manager._determine_commit_type(files)
        
        assert commit_type == "feat"
    
    def test_determine_commit_type_docs(self, git_manager):
        """Test commit type determination for documentation files."""
        files = ["README.md", "docs.rst"]
        commit_type = git_manager._determine_commit_type(files)
        
        assert commit_type == "docs"
    
    def test_determine_commit_type_tests(self, git_manager):
        """Test commit type determination for test files."""
        files = ["test_main.py", "tests.py"]
        commit_type = git_manager._determine_commit_type(files)
        
        assert commit_type == "test"
    
    def test_determine_commit_type_config(self, git_manager):
        """Test commit type determination for config files."""
        files = ["config.json", "settings.yaml"]
        commit_type = git_manager._determine_commit_type(files)
        
        assert commit_type == "chore"
    
    @pytest.mark.asyncio
    async def test_stage_files_specific(self, git_manager):
        """Test staging specific files."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            result = await git_manager._stage_files(["test.py", "main.py"])
            
            assert result["success"] is True
            assert result["files_staged"] == ["test.py", "main.py"]
    
    @pytest.mark.asyncio
    async def test_stage_files_all(self, git_manager):
        """Test staging all files."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
        with patch.object(git_manager, '_get_staged_files', return_value=["test.py", "main.py"]):
            result = await git_manager._stage_files()
            
            assert result["success"] is True
            assert result["files_staged"] == ["test.py", "main.py"]
    
    @pytest.mark.asyncio
    async def test_stage_files_failure(self, git_manager):
        """Test staging files failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Staging failed"
            
            result = await git_manager._stage_files(["test.py"])
            
            assert result["success"] is False
            assert "Staging failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_create_commit_success(self, git_manager):
        """Test successful commit creation."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                Mock(returncode=0),  # git commit
                Mock(returncode=0, stdout="abc123456")  # git rev-parse
            ]
            
            result = await git_manager._create_commit("Test commit")
            
            assert result["success"] is True
            assert result["commit_hash"] == "abc123456"
            assert result["message"] == "Test commit"
    
    @pytest.mark.asyncio
    async def test_create_commit_failure(self, git_manager):
        """Test commit creation failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Commit failed"
            
            result = await git_manager._create_commit("Test commit")
            
            assert result["success"] is False
            assert "Commit failed" in result["error"]
    
    def test_branch_exists_true(self, git_manager):
        """Test branch exists check - true case."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "feature/test"
            
            exists = git_manager._branch_exists("feature/test")
            
            assert exists is True
    
    def test_branch_exists_false(self, git_manager):
        """Test branch exists check - false case."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = ""
            
            exists = git_manager._branch_exists("feature/nonexistent")
            
            assert exists is False
    
    @pytest.mark.asyncio
    async def test_switch_branch_success(self, git_manager):
        """Test successful branch switching."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            result = await git_manager._switch_branch("feature/test")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_switch_branch_failure(self, git_manager):
        """Test branch switching failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            result = await git_manager._switch_branch("feature/nonexistent")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_pull_latest_success(self, git_manager):
        """Test successful pull."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            result = await git_manager._pull_latest()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_pull_latest_failure(self, git_manager):
        """Test pull failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            result = await git_manager._pull_latest()
            
            assert result is False
    
    def test_has_staged_changes_true(self, git_manager):
        """Test has staged changes - true case."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1  # Changes exist
            
            has_changes = git_manager._has_staged_changes()
            
            assert has_changes is True
    
    def test_has_staged_changes_false(self, git_manager):
        """Test has staged changes - false case."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0  # No changes
            
            has_changes = git_manager._has_staged_changes()
            
            assert has_changes is False
    
    def test_get_staged_files(self, git_manager):
        """Test getting staged files."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "test.py\nmain.py\nconfig.json"
            
            files = git_manager._get_staged_files()
            
            assert files == ["test.py", "main.py", "config.json"]
    
    def test_get_current_branch(self, git_manager):
        """Test getting current branch."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "feature/test"
            
            branch = git_manager._get_current_branch()
            
            assert branch == "feature/test"
    
    def test_get_status(self, git_manager):
        """Test getting Git status."""
        with patch.object(git_manager, '_get_current_branch', return_value="main"):
            with patch.object(git_manager, '_has_staged_changes', return_value=True):
                with patch.object(git_manager, '_get_staged_files', return_value=["test.py"]):
                    
                    status = git_manager.get_status()
                    
                    assert status["available"] is True
                    assert status["current_branch"] == "main"
                    assert status["has_staged_changes"] is True
                    assert status["staged_files"] == ["test.py"]
    
    def test_configure_validation_gates(self, git_manager):
        """Test configuring validation gates."""
        git_manager.configure_validation_gates(
            linting=False,
            security_scan=True
        )
        
        assert git_manager.validation_gates["linting"] is False
        assert git_manager.validation_gates["security_scan"] is True
    
    def test_get_validation_gates(self, git_manager):
        """Test getting validation gates."""
        gates = git_manager.get_validation_gates()
        
        assert gates["linting"] is True
        assert gates["type_checking"] is True
        assert gates["unit_tests"] is True
        assert gates["security_scan"] is False