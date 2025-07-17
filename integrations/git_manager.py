"""
Git Manager for smart Git operations with validation gates.

This module provides Git operations with validation gates, commit message generation,
and branch management for safe and automated version control.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, List

try:
    import git
    from git import Repo
    GIT_PYTHON_AVAILABLE = True
except ImportError:
    GIT_PYTHON_AVAILABLE = False
    git = None
    Repo = None

from core.logger import get_logger
from core.config import get_config
from core.voice_alerts import get_voice_alerts


class GitValidationError(Exception):
    """Exception raised when Git validation fails."""
    pass


class GitManager:
    """
    Smart Git operations with validation gates.
    
    Provides Git operations with built-in validation gates, commit message generation,
    and branch management for safe automated version control.
    """
    
    def __init__(self, repo_path: str):
        """
        Initialize Git manager.
        
        Args:
            repo_path: Path to Git repository
        """
        self.repo_path = Path(repo_path)
        self.logger = get_logger("git_manager")
        self.config = get_config()
        self.voice = get_voice_alerts()
        
        # Initialize Git repository
        self.repo = None
        self.git_available = False
        self._initialize_repo()
        
        # Validation gates configuration
        self.validation_gates = {
            "linting": True,
            "type_checking": True,
            "unit_tests": True,
            "security_scan": False,  # Optional by default
            "dependency_check": False  # Optional by default
        }
        
        # Commit message templates
        self.commit_templates = {
            "feat": "feat: {description}",
            "fix": "fix: {description}",
            "docs": "docs: {description}",
            "style": "style: {description}",
            "refactor": "refactor: {description}",
            "test": "test: {description}",
            "chore": "chore: {description}"
        }
    
    def _initialize_repo(self):
        """Initialize Git repository connection."""
        try:
            if not GIT_PYTHON_AVAILABLE:
                self.logger.warning("GitPython not available, falling back to subprocess")
                # Check if git is available via subprocess
                result = subprocess.run(["git", "--version"], 
                                      capture_output=True, text=True, cwd=self.repo_path)
                if result.returncode == 0:
                    self.git_available = True
                    self.logger.info("Git available via subprocess")
                else:
                    self.logger.error("Git not available")
                    return
            else:
                # Use GitPython
                self.repo = Repo(self.repo_path)
                self.git_available = True
                self.logger.info(f"Git repository initialized at {self.repo_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Git repository: {e}")
            self.git_available = False
    
    async def create_feature_branch(self, branch_name: str, from_branch: str = "main") -> Dict[str, Any]:
        """
        Create and switch to feature branch.
        
        Args:
            branch_name: Name of the feature branch
            from_branch: Base branch to create from
        
        Returns:
            Dict containing operation result
        """
        try:
            self.voice.speak(f"Creating feature branch {branch_name}")
            
            if not self.git_available:
                return {
                    "success": False,
                    "error": "Git not available"
                }
            
            # Check if branch already exists
            if self._branch_exists(branch_name):
                self.logger.warning(f"Branch {branch_name} already exists")
                
                # Switch to existing branch
                success = await self._switch_branch(branch_name)
                if success:
                    self.voice.speak_success(f"Switched to existing branch {branch_name}")
                    return {
                        "success": True,
                        "message": f"Switched to existing branch {branch_name}",
                        "branch_name": branch_name,
                        "created_new": False
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to switch to branch {branch_name}"
                    }
            
            # Ensure we're on the base branch
            await self._switch_branch(from_branch)
            
            # Pull latest changes
            await self._pull_latest()
            
            # Create new branch
            if GIT_PYTHON_AVAILABLE and self.repo:
                new_branch = self.repo.create_head(branch_name)
                new_branch.checkout()
            else:
                result = subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode != 0:
                    raise GitValidationError(f"Failed to create branch: {result.stderr}")
            
            self.logger.info(f"Created feature branch: {branch_name}")
            self.voice.speak_success(f"Feature branch {branch_name} created")
            
            return {
                "success": True,
                "message": f"Created feature branch {branch_name}",
                "branch_name": branch_name,
                "created_new": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create branch {branch_name}: {e}")
            self.voice.speak_error(f"Branch creation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def commit_with_validation(self, 
                                   message: str = None, 
                                   files: List[str] = None,
                                   skip_validation: bool = False) -> Dict[str, Any]:
        """
        Commit changes with validation gates.
        
        Args:
            message: Commit message (will be generated if not provided)
            files: List of files to commit (all changes if not provided)
            skip_validation: Skip validation gates
        
        Returns:
            Dict containing commit result
        """
        try:
            self.voice.speak("Committing changes with validation")
            
            if not self.git_available:
                return {
                    "success": False,
                    "error": "Git not available"
                }
            
            # Stage files
            staging_result = await self._stage_files(files)
            if not staging_result["success"]:
                return staging_result
            
            # Check if there are changes to commit
            if not self._has_staged_changes():
                return {
                    "success": False,
                    "error": "No changes to commit"
                }
            
            # Run validation gates
            if not skip_validation:
                validation_result = await self._run_validation_gates()
                if not validation_result["passed"]:
                    return {
                        "success": False,
                        "error": "Validation gates failed",
                        "details": validation_result["failures"]
                    }
            
            # Generate commit message if not provided
            if not message:
                message = await self._generate_commit_message()
            
            # Create commit
            commit_result = await self._create_commit(message)
            
            if commit_result["success"]:
                self.voice.speak_success("Changes committed successfully")
                return {
                    "success": True,
                    "commit_hash": commit_result["commit_hash"],
                    "message": message,
                    "files_committed": staging_result.get("files_staged", [])
                }
            else:
                return commit_result
                
        except Exception as e:
            self.logger.error(f"Commit failed: {e}")
            self.voice.speak_error(f"Commit failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def push_branch(self, branch_name: str = None, remote: str = "origin") -> Dict[str, Any]:
        """
        Push branch to remote repository.
        
        Args:
            branch_name: Name of branch to push (current branch if None)
            remote: Remote repository name
        
        Returns:
            Dict containing push result
        """
        try:
            if not branch_name:
                branch_name = self._get_current_branch()
            
            self.voice.speak(f"Pushing branch {branch_name} to {remote}")
            
            if not self.git_available:
                return {
                    "success": False,
                    "error": "Git not available"
                }
            
            # Push branch
            if GIT_PYTHON_AVAILABLE and self.repo:
                remote_ref = self.repo.remotes[remote]
                remote_ref.push(branch_name)
            else:
                result = subprocess.run(
                    ["git", "push", remote, branch_name],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode != 0:
                    raise GitValidationError(f"Failed to push branch: {result.stderr}")
            
            self.logger.info(f"Pushed branch {branch_name} to {remote}")
            self.voice.speak_success(f"Branch {branch_name} pushed successfully")
            
            return {
                "success": True,
                "message": f"Pushed branch {branch_name} to {remote}",
                "branch_name": branch_name,
                "remote": remote
            }
            
        except Exception as e:
            self.logger.error(f"Failed to push branch {branch_name}: {e}")
            self.voice.speak_error(f"Push failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_pull_request(self, 
                                title: str,
                                description: str = None,
                                target_branch: str = "main") -> Dict[str, Any]:
        """
        Create pull request (requires GitHub CLI).
        
        Args:
            title: PR title
            description: PR description
            target_branch: Target branch for PR
        
        Returns:
            Dict containing PR creation result
        """
        try:
            self.voice.speak("Creating pull request")
            
            # Check if GitHub CLI is available
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": "GitHub CLI not available"
                }
            
            # Get current branch
            current_branch = self._get_current_branch()
            
            # Create PR
            cmd = [
                "gh", "pr", "create",
                "--title", title,
                "--base", target_branch,
                "--head", current_branch
            ]
            
            if description:
                cmd.extend(["--body", description])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                pr_url = result.stdout.strip()
                self.logger.info(f"Created PR: {pr_url}")
                self.voice.speak_success("Pull request created successfully")
                
                return {
                    "success": True,
                    "message": "Pull request created successfully",
                    "pr_url": pr_url,
                    "title": title,
                    "source_branch": current_branch,
                    "target_branch": target_branch
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create PR: {result.stderr}"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to create PR: {e}")
            self.voice.speak_error(f"PR creation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _stage_files(self, files: List[str] = None) -> Dict[str, Any]:
        """Stage files for commit."""
        try:
            files_staged = []
            
            if files:
                # Stage specific files
                for file_path in files:
                    if GIT_PYTHON_AVAILABLE and self.repo:
                        self.repo.index.add([file_path])
                    else:
                        result = subprocess.run(
                            ["git", "add", file_path],
                            capture_output=True,
                            text=True,
                            cwd=self.repo_path
                        )
                        if result.returncode != 0:
                            raise GitValidationError(f"Failed to stage {file_path}: {result.stderr}")
                    
                    files_staged.append(file_path)
            else:
                # Stage all changes
                if GIT_PYTHON_AVAILABLE and self.repo:
                    self.repo.git.add('.')
                else:
                    result = subprocess.run(
                        ["git", "add", "."],
                        capture_output=True,
                        text=True,
                        cwd=self.repo_path
                    )
                    if result.returncode != 0:
                        raise GitValidationError(f"Failed to stage files: {result.stderr}")
                
                files_staged = self._get_staged_files()
            
            return {
                "success": True,
                "files_staged": files_staged
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_validation_gates(self) -> Dict[str, Any]:
        """Run validation gates before commit."""
        gates_passed = []
        failures = []
        
        # Linting check
        if self.validation_gates["linting"]:
            try:
                result = subprocess.run(
                    ["ruff", "check", "."],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode == 0:
                    gates_passed.append("linting")
                else:
                    failures.append(f"Linting failed: {result.stdout}")
            except FileNotFoundError:
                self.logger.warning("Ruff not found, skipping linting check")
            except Exception as e:
                failures.append(f"Linting check error: {e}")
        
        # Type checking
        if self.validation_gates["type_checking"]:
            try:
                result = subprocess.run(
                    ["mypy", "."],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode == 0:
                    gates_passed.append("type_checking")
                else:
                    failures.append(f"Type checking failed: {result.stdout}")
            except FileNotFoundError:
                self.logger.warning("MyPy not found, skipping type checking")
            except Exception as e:
                failures.append(f"Type checking error: {e}")
        
        # Unit tests
        if self.validation_gates["unit_tests"]:
            try:
                result = subprocess.run(
                    ["pytest", "tests/", "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode == 0:
                    gates_passed.append("unit_tests")
                else:
                    failures.append(f"Tests failed: {result.stdout}")
            except FileNotFoundError:
                self.logger.warning("Pytest not found, skipping unit tests")
            except Exception as e:
                failures.append(f"Test execution error: {e}")
        
        # Security scan (optional)
        if self.validation_gates["security_scan"]:
            try:
                result = subprocess.run(
                    ["bandit", "-r", ".", "-f", "txt"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode == 0:
                    gates_passed.append("security_scan")
                else:
                    failures.append(f"Security scan failed: {result.stdout}")
            except FileNotFoundError:
                self.logger.warning("Bandit not found, skipping security scan")
            except Exception as e:
                failures.append(f"Security scan error: {e}")
        
        return {
            "passed": len(failures) == 0,
            "gates_passed": gates_passed,
            "failures": failures
        }
    
    async def _generate_commit_message(self) -> str:
        """Generate commit message based on changes."""
        try:
            # Get diff of staged changes
            if GIT_PYTHON_AVAILABLE and self.repo:
                diff = self.repo.git.diff('--cached', '--name-only')
            else:
                result = subprocess.run(
                    ["git", "diff", "--cached", "--name-only"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                diff = result.stdout
            
            files_changed = [f.strip() for f in diff.strip().split('\n') if f.strip()]
            
            if not files_changed:
                return "chore: minor updates"
            
            # Analyze changes to determine commit type
            commit_type = self._determine_commit_type(files_changed)
            
            # Generate description
            if len(files_changed) == 1:
                file_path = Path(files_changed[0])
                description = f"update {file_path.stem}"
            else:
                description = f"update {len(files_changed)} files"
            
            # Use template
            message = self.commit_templates[commit_type].format(description=description)
            
            self.logger.info(f"Generated commit message: {message}")
            return message
            
        except Exception as e:
            self.logger.error(f"Failed to generate commit message: {e}")
            return "chore: update files"
    
    def _determine_commit_type(self, files_changed: List[str]) -> str:
        """Determine commit type based on changed files."""
        # Analyze file types and paths
        has_python = any(f.endswith('.py') for f in files_changed)
        has_docs = any(f.endswith(('.md', '.rst', '.txt')) for f in files_changed)
        has_tests = any('test' in f.lower() for f in files_changed)
        has_config = any(f.endswith(('.json', '.yaml', '.yml', '.toml')) for f in files_changed)
        
        # Determine commit type based on file analysis
        if has_tests:
            return "test"
        elif has_docs:
            return "docs"
        elif has_python:
            return "feat"  # Assume new feature by default
        elif has_config:
            return "chore"
        else:
            return "chore"
    
    async def _create_commit(self, message: str) -> Dict[str, Any]:
        """Create Git commit."""
        try:
            if GIT_PYTHON_AVAILABLE and self.repo:
                commit = self.repo.index.commit(message)
                commit_hash = commit.hexsha
            else:
                result = subprocess.run(
                    ["git", "commit", "-m", message],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode != 0:
                    raise GitValidationError(f"Failed to commit: {result.stderr}")
                
                # Get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                commit_hash = hash_result.stdout.strip()
            
            self.logger.info(f"Committed changes: {commit_hash[:8]}")
            
            return {
                "success": True,
                "commit_hash": commit_hash,
                "message": message
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _branch_exists(self, branch_name: str) -> bool:
        """Check if branch exists."""
        try:
            if GIT_PYTHON_AVAILABLE and self.repo:
                return branch_name in [b.name for b in self.repo.branches]
            else:
                result = subprocess.run(
                    ["git", "branch", "--list", branch_name],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                return bool(result.stdout.strip())
        except Exception:
            return False
    
    async def _switch_branch(self, branch_name: str) -> bool:
        """Switch to branch."""
        try:
            if GIT_PYTHON_AVAILABLE and self.repo:
                self.repo.git.checkout(branch_name)
            else:
                result = subprocess.run(
                    ["git", "checkout", branch_name],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode != 0:
                    return False
            
            return True
        except Exception:
            return False
    
    async def _pull_latest(self) -> bool:
        """Pull latest changes."""
        try:
            if GIT_PYTHON_AVAILABLE and self.repo:
                origin = self.repo.remotes.origin
                origin.pull()
            else:
                result = subprocess.run(
                    ["git", "pull"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                if result.returncode != 0:
                    return False
            
            return True
        except Exception:
            return False
    
    def _has_staged_changes(self) -> bool:
        """Check if there are staged changes."""
        try:
            if GIT_PYTHON_AVAILABLE and self.repo:
                return len(self.repo.index.diff("HEAD")) > 0
            else:
                result = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                return result.returncode != 0
        except Exception:
            return False
    
    def _get_staged_files(self) -> List[str]:
        """Get list of staged files."""
        try:
            if GIT_PYTHON_AVAILABLE and self.repo:
                return [item.a_path for item in self.repo.index.diff("HEAD")]
            else:
                result = subprocess.run(
                    ["git", "diff", "--cached", "--name-only"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except Exception:
            return []
    
    def _get_current_branch(self) -> str:
        """Get current branch name."""
        try:
            if GIT_PYTHON_AVAILABLE and self.repo:
                return self.repo.active_branch.name
            else:
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                return result.stdout.strip()
        except Exception:
            return "main"
    
    def get_status(self) -> Dict[str, Any]:
        """Get Git repository status."""
        try:
            if not self.git_available:
                return {
                    "available": False,
                    "error": "Git not available"
                }
            
            status = {
                "available": True,
                "current_branch": self._get_current_branch(),
                "has_staged_changes": self._has_staged_changes(),
                "staged_files": self._get_staged_files(),
                "validation_gates": self.validation_gates
            }
            
            # Add repository info if using GitPython
            if GIT_PYTHON_AVAILABLE and self.repo:
                status.update({
                    "is_dirty": self.repo.is_dirty(),
                    "untracked_files": self.repo.untracked_files,
                    "remotes": [remote.name for remote in self.repo.remotes]
                })
            
            return status
            
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def configure_validation_gates(self, **gates):
        """Configure validation gates."""
        for gate_name, enabled in gates.items():
            if gate_name in self.validation_gates:
                self.validation_gates[gate_name] = enabled
                self.logger.info(f"Validation gate '{gate_name}' set to {enabled}")
    
    def get_validation_gates(self) -> Dict[str, bool]:
        """Get current validation gates configuration."""
        return self.validation_gates.copy()