"""
Parser Agent for the Ultimate Agentic StarterKit.

This module implements the Parser Agent that extracts tasks and milestones from
project specifications using RAG (Retrieval-Augmented Generation) with semantic
search capabilities.
"""

import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    SentenceTransformer = None
    cosine_similarity = None
    np = None

from agents.base_agent import BaseAgent
from core.models import AgentResult, ProjectTask, TaskStatus, AgentType, create_project_task
from core.logger import get_logger


class ParserAgent(BaseAgent):
    """
    Parser Agent that extracts tasks and milestones from project specifications.
    
    Uses semantic search with sentence transformers to identify task-like patterns
    in markdown and text documents, then structures them into ProjectTask objects.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Parser Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("parser", config)
        
        self.model = None
        self.model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
        self.chunk_size = self.config.get('chunk_size', 256)  # Token limit for model
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.max_tasks_per_chunk = self.config.get('max_tasks_per_chunk', 5)
        
        # Task extraction patterns
        self.task_patterns = [
            r'^\s*[-*+]\s+(.+)$',  # Markdown list items
            r'^\s*\d+\.\s+(.+)$',   # Numbered lists
            r'^\s*Task\s+\d+:\s*(.+)$',  # Explicit task labels
            r'^\s*TODO:\s*(.+)$',   # TODO items
            r'^\s*\[ \]\s+(.+)$',   # Checkbox items
            r'^\s*-\s*\[\s*\]\s+(.+)$',  # Markdown checkboxes
        ]
        
        # Query patterns for semantic search
        self.task_queries = [
            "Task to implement",
            "Create component",
            "Build feature", 
            "Implement function",
            "Add functionality",
            "Develop module",
            "Setup system",
            "Configure environment",
            "Write tests",
            "Fix bug",
            "Update documentation",
            "Refactor code"
        ]
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if not DEPENDENCIES_AVAILABLE:
            self.logger.error("Required dependencies not available: sentence-transformers, scikit-learn, numpy")
            return
        
        try:
            self.logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer model: {str(e)}")
            self.model = None
    
    async def execute(self, task: ProjectTask) -> AgentResult:
        """
        Execute the parser agent to extract tasks from specifications.
        
        Args:
            task: The project task containing text to parse
            
        Returns:
            AgentResult: Result with extracted tasks
        """
        start_time = datetime.now()
        
        if not self._validate_task(task):
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error="Invalid task provided",
                execution_time=0.0,
                agent_id=self.agent_id,
                timestamp=start_time
            )
        
        if not self.model:
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error="Sentence transformer model not available",
                execution_time=0.0,
                agent_id=self.agent_id,
                timestamp=start_time
            )
        
        try:
            # Extract content from task
            content = task.description
            if not content.strip():
                return AgentResult(
                    success=False,
                    confidence=0.0,
                    output=None,
                    error="No content to parse",
                    execution_time=0.0,
                    agent_id=self.agent_id,
                    timestamp=start_time
                )
            
            self.logger.info(f"Parsing content of length {len(content)}")
            
            # Create semantic chunks
            chunks = self._create_chunks(content)
            self.logger.debug(f"Created {len(chunks)} chunks for processing")
            
            # Generate embeddings for chunks
            chunk_embeddings = self.model.encode(chunks)
            
            # Extract tasks using semantic search
            extracted_tasks = await self._extract_tasks_semantic(chunks, chunk_embeddings)
            
            # Extract tasks using pattern matching as fallback
            pattern_tasks = self._extract_tasks_patterns(content)
            
            # Combine and deduplicate tasks
            all_tasks = self._combine_and_deduplicate(extracted_tasks, pattern_tasks)
            
            # Calculate confidence based on extraction quality
            confidence = self._calculate_parsing_confidence(all_tasks, content)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Extracted {len(all_tasks)} tasks with confidence {confidence:.2f}")
            
            return AgentResult(
                success=True,
                confidence=confidence,
                output={
                    'tasks': all_tasks,
                    'total_tasks': len(all_tasks),
                    'chunks_processed': len(chunks),
                    'extraction_method': 'semantic_and_pattern'
                },
                execution_time=execution_time,
                agent_id=self.agent_id,
                timestamp=start_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.exception(f"Error during parsing: {str(e)}")
            
            return AgentResult(
                success=False,
                confidence=0.0,
                output=None,
                error=str(e),
                execution_time=execution_time,
                agent_id=self.agent_id,
                timestamp=start_time
            )
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Create semantic chunks from text while preserving structure.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        current_header = ""
        
        for line in lines:
            line = line.strip()
            
            # Check if line is a header
            if line.startswith('#'):
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(f"{current_header}\n{current_chunk}".strip())
                
                # Start new chunk with header
                current_header = line
                current_chunk = ""
            else:
                current_chunk += line + '\n'
                
                # Check if chunk is getting too long
                if len(current_chunk.split()) > self.chunk_size:
                    chunks.append(f"{current_header}\n{current_chunk}".strip())
                    current_chunk = ""
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(f"{current_header}\n{current_chunk}".strip())
        
        # Filter out very short chunks
        return [chunk for chunk in chunks if len(chunk.split()) > 5]
    
    async def _extract_tasks_semantic(self, chunks: List[str], chunk_embeddings: Any) -> List[Dict[str, Any]]:
        """
        Extract tasks using semantic search.
        
        Args:
            chunks: List of text chunks
            chunk_embeddings: Embeddings for the chunks
            
        Returns:
            List of extracted task dictionaries
        """
        extracted_tasks = []
        
        for query in self.task_queries:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Find high-similarity chunks
            for i, similarity in enumerate(similarities):
                if similarity > self.similarity_threshold:
                    tasks_from_chunk = self._extract_task_from_chunk(chunks[i], query, similarity)
                    extracted_tasks.extend(tasks_from_chunk)
        
        return extracted_tasks
    
    def _extract_task_from_chunk(self, chunk: str, query: str, similarity: float) -> List[Dict[str, Any]]:
        """
        Extract task information from a chunk.
        
        Args:
            chunk: Text chunk to process
            query: Original query that matched this chunk
            similarity: Similarity score
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        lines = chunk.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for task patterns
            for pattern in self.task_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    task_text = match.group(1).strip()
                    
                    # Skip very short or generic tasks
                    if len(task_text) < 10 or task_text.lower() in ['todo', 'task', 'item']:
                        continue
                    
                    # Determine task type based on content
                    task_type = self._classify_task_type(task_text)
                    
                    # Create task dictionary
                    task_dict = {
                        'title': task_text[:100],  # Limit title length
                        'description': task_text,
                        'type': task_type,
                        'agent_type': self._suggest_agent_type(task_text),
                        'confidence': similarity,
                        'source_query': query,
                        'source_chunk': chunk[:200],  # Preview of source
                        'extraction_method': 'semantic'
                    }
                    
                    tasks.append(task_dict)
                    
                    # Limit tasks per chunk
                    if len(tasks) >= self.max_tasks_per_chunk:
                        break
            
            if len(tasks) >= self.max_tasks_per_chunk:
                break
        
        return tasks
    
    def _extract_tasks_patterns(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract tasks using pattern matching as fallback.
        
        Args:
            content: Full content to search
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in self.task_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    task_text = match.group(1).strip()
                    
                    # Skip very short tasks
                    if len(task_text) < 10:
                        continue
                    
                    task_type = self._classify_task_type(task_text)
                    
                    task_dict = {
                        'title': task_text[:100],
                        'description': task_text,
                        'type': task_type,
                        'agent_type': self._suggest_agent_type(task_text),
                        'confidence': 0.6,  # Lower confidence for pattern matching
                        'source_query': 'pattern_match',
                        'source_chunk': line,
                        'extraction_method': 'pattern'
                    }
                    
                    tasks.append(task_dict)
        
        return tasks
    
    def _classify_task_type(self, task_text: str) -> str:
        """
        Classify the type of task based on its content.
        
        Args:
            task_text: The task description
            
        Returns:
            Task type string
        """
        text_lower = task_text.lower()
        
        if any(word in text_lower for word in ['create', 'build', 'develop', 'implement', 'add', 'setup']):
            return 'CREATE'
        elif any(word in text_lower for word in ['test', 'validate', 'verify', 'check']):
            return 'TEST'
        elif any(word in text_lower for word in ['update', 'modify', 'change', 'fix', 'refactor']):
            return 'MODIFY'
        elif any(word in text_lower for word in ['validate', 'review', 'audit', 'inspect']):
            return 'VALIDATE'
        else:
            return 'CREATE'  # Default
    
    def _suggest_agent_type(self, task_text: str) -> AgentType:
        """
        Suggest the appropriate agent type for a task.
        
        Args:
            task_text: The task description
            
        Returns:
            AgentType enum value
        """
        text_lower = task_text.lower()
        
        if any(word in text_lower for word in ['test', 'validate', 'verify', 'check']):
            return AgentType.TESTER
        elif any(word in text_lower for word in ['review', 'audit', 'improve', 'optimize']):
            return AgentType.ADVISOR
        elif any(word in text_lower for word in ['parse', 'extract', 'analyze', 'read']):
            return AgentType.PARSER
        else:
            return AgentType.CODER  # Default for most implementation tasks
    
    def _combine_and_deduplicate(self, semantic_tasks: List[Dict[str, Any]], 
                                pattern_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine and deduplicate tasks from different extraction methods.
        
        Args:
            semantic_tasks: Tasks from semantic search
            pattern_tasks: Tasks from pattern matching
            
        Returns:
            Combined and deduplicated task list
        """
        all_tasks = semantic_tasks + pattern_tasks
        
        # Simple deduplication based on title similarity
        unique_tasks = []
        seen_titles = set()
        
        for task in all_tasks:
            title_normalized = task['title'].lower().strip()
            
            # Check for similar titles
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(title_normalized, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tasks.append(task)
                seen_titles.add(title_normalized)
        
        return unique_tasks
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """
        Check if two titles are similar enough to be considered duplicates.
        
        Args:
            title1: First title
            title2: Second title
            threshold: Similarity threshold
            
        Returns:
            True if titles are similar
        """
        # Simple similarity check based on word overlap
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return False
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity > threshold
    
    def _calculate_parsing_confidence(self, tasks: List[Dict[str, Any]], content: str) -> float:
        """
        Calculate confidence score for the parsing result.
        
        Args:
            tasks: List of extracted tasks
            content: Original content
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not tasks:
            return 0.0
        
        # Base confidence from task count
        task_count_score = min(len(tasks) / 10, 1.0)  # Normalize to 10 tasks
        
        # Average confidence from individual tasks
        avg_task_confidence = sum(task['confidence'] for task in tasks) / len(tasks)
        
        # Diversity score (different extraction methods)
        semantic_count = sum(1 for task in tasks if task['extraction_method'] == 'semantic')
        diversity_score = 0.8 if semantic_count > 0 else 0.4
        
        # Content coverage score
        content_words = len(content.split())
        coverage_score = min(len(tasks) / (content_words / 100), 1.0)
        
        # Combine scores
        confidence = (
            task_count_score * 0.3 +
            avg_task_confidence * 0.4 +
            diversity_score * 0.2 +
            coverage_score * 0.1
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _validate_task(self, task: ProjectTask) -> bool:
        """
        Validate that the task is appropriate for the parser agent.
        
        Args:
            task: The task to validate
            
        Returns:
            True if task is valid
        """
        if not super()._validate_task(task):
            return False
        
        # Parser-specific validation
        if task.agent_type != AgentType.PARSER:
            self.logger.warning(f"Task {task.id} is not for parser agent: {task.agent_type}")
            return False
        
        if not task.description or len(task.description.strip()) < 10:
            self.logger.warning(f"Task {task.id} has insufficient content for parsing")
            return False
        
        return True