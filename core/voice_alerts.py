"""
Voice alert system for the Ultimate Agentic StarterKit.

This module provides cross-platform text-to-speech notifications with proper
error handling, async support, and platform-specific optimizations.
"""

import asyncio
import platform
import threading
import queue
import time
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import logging

from .config import get_config
from .logger import get_logger

# Try to import pyttsx3 with error handling
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None


class VoiceAlertType(str, Enum):
    """Types of voice alerts."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    MILESTONE = "milestone"
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    TASK_COMPLETE = "task_complete"


class VoiceAlert:
    """
    Individual voice alert with metadata.
    """
    
    def __init__(self, message: str, alert_type: VoiceAlertType = VoiceAlertType.INFO,
                 priority: int = 1, metadata: Optional[Dict[str, Any]] = None):
        self.message = message
        self.alert_type = alert_type
        self.priority = priority
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.attempts = 0
        self.max_attempts = 3
    
    def __str__(self):
        return f"VoiceAlert({self.alert_type}: {self.message})"
    
    def __repr__(self):
        return self.__str__()


class VoiceEngine:
    """
    Voice engine wrapper with error handling and platform detection.
    """
    
    def __init__(self):
        self.engine = None
        self.voices = []
        self.current_voice_id = 0
        self.platform = platform.system().lower()
        self.logger = get_logger('starterkit.voice.engine')
        self.last_error = None
        self.is_initialized = False
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the TTS engine with error handling."""
        if not PYTTSX3_AVAILABLE:
            self.logger.debug("pyttsx3 not available - voice alerts disabled")
            self.is_initialized = False
            return
        
        try:
            # Try to initialize with minimal configuration
            self.engine = pyttsx3.init(driverName=None, debug=False)
            
            # Don't try to configure voices - just use system defaults
            self.voices = []
            
            # Mark as initialized - we'll handle errors during actual speech
            self.is_initialized = True
            self.logger.info(f"Voice engine initialized on {self.platform} with system defaults")
            
        except Exception as e:
            # If TTS fails, continue without voice alerts - this is expected on many systems
            self.logger.debug(f"Voice engine initialization failed (expected on headless systems): {str(e)}")
            self.last_error = str(e)
            self.is_initialized = False
    
    def set_voice_properties(self, rate: int = 200, volume: float = 0.7, voice_id: int = 0):
        """Set voice properties with validation."""
        if not self.is_initialized:
            return False
        
        try:
            # Set speaking rate
            self.engine.setProperty('rate', max(50, min(500, rate)))
            
            # Set volume
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
            
            # Set voice
            if self.voices and 0 <= voice_id < len(self.voices):
                self.engine.setProperty('voice', self.voices[voice_id].id)
                self.current_voice_id = voice_id
                self.logger.debug(f"Voice set to: {self.voices[voice_id].name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set voice properties: {str(e)}")
            return False
    
    def speak(self, text: str, block: bool = False) -> bool:
        """
        Speak text with error handling.
        
        Args:
            text: Text to speak
            block: Whether to block until speech is complete
        
        Returns:
            bool: True if speech was successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.debug(f"Voice not initialized - silently skipping: {text}")
            return False
        
        try:
            # Clean up text for TTS
            clean_text = self._clean_text(text)
            
            if not clean_text.strip():
                return False
            
            # Use the engine to speak
            self.engine.say(clean_text)
            
            if block:
                self.engine.runAndWait()
            else:
                # Non-blocking speech
                threading.Thread(target=self.engine.runAndWait, daemon=True).start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to speak text: {str(e)}")
            self.last_error = str(e)
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean text for TTS processing."""
        # Remove special characters that might cause issues
        import re
        
        # Remove URLs
        text = re.sub(r'https?://\S+', 'URL', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Replace common technical terms with speakable versions
        replacements = {
            'API': 'A P I',
            'JSON': 'J S O N',
            'HTTP': 'H T T P',
            'URL': 'U R L',
            'AI': 'A I',
            'ML': 'M L',
            'ID': 'I D',
        }
        
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        
        # Limit length
        if len(text) > 500:
            text = text[:497] + "..."
        
        return text
    
    def stop(self):
        """Stop current speech."""
        if self.is_initialized and self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                self.logger.error(f"Failed to stop speech: {str(e)}")
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about current voice setup."""
        if not self.is_initialized:
            return {
                'initialized': False,
                'error': self.last_error,
                'platform': self.platform
            }
        
        info = {
            'initialized': True,
            'platform': self.platform,
            'voices_available': len(self.voices),
            'current_voice_id': self.current_voice_id,
            'current_voice_name': self.voices[self.current_voice_id].name if self.voices else None
        }
        
        try:
            info['rate'] = self.engine.getProperty('rate')
            info['volume'] = self.engine.getProperty('volume')
        except Exception:
            pass
        
        return info


class VoiceAlerts:
    """
    Main voice alerts system with queue management and async support.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('starterkit.voice')
        self.engine = VoiceEngine()
        self.alert_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        self.last_message = None
        self.message_history = []
        self.max_history = 100
        
        # Alert templates
        self.alert_templates = {
            VoiceAlertType.INFO: "Information: {message}",
            VoiceAlertType.SUCCESS: "Success: {message}",
            VoiceAlertType.WARNING: "Warning: {message}",
            VoiceAlertType.ERROR: "Error: {message}",
            VoiceAlertType.MILESTONE: "Milestone reached: {message}",
            VoiceAlertType.AGENT_START: "Agent {agent_id} starting: {message}",
            VoiceAlertType.AGENT_COMPLETE: "Agent {agent_id} completed: {message}",
            VoiceAlertType.TASK_COMPLETE: "Task completed: {message}",
        }
        
        self._setup_voice_properties()
        self._start_worker()
    
    def _setup_voice_properties(self):
        """Setup voice properties from configuration."""
        if self.config.voice.enabled and self.engine.is_initialized:
            success = self.engine.set_voice_properties(
                rate=self.config.voice.rate,
                volume=self.config.voice.volume,
                voice_id=self.config.voice.voice_id
            )
            
            if success:
                self.logger.info("Voice properties configured successfully")
            else:
                self.logger.warning("Failed to configure voice properties")
    
    def _start_worker(self):
        """Start the worker thread for processing voice alerts."""
        if not self.config.voice.enabled:
            self.logger.info("Voice alerts disabled in configuration")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Voice alerts worker thread started")
    
    def _worker_loop(self):
        """Worker loop for processing voice alerts."""
        while self.is_running:
            try:
                # Get alert from queue with timeout
                alert = self.alert_queue.get(timeout=1.0)
                
                if alert is None:  # Shutdown signal
                    break
                
                self._process_alert(alert)
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in voice worker loop: {str(e)}")
    
    def _process_alert(self, alert: VoiceAlert):
        """Process a single voice alert."""
        try:
            alert.attempts += 1
            
            # Format message using template
            formatted_message = self._format_alert_message(alert)
            
            # Speak the message
            success = self.engine.speak(formatted_message, block=True)
            
            if success:
                self.last_message = formatted_message
                self._add_to_history(alert, formatted_message, True)
                self.logger.debug(f"Voice alert spoken: {formatted_message}")
            else:
                self.logger.warning(f"Failed to speak alert: {alert.message}")
                self._add_to_history(alert, formatted_message, False)
                
                # Retry if under max attempts
                if alert.attempts < alert.max_attempts:
                    self.logger.info(f"Retrying voice alert (attempt {alert.attempts + 1})")
                    self.alert_queue.put(alert)
            
        except Exception as e:
            self.logger.error(f"Error processing voice alert: {str(e)}")
            self._add_to_history(alert, alert.message, False)
    
    def _format_alert_message(self, alert: VoiceAlert) -> str:
        """Format alert message using template."""
        template = self.alert_templates.get(alert.alert_type, "{message}")
        
        try:
            return template.format(message=alert.message, **alert.metadata)
        except KeyError as e:
            self.logger.warning(f"Missing template variable {e} for alert: {alert.message}")
            return alert.message
    
    def _add_to_history(self, alert: VoiceAlert, formatted_message: str, success: bool):
        """Add alert to history with size limit."""
        history_entry = {
            'timestamp': alert.timestamp,
            'alert_type': alert.alert_type,
            'original_message': alert.message,
            'formatted_message': formatted_message,
            'success': success,
            'attempts': alert.attempts
        }
        
        self.message_history.append(history_entry)
        
        # Maintain history size limit
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
    
    def speak(self, message: str, alert_type: VoiceAlertType = VoiceAlertType.INFO,
             priority: int = 1, speak: bool = True, **kwargs) -> bool:
        """
        Queue a voice alert for speaking.
        
        Args:
            message: Message to speak
            alert_type: Type of alert
            priority: Priority level (higher = more important)
            speak: Whether to actually speak (useful for testing)
            **kwargs: Additional metadata for message formatting
        
        Returns:
            bool: True if alert was queued successfully
        """
        if not speak or not self.config.voice.enabled:
            self.last_message = message
            return True
        
        if not self.engine.is_initialized:
            self.logger.warning("Voice engine not initialized - alert skipped")
            return False
        
        try:
            alert = VoiceAlert(message, alert_type, priority, kwargs)
            self.alert_queue.put(alert)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue voice alert: {str(e)}")
            return False
    
    def speak_milestone(self, message: str, **kwargs) -> bool:
        """Speak a milestone alert."""
        return self.speak(message, VoiceAlertType.MILESTONE, priority=3, **kwargs)
    
    def speak_agent_start(self, agent_id: str, message: str = "started", **kwargs) -> bool:
        """Speak an agent start alert."""
        return self.speak(message, VoiceAlertType.AGENT_START, priority=2, 
                         agent_id=agent_id, **kwargs)
    
    def speak_agent_complete(self, agent_id: str, message: str = "completed successfully", **kwargs) -> bool:
        """Speak an agent completion alert."""
        return self.speak(message, VoiceAlertType.AGENT_COMPLETE, priority=2,
                         agent_id=agent_id, **kwargs)
    
    def speak_task_complete(self, task_name: str, **kwargs) -> bool:
        """Speak a task completion alert."""
        return self.speak(f"Task {task_name} completed", VoiceAlertType.TASK_COMPLETE, 
                         priority=2, **kwargs)
    
    def speak_error(self, message: str, **kwargs) -> bool:
        """Speak an error alert."""
        return self.speak(message, VoiceAlertType.ERROR, priority=4, **kwargs)
    
    def speak_warning(self, message: str, **kwargs) -> bool:
        """Speak a warning alert."""
        return self.speak(message, VoiceAlertType.WARNING, priority=3, **kwargs)
    
    def speak_success(self, message: str, **kwargs) -> bool:
        """Speak a success alert."""
        return self.speak(message, VoiceAlertType.SUCCESS, priority=2, **kwargs)
    
    def test_voice(self) -> bool:
        """Test voice functionality."""
        test_message = "Voice alerts are working correctly"
        
        if not self.engine.is_initialized:
            self.logger.info("Voice engine not initialized - voice alerts disabled (normal on headless systems)")
            return False
        
        success = self.engine.speak(test_message, block=True)
        
        if success:
            self.logger.info("Voice test completed successfully")
            self.last_message = test_message
        else:
            self.logger.debug("Voice test failed - continuing without voice alerts")
        
        return success
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of voice alerts system."""
        return {
            'enabled': self.config.voice.enabled,
            'engine_initialized': self.engine.is_initialized,
            'worker_running': self.is_running,
            'queue_size': self.alert_queue.qsize(),
            'last_message': self.last_message,
            'voice_info': self.engine.get_voice_info(),
            'message_history_count': len(self.message_history)
        }
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent message history."""
        return self.message_history[-limit:]
    
    def clear_history(self):
        """Clear message history."""
        self.message_history.clear()
    
    def stop(self):
        """Stop the voice alerts system."""
        self.is_running = False
        
        # Signal worker to stop
        self.alert_queue.put(None)
        
        # Wait for worker to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        # Stop engine
        self.engine.stop()
        
        self.logger.info("Voice alerts system stopped")


# Global voice alerts instance
_voice_alerts_instance: Optional[VoiceAlerts] = None


def get_voice_alerts() -> VoiceAlerts:
    """
    Get the global voice alerts instance.
    
    Returns:
        VoiceAlerts: The global voice alerts instance
    """
    global _voice_alerts_instance
    if _voice_alerts_instance is None:
        _voice_alerts_instance = VoiceAlerts()
    return _voice_alerts_instance


async def async_speak(message: str, alert_type: VoiceAlertType = VoiceAlertType.INFO, **kwargs) -> bool:
    """
    Async wrapper for speaking voice alerts.
    
    Args:
        message: Message to speak
        alert_type: Type of alert
        **kwargs: Additional metadata
    
    Returns:
        bool: True if alert was queued successfully
    """
    voice_alerts = get_voice_alerts()
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, voice_alerts.speak, message, alert_type, **kwargs)


# Convenience functions
def speak_milestone(message: str, **kwargs) -> bool:
    """Speak a milestone alert."""
    return get_voice_alerts().speak_milestone(message, **kwargs)


def speak_agent_start(agent_id: str, message: str = "started", **kwargs) -> bool:
    """Speak an agent start alert."""
    return get_voice_alerts().speak_agent_start(agent_id, message, **kwargs)


def speak_agent_complete(agent_id: str, message: str = "completed successfully", **kwargs) -> bool:
    """Speak an agent completion alert."""
    return get_voice_alerts().speak_agent_complete(agent_id, message, **kwargs)


def speak_task_complete(task_name: str, **kwargs) -> bool:
    """Speak a task completion alert."""
    return get_voice_alerts().speak_task_complete(task_name, **kwargs)


def test_voice_system() -> bool:
    """Test the voice system."""
    return get_voice_alerts().test_voice()