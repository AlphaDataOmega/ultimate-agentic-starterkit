"""
Test voice alerts system.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from StarterKit.core.voice_alerts import (
    VoiceAlertType,
    VoiceAlert,
    VoiceEngine,
    VoiceAlerts,
    get_voice_alerts,
    async_speak,
    speak_milestone,
    speak_agent_start,
    speak_agent_complete,
    speak_task_complete,
    test_voice_system
)


class TestVoiceAlertType:
    """Test voice alert type enumeration."""
    
    def test_voice_alert_types(self):
        """Test voice alert type values."""
        assert VoiceAlertType.INFO == "info"
        assert VoiceAlertType.SUCCESS == "success"
        assert VoiceAlertType.WARNING == "warning"
        assert VoiceAlertType.ERROR == "error"
        assert VoiceAlertType.MILESTONE == "milestone"
        assert VoiceAlertType.AGENT_START == "agent_start"
        assert VoiceAlertType.AGENT_COMPLETE == "agent_complete"
        assert VoiceAlertType.TASK_COMPLETE == "task_complete"


class TestVoiceAlert:
    """Test voice alert model."""
    
    def test_create_voice_alert(self):
        """Test creating a voice alert."""
        alert = VoiceAlert(
            message="Test message",
            alert_type=VoiceAlertType.INFO,
            priority=1,
            metadata={"key": "value"}
        )
        
        assert alert.message == "Test message"
        assert alert.alert_type == VoiceAlertType.INFO
        assert alert.priority == 1
        assert alert.metadata == {"key": "value"}
        assert alert.attempts == 0
        assert alert.max_attempts == 3
        assert alert.timestamp > 0
    
    def test_voice_alert_defaults(self):
        """Test voice alert with default values."""
        alert = VoiceAlert("Test message")
        
        assert alert.message == "Test message"
        assert alert.alert_type == VoiceAlertType.INFO
        assert alert.priority == 1
        assert alert.metadata == {}
        assert alert.attempts == 0
        assert alert.max_attempts == 3
    
    def test_voice_alert_string_representation(self):
        """Test string representation of voice alert."""
        alert = VoiceAlert("Test message", VoiceAlertType.ERROR)
        
        str_repr = str(alert)
        assert "VoiceAlert" in str_repr
        assert "error" in str_repr
        assert "Test message" in str_repr
        
        repr_str = repr(alert)
        assert repr_str == str_repr


class TestVoiceEngine:
    """Test voice engine."""
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_voice_engine_initialization_success(self, mock_pyttsx3):
        """Test successful voice engine initialization."""
        # Mock pyttsx3 engine
        mock_engine = Mock()
        mock_voices = [Mock(name="Voice 1", id="voice1"), Mock(name="Voice 2", id="voice2")]
        mock_engine.getProperty.return_value = mock_voices
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        
        assert engine.is_initialized is True
        assert engine.engine is mock_engine
        assert engine.voices == mock_voices
        assert engine.platform is not None
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', False)
    def test_voice_engine_initialization_no_pyttsx3(self):
        """Test voice engine initialization without pyttsx3."""
        engine = VoiceEngine()
        
        assert engine.is_initialized is False
        assert engine.engine is None
        assert engine.voices == []
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_voice_engine_initialization_failure(self, mock_pyttsx3):
        """Test voice engine initialization failure."""
        mock_pyttsx3.init.side_effect = Exception("Initialization failed")
        
        engine = VoiceEngine()
        
        assert engine.is_initialized is False
        assert engine.engine is None
        assert engine.last_error == "Initialization failed"
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_set_voice_properties(self, mock_pyttsx3):
        """Test setting voice properties."""
        mock_engine = Mock()
        mock_voices = [Mock(name="Voice 1", id="voice1")]
        mock_engine.getProperty.return_value = mock_voices
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        result = engine.set_voice_properties(rate=250, volume=0.8, voice_id=0)
        
        assert result is True
        mock_engine.setProperty.assert_any_call('rate', 250)
        mock_engine.setProperty.assert_any_call('volume', 0.8)
        mock_engine.setProperty.assert_any_call('voice', 'voice1')
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_set_voice_properties_bounds(self, mock_pyttsx3):
        """Test voice properties bounds checking."""
        mock_engine = Mock()
        mock_voices = [Mock(name="Voice 1", id="voice1")]
        mock_engine.getProperty.return_value = mock_voices
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        engine.set_voice_properties(rate=1000, volume=2.0, voice_id=99)
        
        # Should clamp values to valid ranges
        mock_engine.setProperty.assert_any_call('rate', 500)  # Max rate
        mock_engine.setProperty.assert_any_call('volume', 1.0)  # Max volume
        # Invalid voice_id should not set voice
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_speak_blocking(self, mock_pyttsx3):
        """Test blocking speech."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        result = engine.speak("Test message", block=True)
        
        assert result is True
        mock_engine.say.assert_called_once_with("Test message")
        mock_engine.runAndWait.assert_called_once()
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    @patch('threading.Thread')
    def test_speak_non_blocking(self, mock_thread, mock_pyttsx3):
        """Test non-blocking speech."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        result = engine.speak("Test message", block=False)
        
        assert result is True
        mock_engine.say.assert_called_once_with("Test message")
        mock_thread.assert_called_once()
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_speak_uninitialized(self, mock_pyttsx3):
        """Test speaking with uninitialized engine."""
        mock_pyttsx3.init.side_effect = Exception("Failed")
        
        engine = VoiceEngine()
        result = engine.speak("Test message")
        
        assert result is False
        assert engine.is_initialized is False
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_speak_exception(self, mock_pyttsx3):
        """Test speaking with exception."""
        mock_engine = Mock()
        mock_engine.say.side_effect = Exception("Speech failed")
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        result = engine.speak("Test message")
        
        assert result is False
        assert engine.last_error == "Speech failed"
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_clean_text(self, mock_pyttsx3):
        """Test text cleaning for TTS."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        
        # Test URL removal
        clean_text = engine._clean_text("Visit https://example.com for more info")
        assert "https://example.com" not in clean_text
        assert "URL" in clean_text
        
        # Test punctuation cleanup
        clean_text = engine._clean_text("Wait... really?? Yes!!!")
        assert "..." not in clean_text
        assert "??" not in clean_text
        assert "!!!" not in clean_text
        
        # Test technical term replacement
        clean_text = engine._clean_text("Use the API to send JSON data")
        assert "A P I" in clean_text
        assert "J S O N" in clean_text
        
        # Test length limiting
        long_text = "x" * 600
        clean_text = engine._clean_text(long_text)
        assert len(clean_text) <= 500
        assert clean_text.endswith("...")
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_stop(self, mock_pyttsx3):
        """Test stopping speech."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        engine.stop()
        
        mock_engine.stop.assert_called_once()
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_get_voice_info(self, mock_pyttsx3):
        """Test getting voice information."""
        mock_engine = Mock()
        mock_voices = [Mock(name="Voice 1", id="voice1")]
        mock_engine.getProperty.side_effect = lambda prop: {
            'voices': mock_voices,
            'rate': 200,
            'volume': 0.7
        }.get(prop, None)
        mock_pyttsx3.init.return_value = mock_engine
        
        engine = VoiceEngine()
        info = engine.get_voice_info()
        
        assert info['initialized'] is True
        assert info['voices_available'] == 1
        assert info['current_voice_name'] == "Voice 1"
        assert info['rate'] == 200
        assert info['volume'] == 0.7
    
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_get_voice_info_uninitialized(self, mock_pyttsx3):
        """Test getting voice info for uninitialized engine."""
        mock_pyttsx3.init.side_effect = Exception("Failed")
        
        engine = VoiceEngine()
        info = engine.get_voice_info()
        
        assert info['initialized'] is False
        assert info['error'] == "Failed"
        assert 'platform' in info


class TestVoiceAlerts:
    """Test main voice alerts system."""
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_voice_alerts_initialization(self, mock_voice_engine, mock_get_config):
        """Test voice alerts initialization."""
        # Mock configuration
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_config.voice.rate = 200
        mock_config.voice.volume = 0.7
        mock_config.voice.voice_id = 0
        mock_get_config.return_value = mock_config
        
        # Mock voice engine
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        assert voice_alerts.config == mock_config
        assert voice_alerts.engine == mock_engine
        assert voice_alerts.is_running is True
        assert voice_alerts.worker_thread is not None
        assert voice_alerts.max_history == 100
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_voice_alerts_disabled(self, mock_voice_engine, mock_get_config):
        """Test voice alerts with disabled configuration."""
        mock_config = Mock()
        mock_config.voice.enabled = False
        mock_get_config.return_value = mock_config
        
        voice_alerts = VoiceAlerts()
        
        assert voice_alerts.is_running is False
        assert voice_alerts.worker_thread is None
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_speak_method(self, mock_voice_engine, mock_get_config):
        """Test speak method."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        # Test speaking
        result = voice_alerts.speak("Test message", VoiceAlertType.INFO)
        assert result is True
        assert voice_alerts.alert_queue.qsize() == 1
        
        # Test speaking disabled
        result = voice_alerts.speak("Test message", speak=False)
        assert result is True
        assert voice_alerts.last_message == "Test message"
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_speak_uninitialized_engine(self, mock_voice_engine, mock_get_config):
        """Test speaking with uninitialized engine."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = False
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        result = voice_alerts.speak("Test message")
        assert result is False
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_convenience_methods(self, mock_voice_engine, mock_get_config):
        """Test convenience speaking methods."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        # Test convenience methods
        voice_alerts.speak_milestone("Milestone reached")
        voice_alerts.speak_agent_start("agent_001")
        voice_alerts.speak_agent_complete("agent_001")
        voice_alerts.speak_task_complete("test_task")
        voice_alerts.speak_error("Error occurred")
        voice_alerts.speak_warning("Warning message")
        voice_alerts.speak_success("Success message")
        
        # Should have queued multiple alerts
        assert voice_alerts.alert_queue.qsize() == 7
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_test_voice(self, mock_voice_engine, mock_get_config):
        """Test voice testing functionality."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.speak.return_value = True
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        result = voice_alerts.test_voice()
        assert result is True
        assert voice_alerts.last_message == "Voice alerts are working correctly"
        mock_engine.speak.assert_called_once()
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_get_status(self, mock_voice_engine, mock_get_config):
        """Test getting voice alerts status."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.get_voice_info.return_value = {"initialized": True}
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        voice_alerts.last_message = "Test message"
        
        status = voice_alerts.get_status()
        
        assert status['enabled'] is True
        assert status['engine_initialized'] is True
        assert status['worker_running'] is True
        assert status['last_message'] == "Test message"
        assert 'voice_info' in status
        assert 'message_history_count' in status
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_message_history(self, mock_voice_engine, mock_get_config):
        """Test message history functionality."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        # Add some history entries
        alert = VoiceAlert("Test message", VoiceAlertType.INFO)
        voice_alerts._add_to_history(alert, "Formatted message", True)
        
        # Test getting history
        history = voice_alerts.get_history(limit=5)
        assert len(history) == 1
        assert history[0]['original_message'] == "Test message"
        assert history[0]['formatted_message'] == "Formatted message"
        assert history[0]['success'] is True
        
        # Test clearing history
        voice_alerts.clear_history()
        history = voice_alerts.get_history()
        assert len(history) == 0
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_format_alert_message(self, mock_voice_engine, mock_get_config):
        """Test alert message formatting."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        # Test basic formatting
        alert = VoiceAlert("Test message", VoiceAlertType.INFO)
        formatted = voice_alerts._format_alert_message(alert)
        assert formatted == "Information: Test message"
        
        # Test formatting with metadata
        alert = VoiceAlert("started", VoiceAlertType.AGENT_START, metadata={"agent_id": "agent_001"})
        formatted = voice_alerts._format_alert_message(alert)
        assert formatted == "Agent agent_001 starting: started"
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_stop(self, mock_voice_engine, mock_get_config):
        """Test stopping voice alerts system."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        # Stop the system
        voice_alerts.stop()
        
        assert voice_alerts.is_running is False
        mock_engine.stop.assert_called_once()
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_worker_loop_processing(self, mock_voice_engine, mock_get_config):
        """Test worker loop alert processing."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.speak.return_value = True
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        # Add an alert to the queue
        alert = VoiceAlert("Test message", VoiceAlertType.INFO)
        voice_alerts.alert_queue.put(alert)
        
        # Process the alert
        voice_alerts._process_alert(alert)
        
        # Check that the alert was processed
        assert alert.attempts == 1
        assert len(voice_alerts.message_history) == 1
        mock_engine.speak.assert_called_once()
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.VoiceEngine')
    def test_worker_loop_retry(self, mock_voice_engine, mock_get_config):
        """Test worker loop retry logic."""
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.speak.return_value = False  # Simulate failure
        mock_voice_engine.return_value = mock_engine
        
        voice_alerts = VoiceAlerts()
        
        # Add an alert to the queue
        alert = VoiceAlert("Test message", VoiceAlertType.INFO)
        
        # Process the alert (should fail and retry)
        voice_alerts._process_alert(alert)
        
        # Check that retry was attempted
        assert alert.attempts == 1
        assert voice_alerts.alert_queue.qsize() == 1  # Should be requeued


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    @patch('StarterKit.core.voice_alerts.get_voice_alerts')
    def test_speak_milestone(self, mock_get_voice_alerts):
        """Test global speak_milestone function."""
        mock_voice_alerts = Mock()
        mock_get_voice_alerts.return_value = mock_voice_alerts
        
        speak_milestone("Milestone reached")
        
        mock_voice_alerts.speak_milestone.assert_called_once_with("Milestone reached")
    
    @patch('StarterKit.core.voice_alerts.get_voice_alerts')
    def test_speak_agent_start(self, mock_get_voice_alerts):
        """Test global speak_agent_start function."""
        mock_voice_alerts = Mock()
        mock_get_voice_alerts.return_value = mock_voice_alerts
        
        speak_agent_start("agent_001", "custom message")
        
        mock_voice_alerts.speak_agent_start.assert_called_once_with("agent_001", "custom message")
    
    @patch('StarterKit.core.voice_alerts.get_voice_alerts')
    def test_speak_agent_complete(self, mock_get_voice_alerts):
        """Test global speak_agent_complete function."""
        mock_voice_alerts = Mock()
        mock_get_voice_alerts.return_value = mock_voice_alerts
        
        speak_agent_complete("agent_001")
        
        mock_voice_alerts.speak_agent_complete.assert_called_once_with("agent_001", "completed successfully")
    
    @patch('StarterKit.core.voice_alerts.get_voice_alerts')
    def test_speak_task_complete(self, mock_get_voice_alerts):
        """Test global speak_task_complete function."""
        mock_voice_alerts = Mock()
        mock_get_voice_alerts.return_value = mock_voice_alerts
        
        speak_task_complete("test_task")
        
        mock_voice_alerts.speak_task_complete.assert_called_once_with("test_task")
    
    @patch('StarterKit.core.voice_alerts.get_voice_alerts')
    def test_test_voice_system(self, mock_get_voice_alerts):
        """Test global test_voice_system function."""
        mock_voice_alerts = Mock()
        mock_voice_alerts.test_voice.return_value = True
        mock_get_voice_alerts.return_value = mock_voice_alerts
        
        result = test_voice_system()
        
        assert result is True
        mock_voice_alerts.test_voice.assert_called_once()
    
    @patch('StarterKit.core.voice_alerts.get_voice_alerts')
    def test_get_voice_alerts_singleton(self, mock_get_voice_alerts):
        """Test that get_voice_alerts returns singleton instance."""
        # Reset global instance
        import StarterKit.core.voice_alerts
        StarterKit.core.voice_alerts._voice_alerts_instance = None
        
        # Remove the mock to test real function
        mock_get_voice_alerts.stop()
        
        with patch('StarterKit.core.voice_alerts.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.voice.enabled = True
            mock_get_config.return_value = mock_config
            
            with patch('StarterKit.core.voice_alerts.VoiceEngine'):
                voice_alerts1 = get_voice_alerts()
                voice_alerts2 = get_voice_alerts()
                
                assert voice_alerts1 is voice_alerts2
    
    @patch('StarterKit.core.voice_alerts.get_voice_alerts')
    async def test_async_speak(self, mock_get_voice_alerts):
        """Test async speak function."""
        mock_voice_alerts = Mock()
        mock_voice_alerts.speak.return_value = True
        mock_get_voice_alerts.return_value = mock_voice_alerts
        
        result = await async_speak("Test message", VoiceAlertType.INFO)
        
        assert result is True
        mock_voice_alerts.speak.assert_called_once_with("Test message", VoiceAlertType.INFO)


class TestVoiceAlertsIntegration:
    """Test voice alerts integration scenarios."""
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_full_integration(self, mock_pyttsx3, mock_get_config):
        """Test full integration of voice alerts system."""
        # Mock configuration
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_config.voice.rate = 200
        mock_config.voice.volume = 0.7
        mock_config.voice.voice_id = 0
        mock_get_config.return_value = mock_config
        
        # Mock pyttsx3 engine
        mock_engine = Mock()
        mock_voices = [Mock(name="Test Voice", id="voice1")]
        mock_engine.getProperty.return_value = mock_voices
        mock_engine.speak = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        # Create voice alerts system
        voice_alerts = VoiceAlerts()
        
        # Test speaking
        result = voice_alerts.speak("Test message", VoiceAlertType.INFO)
        assert result is True
        
        # Wait briefly for worker thread
        time.sleep(0.1)
        
        # Test system status
        status = voice_alerts.get_status()
        assert status['enabled'] is True
        assert status['engine_initialized'] is True
        
        # Test voice testing
        test_result = voice_alerts.test_voice()
        assert test_result is True
        
        # Clean up
        voice_alerts.stop()
    
    @patch('StarterKit.core.voice_alerts.get_config')
    @patch('StarterKit.core.voice_alerts.PYTTSX3_AVAILABLE', True)
    @patch('StarterKit.core.voice_alerts.pyttsx3')
    def test_error_handling_integration(self, mock_pyttsx3, mock_get_config):
        """Test error handling in integration scenario."""
        # Mock configuration
        mock_config = Mock()
        mock_config.voice.enabled = True
        mock_get_config.return_value = mock_config
        
        # Mock pyttsx3 to fail
        mock_pyttsx3.init.side_effect = Exception("TTS not available")
        
        # Create voice alerts system
        voice_alerts = VoiceAlerts()
        
        # System should handle initialization failure gracefully
        assert voice_alerts.engine.is_initialized is False
        
        # Speaking should fail gracefully
        result = voice_alerts.speak("Test message")
        assert result is False
        
        # Test voice should fail gracefully
        test_result = voice_alerts.test_voice()
        assert test_result is False
        
        # System should still provide status
        status = voice_alerts.get_status()
        assert status['enabled'] is True
        assert status['engine_initialized'] is False
    
    @patch('StarterKit.core.voice_alerts.get_config')
    def test_disabled_integration(self, mock_get_config):
        """Test integration with voice alerts disabled."""
        # Mock configuration with voice disabled
        mock_config = Mock()
        mock_config.voice.enabled = False
        mock_get_config.return_value = mock_config
        
        # Create voice alerts system
        voice_alerts = VoiceAlerts()
        
        # System should not start worker thread
        assert voice_alerts.is_running is False
        assert voice_alerts.worker_thread is None
        
        # Speaking should still work (but not actually speak)
        result = voice_alerts.speak("Test message")
        assert result is True
        assert voice_alerts.last_message == "Test message"
        
        # Status should reflect disabled state
        status = voice_alerts.get_status()
        assert status['enabled'] is False
        assert status['worker_running'] is False