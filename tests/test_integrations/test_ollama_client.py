"""
Unit tests for Ollama Client.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

from StarterKit.integrations.ollama_client import OllamaClient, OllamaError, OllamaConnectionError


class TestOllamaClient:
    """Test suite for Ollama Client."""
    
    @pytest.fixture
    def ollama_client(self):
        """Create Ollama client instance."""
        return OllamaClient()
    
    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        return mock_resp
    
    def test_initialization(self, ollama_client):
        """Test Ollama client initialization."""
        assert ollama_client.base_url == "http://localhost:11434"
        assert ollama_client.timeout == 30.0
        assert ollama_client.available_models == []
        assert ollama_client.default_models["chat"] == "llama3.2:latest"
        assert ollama_client.default_models["code"] == "codellama:latest"
        assert ollama_client.default_models["tools"] == "mistral:latest"
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        client = OllamaClient(base_url="http://custom:8080", timeout=60.0)
        
        assert client.base_url == "http://custom:8080"
        assert client.timeout == 60.0
    
    def test_initialization_without_http_client(self):
        """Test initialization without HTTP client available."""
        with patch('StarterKit.integrations.ollama_client.HTTPX_AVAILABLE', False):
            with patch('StarterKit.integrations.ollama_client.REQUESTS_AVAILABLE', False):
                client = OllamaClient()
                
                assert client.http_available is False
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_client, mock_response):
        """Test successful health check."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_client.get.return_value = mock_response
            
            result = await ollama_client.health_check()
            
            assert result["healthy"] is True
            assert result["available_models"] == ["llama3.2:latest"]
            assert result["model_count"] == 1
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_client):
        """Test health check failure."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_client.get.side_effect = Exception("Connection failed")
            
            result = await ollama_client.health_check()
            
            assert result["healthy"] is False
            assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_health_check_cached(self, ollama_client):
        """Test cached health check."""
        # First call
        with patch.object(ollama_client, 'client') as mock_client:
            mock_client.get.return_value = Mock(status_code=200, json=Mock(return_value={"models": []}))
            
            result1 = await ollama_client.health_check()
            
            # Second call within cache interval
            result2 = await ollama_client.health_check()
            
            assert result1["healthy"] is True
            assert result2["healthy"] is True
            assert result2["cached"] is True
            
            # Should only call the API once
            assert mock_client.get.call_count == 1
    
    @pytest.mark.asyncio
    async def test_health_check_no_http_client(self, ollama_client):
        """Test health check without HTTP client."""
        ollama_client.http_available = False
        
        result = await ollama_client.health_check()
        
        assert result["healthy"] is False
        assert result["error"] == "HTTP client not available"
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, ollama_client, mock_response):
        """Test successful model listing."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_client.get.return_value = mock_response
            
            result = await ollama_client.list_models()
            
            assert result["success"] is True
            assert result["count"] == 1
            assert "llama3.2:latest" in result["model_names"]
    
    @pytest.mark.asyncio
    async def test_list_models_failure(self, ollama_client):
        """Test model listing failure."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal server error"
            mock_client.get.return_value = mock_resp
            
            result = await ollama_client.list_models()
            
            assert result["success"] is False
            assert result["error"] == "API error: 500"
    
    @pytest.mark.asyncio
    async def test_pull_model_success(self, ollama_client):
        """Test successful model pull."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_client.post.return_value = mock_resp
            
            with patch.object(ollama_client, 'list_models') as mock_list:
                mock_list.return_value = {"success": True}
                
                result = await ollama_client.pull_model("llama3.2:latest")
                
                assert result["success"] is True
                assert result["model_name"] == "llama3.2:latest"
    
    @pytest.mark.asyncio
    async def test_pull_model_failure(self, ollama_client):
        """Test model pull failure."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 404
            mock_resp.text = "Model not found"
            mock_client.post.return_value = mock_resp
            
            result = await ollama_client.pull_model("nonexistent:latest")
            
            assert result["success"] is False
            assert "Failed to pull model" in result["error"]
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, ollama_client):
        """Test successful response generation."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "response": "Generated response",
                "done": True,
                "total_duration": 1000,
                "eval_count": 50
            }
            mock_client.post.return_value = mock_resp
            
            ollama_client.available_models = ["llama3.2:latest"]
            
            result = await ollama_client.generate_response(
                "llama3.2:latest",
                "Test prompt"
            )
            
            assert result["success"] is True
            assert result["response"] == "Generated response"
            assert result["done"] is True
            assert result["eval_count"] == 50
    
    @pytest.mark.asyncio
    async def test_generate_response_model_not_available(self, ollama_client):
        """Test response generation with unavailable model."""
        with patch.object(ollama_client, 'pull_model') as mock_pull:
            mock_pull.return_value = {"success": False}
            
            result = await ollama_client.generate_response(
                "unavailable:latest",
                "Test prompt"
            )
            
            assert result["success"] is False
            assert "not available and pull failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_generate_response_with_auto_pull(self, ollama_client):
        """Test response generation with automatic model pull."""
        with patch.object(ollama_client, 'pull_model') as mock_pull:
            mock_pull.return_value = {"success": True}
            
            with patch.object(ollama_client, 'client') as mock_client:
                mock_resp = Mock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "response": "Generated response",
                    "done": True
                }
                mock_client.post.return_value = mock_resp
                
                result = await ollama_client.generate_response(
                    "new_model:latest",
                    "Test prompt"
                )
                
                assert result["success"] is True
                assert result["response"] == "Generated response"
                mock_pull.assert_called_once_with("new_model:latest")
    
    @pytest.mark.asyncio
    async def test_generate_response_with_system_prompt(self, ollama_client):
        """Test response generation with system prompt."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "response": "Generated response",
                "done": True
            }
            mock_client.post.return_value = mock_resp
            
            ollama_client.available_models = ["llama3.2:latest"]
            
            result = await ollama_client.generate_response(
                "llama3.2:latest",
                "Test prompt",
                system="You are a helpful assistant"
            )
            
            assert result["success"] is True
            
            # Check that system prompt was included in the request
            call_args = mock_client.post.call_args
            request_data = call_args[1]["json"]
            assert request_data["system"] == "You are a helpful assistant"
    
    @pytest.mark.asyncio
    async def test_generate_response_streaming(self, ollama_client):
        """Test streaming response generation."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_client.post.return_value = mock_resp
            
            ollama_client.available_models = ["llama3.2:latest"]
            
            result = await ollama_client.generate_response(
                "llama3.2:latest",
                "Test prompt",
                stream=True
            )
            
            assert result["success"] is True
            assert result["stream"] is True
            
            # Check that streaming was requested
            call_args = mock_client.post.call_args
            request_data = call_args[1]["json"]
            assert request_data["stream"] is True
    
    @pytest.mark.asyncio
    async def test_chat_success(self, ollama_client):
        """Test successful chat."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "message": {"role": "assistant", "content": "Chat response"},
                "done": True
            }
            mock_client.post.return_value = mock_resp
            
            ollama_client.available_models = ["llama3.2:latest"]
            
            messages = [
                {"role": "user", "content": "Hello"}
            ]
            
            result = await ollama_client.chat("llama3.2:latest", messages)
            
            assert result["success"] is True
            assert result["message"]["content"] == "Chat response"
            assert result["done"] is True
    
    @pytest.mark.asyncio
    async def test_chat_streaming(self, ollama_client):
        """Test streaming chat."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_client.post.return_value = mock_resp
            
            ollama_client.available_models = ["llama3.2:latest"]
            
            messages = [{"role": "user", "content": "Hello"}]
            
            result = await ollama_client.chat("llama3.2:latest", messages, stream=True)
            
            assert result["success"] is True
            assert result["stream"] is True
    
    @pytest.mark.asyncio
    async def test_tool_calling_success(self, ollama_client):
        """Test successful tool calling."""
        with patch.object(ollama_client, 'generate_response') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "response": "TOOL_CALL: get_weather\nPARAMETERS: {\"location\": \"New York\"}"
            }
            
            tools = [
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "properties": {
                            "location": {"type": "string", "description": "Location name"}
                        }
                    }
                }
            ]
            
            result = await ollama_client.tool_calling(
                "mistral:latest",
                "What's the weather in New York?",
                tools
            )
            
            assert result["success"] is True
            assert result["has_tool_calls"] is True
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "get_weather"
            assert result["tool_calls"][0]["parameters"]["location"] == "New York"
    
    @pytest.mark.asyncio
    async def test_tool_calling_no_tools(self, ollama_client):
        """Test tool calling with no tool calls in response."""
        with patch.object(ollama_client, 'generate_response') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "response": "I cannot help with that."
            }
            
            tools = [
                {
                    "name": "get_weather",
                    "description": "Get weather information"
                }
            ]
            
            result = await ollama_client.tool_calling(
                "mistral:latest",
                "Hello",
                tools
            )
            
            assert result["success"] is True
            assert result["has_tool_calls"] is False
            assert len(result["tool_calls"]) == 0
    
    @pytest.mark.asyncio
    async def test_tool_calling_generation_failure(self, ollama_client):
        """Test tool calling with generation failure."""
        with patch.object(ollama_client, 'generate_response') as mock_generate:
            mock_generate.return_value = {
                "success": False,
                "error": "Model not available"
            }
            
            tools = [{"name": "test_tool", "description": "Test tool"}]
            
            result = await ollama_client.tool_calling(
                "mistral:latest",
                "Test prompt",
                tools
            )
            
            assert result["success"] is False
            assert result["error"] == "Model not available"
    
    def test_format_tools_for_ollama(self, ollama_client):
        """Test tool formatting for Ollama."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "properties": {
                        "location": {"type": "string", "description": "Location name"},
                        "units": {"type": "string", "description": "Temperature units"}
                    }
                }
            }
        ]
        
        formatted = ollama_client._format_tools_for_ollama(tools, "Test prompt")
        
        assert "get_weather" in formatted
        assert "Get weather information" in formatted
        assert "location (string): Location name" in formatted
        assert "units (string): Temperature units" in formatted
        assert "Test prompt" in formatted
    
    def test_parse_tool_calls_single(self, ollama_client):
        """Test parsing single tool call."""
        response = """
        TOOL_CALL: get_weather
        PARAMETERS: {"location": "New York", "units": "celsius"}
        """
        
        tool_calls = ollama_client._parse_tool_calls(response)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["parameters"]["location"] == "New York"
        assert tool_calls[0]["parameters"]["units"] == "celsius"
    
    def test_parse_tool_calls_multiple(self, ollama_client):
        """Test parsing multiple tool calls."""
        response = """
        TOOL_CALL: get_weather
        PARAMETERS: {"location": "New York"}
        
        TOOL_CALL: get_time
        PARAMETERS: {"timezone": "EST"}
        """
        
        tool_calls = ollama_client._parse_tool_calls(response)
        
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[1]["name"] == "get_time"
    
    def test_parse_tool_calls_no_parameters(self, ollama_client):
        """Test parsing tool call without parameters."""
        response = """
        TOOL_CALL: simple_tool
        """
        
        tool_calls = ollama_client._parse_tool_calls(response)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "simple_tool"
        assert tool_calls[0]["parameters"] == {}
    
    def test_parse_tool_calls_invalid_json(self, ollama_client):
        """Test parsing tool call with invalid JSON parameters."""
        response = """
        TOOL_CALL: broken_tool
        PARAMETERS: {invalid json}
        """
        
        tool_calls = ollama_client._parse_tool_calls(response)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "broken_tool"
        assert tool_calls[0]["parameters"] == {}
    
    def test_parse_tool_calls_none(self, ollama_client):
        """Test parsing response with no tool calls."""
        response = "This is a regular response without tool calls."
        
        tool_calls = ollama_client._parse_tool_calls(response)
        
        assert len(tool_calls) == 0
    
    @pytest.mark.asyncio
    async def test_get_model_info_success(self, ollama_client):
        """Test successful model info retrieval."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "name": "llama3.2:latest",
                "size": 4000000000,
                "digest": "abc123"
            }
            mock_client.post.return_value = mock_resp
            
            result = await ollama_client.get_model_info("llama3.2:latest")
            
            assert result["success"] is True
            assert result["model_info"]["name"] == "llama3.2:latest"
            assert result["cached"] is False
    
    @pytest.mark.asyncio
    async def test_get_model_info_cached(self, ollama_client):
        """Test cached model info retrieval."""
        ollama_client.model_info_cache["llama3.2:latest"] = {
            "name": "llama3.2:latest",
            "size": 4000000000
        }
        
        result = await ollama_client.get_model_info("llama3.2:latest")
        
        assert result["success"] is True
        assert result["cached"] is True
    
    @pytest.mark.asyncio
    async def test_get_model_info_failure(self, ollama_client):
        """Test model info retrieval failure."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_resp = Mock()
            mock_resp.status_code = 404
            mock_resp.text = "Model not found"
            mock_client.post.return_value = mock_resp
            
            result = await ollama_client.get_model_info("nonexistent:latest")
            
            assert result["success"] is False
            assert "API error: 404" in result["error"]
    
    @pytest.mark.asyncio
    async def test_close(self, ollama_client):
        """Test client close."""
        with patch.object(ollama_client, 'client') as mock_client:
            mock_client.aclose = AsyncMock()
            
            await ollama_client.close()
            
            mock_client.aclose.assert_called_once()
    
    def test_get_status(self, ollama_client):
        """Test status retrieval."""
        ollama_client.available_models = ["llama3.2:latest", "codellama:latest"]
        
        status = ollama_client.get_status()
        
        assert status["base_url"] == "http://localhost:11434"
        assert status["timeout"] == 30.0
        assert status["model_count"] == 2
        assert status["available_models"] == ["llama3.2:latest", "codellama:latest"]
    
    def test_get_recommended_model_available(self, ollama_client):
        """Test getting recommended model when available."""
        ollama_client.available_models = ["llama3.2:latest", "codellama:latest"]
        
        model = ollama_client.get_recommended_model("chat")
        
        assert model == "llama3.2:latest"
    
    def test_get_recommended_model_not_available(self, ollama_client):
        """Test getting recommended model when not available."""
        ollama_client.available_models = ["other:latest"]
        
        model = ollama_client.get_recommended_model("chat")
        
        assert model == "other:latest"  # Falls back to first available
    
    def test_get_recommended_model_no_models(self, ollama_client):
        """Test getting recommended model with no models available."""
        ollama_client.available_models = []
        
        model = ollama_client.get_recommended_model("chat")
        
        assert model == "llama3.2:latest"  # Falls back to default
    
    def test_set_default_model(self, ollama_client):
        """Test setting default model."""
        ollama_client.set_default_model("chat", "custom:latest")
        
        assert ollama_client.default_models["chat"] == "custom:latest"
    
    def test_get_tool_calling_system_prompt(self, ollama_client):
        """Test getting tool calling system prompt."""
        prompt = ollama_client._get_tool_calling_system_prompt()
        
        assert "helpful assistant" in prompt
        assert "TOOL_CALL:" in prompt
        assert "PARAMETERS:" in prompt