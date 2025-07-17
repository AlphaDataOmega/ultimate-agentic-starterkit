"""
Ollama Client for local model integration.

This module provides an HTTP client for Ollama API with health checks,
model management, and tool calling functionality for local AI models.
"""

import json
import time
from typing import Dict, Any, List
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from core.logger import get_logger
from core.config import get_config
from core.voice_alerts import get_voice_alerts


class OllamaError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Exception raised when Ollama service is not available."""
    pass


class OllamaModelError(OllamaError):
    """Exception raised for model-related errors."""
    pass


class OllamaClient:
    """
    Client for Ollama local models.
    
    Provides HTTP client for Ollama API with health checks, model management,
    and tool calling functionality for local AI models.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 30.0):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = get_logger("ollama_client")
        self.config = get_config()
        self.voice = get_voice_alerts()
        
        # HTTP client
        self.client = None
        self.http_available = False
        self._initialize_client()
        
        # Model cache
        self.available_models: List[str] = []
        self.model_info_cache: Dict[str, Dict[str, Any]] = {}
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Default models
        self.default_models = {
            "chat": "llama3.2:latest",
            "code": "codellama:latest",
            "tools": "mistral:latest"
        }
    
    def _initialize_client(self):
        """Initialize HTTP client."""
        if HTTPX_AVAILABLE:
            self.client = httpx.AsyncClient(timeout=self.timeout)
            self.http_available = True
            self.logger.info("Ollama client initialized with httpx")
        elif REQUESTS_AVAILABLE:
            self.http_available = True
            self.logger.info("Ollama client initialized with requests (sync only)")
        else:
            self.logger.error("Neither httpx nor requests available")
            self.http_available = False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama service is running.
        
        Returns:
            Dict containing health check result
        """
        try:
            current_time = time.time()
            
            # Use cached result if recent
            if (current_time - self.last_health_check) < self.health_check_interval:
                return {"healthy": True, "cached": True}
            
            if not self.http_available:
                return {
                    "healthy": False,
                    "error": "HTTP client not available"
                }
            
            # Make health check request
            if HTTPX_AVAILABLE and self.client:
                response = await self.client.get(f"{self.base_url}/api/tags")
                healthy = response.status_code == 200
                
                if healthy:
                    data = response.json()
                    self.available_models = [model["name"] for model in data.get("models", [])]
            else:
                # Fallback to requests
                response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
                healthy = response.status_code == 200
                
                if healthy:
                    data = response.json()
                    self.available_models = [model["name"] for model in data.get("models", [])]
            
            self.last_health_check = current_time
            
            result = {
                "healthy": healthy,
                "available_models": self.available_models,
                "model_count": len(self.available_models),
                "timestamp": datetime.now().isoformat()
            }
            
            if healthy:
                self.logger.debug(f"Ollama health check passed: {len(self.available_models)} models available")
            else:
                self.logger.warning("Ollama health check failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ollama health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List available models.
        
        Returns:
            Dict containing model list
        """
        try:
            if not self.http_available:
                return {
                    "success": False,
                    "error": "HTTP client not available"
                }
            
            if HTTPX_AVAILABLE and self.client:
                response = await self.client.get(f"{self.base_url}/api/tags")
            else:
                response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                # Cache model information
                for model in models:
                    self.model_info_cache[model["name"]] = {
                        "name": model["name"],
                        "size": model.get("size", 0),
                        "digest": model.get("digest", ""),
                        "modified_at": model.get("modified_at", "")
                    }
                
                self.available_models = [model["name"] for model in models]
                
                return {
                    "success": True,
                    "models": models,
                    "model_names": self.available_models,
                    "count": len(models)
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of model to pull
        
        Returns:
            Dict containing pull result
        """
        try:
            self.voice.speak(f"Pulling model {model_name}")
            
            if not self.http_available:
                return {
                    "success": False,
                    "error": "HTTP client not available"
                }
            
            payload = {"name": model_name}
            
            if HTTPX_AVAILABLE and self.client:
                response = await self.client.post(
                    f"{self.base_url}/api/pull",
                    json=payload,
                    timeout=300  # Extended timeout for model pulling
                )
            else:
                response = requests.post(
                    f"{self.base_url}/api/pull",
                    json=payload,
                    timeout=300
                )
            
            if response.status_code == 200:
                self.voice.speak_success(f"Model {model_name} pulled successfully")
                self.logger.info(f"Model {model_name} pulled successfully")
                
                # Refresh model list
                await self.list_models()
                
                return {
                    "success": True,
                    "message": f"Model {model_name} pulled successfully",
                    "model_name": model_name
                }
            else:
                error_msg = f"Failed to pull model {model_name}: {response.status_code}"
                self.voice.speak_error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "details": response.text
                }
                
        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            self.voice.speak_error(f"Model pull failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_response(self, 
                              model: str,
                              prompt: str,
                              system: str = None,
                              context: List[int] = None,
                              stream: bool = False) -> Dict[str, Any]:
        """
        Generate response from Ollama model.
        
        Args:
            model: Model name to use
            prompt: Input prompt
            system: System prompt
            context: Context tokens from previous conversation
            stream: Whether to stream response
        
        Returns:
            Dict containing generation result
        """
        try:
            if not self.http_available:
                return {
                    "success": False,
                    "error": "HTTP client not available"
                }
            
            # Check if model is available
            if model not in self.available_models:
                # Try to pull the model
                pull_result = await self.pull_model(model)
                if not pull_result["success"]:
                    return {
                        "success": False,
                        "error": f"Model {model} not available and pull failed"
                    }
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
            
            if system:
                payload["system"] = system
            
            if context:
                payload["context"] = context
            
            if HTTPX_AVAILABLE and self.client:
                response = await self.client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
            else:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                if stream:
                    return {
                        "success": True,
                        "stream": True,
                        "response": response  # Return response object for streaming
                    }
                else:
                    if HTTPX_AVAILABLE and self.client:
                        result = response.json()
                    else:
                        result = response.json()
                    
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "model": model,
                        "done": result.get("done", False),
                        "context": result.get("context", []),
                        "total_duration": result.get("total_duration", 0),
                        "load_duration": result.get("load_duration", 0),
                        "prompt_eval_count": result.get("prompt_eval_count", 0),
                        "eval_count": result.get("eval_count", 0)
                    }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def chat(self, 
                  model: str,
                  messages: List[Dict[str, str]],
                  stream: bool = False) -> Dict[str, Any]:
        """
        Chat with Ollama model using conversation format.
        
        Args:
            model: Model name to use
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to stream response
        
        Returns:
            Dict containing chat result
        """
        try:
            if not self.http_available:
                return {
                    "success": False,
                    "error": "HTTP client not available"
                }
            
            # Check if model is available
            if model not in self.available_models:
                # Try to pull the model
                pull_result = await self.pull_model(model)
                if not pull_result["success"]:
                    return {
                        "success": False,
                        "error": f"Model {model} not available and pull failed"
                    }
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            
            if HTTPX_AVAILABLE and self.client:
                response = await self.client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
            else:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                if stream:
                    return {
                        "success": True,
                        "stream": True,
                        "response": response
                    }
                else:
                    if HTTPX_AVAILABLE and self.client:
                        result = response.json()
                    else:
                        result = response.json()
                    
                    return {
                        "success": True,
                        "message": result.get("message", {}),
                        "model": model,
                        "done": result.get("done", False),
                        "total_duration": result.get("total_duration", 0),
                        "load_duration": result.get("load_duration", 0),
                        "prompt_eval_count": result.get("prompt_eval_count", 0),
                        "eval_count": result.get("eval_count", 0)
                    }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            self.logger.error(f"Ollama chat failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def tool_calling(self, 
                         model: str,
                         prompt: str,
                         tools: List[Dict[str, Any]],
                         system: str = None) -> Dict[str, Any]:
        """
        Execute tool calling with Ollama model.
        
        Args:
            model: Model name to use (preferably Mistral for tool calling)
            prompt: Input prompt
            tools: List of tool definitions
            system: System prompt
        
        Returns:
            Dict containing tool calling result
        """
        try:
            if not self.http_available:
                return {
                    "success": False,
                    "error": "HTTP client not available"
                }
            
            # Format tools for Ollama
            tool_prompt = self._format_tools_for_ollama(tools, prompt)
            
            # Add system prompt for tool calling
            if not system:
                system = self._get_tool_calling_system_prompt()
            
            response = await self.generate_response(model, tool_prompt, system)
            
            if response["success"]:
                # Parse tool calls from response
                tool_calls = self._parse_tool_calls(response["response"])
                
                return {
                    "success": True,
                    "tool_calls": tool_calls,
                    "raw_response": response["response"],
                    "model": model,
                    "has_tool_calls": len(tool_calls) > 0
                }
            else:
                return response
                
        except Exception as e:
            self.logger.error(f"Tool calling failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_tools_for_ollama(self, tools: List[Dict[str, Any]], prompt: str) -> str:
        """Format tools for Ollama model."""
        tool_descriptions = []
        for tool in tools:
            tool_desc = f"- **{tool['name']}**: {tool['description']}"
            
            if "parameters" in tool:
                params = tool["parameters"]
                if "properties" in params:
                    param_list = []
                    for param_name, param_info in params["properties"].items():
                        param_type = param_info.get("type", "string")
                        param_desc = param_info.get("description", "")
                        param_list.append(f"{param_name} ({param_type}): {param_desc}")
                    
                    tool_desc += f"\n  Parameters: {', '.join(param_list)}"
            
            tool_descriptions.append(tool_desc)
        
        formatted_prompt = f"""You have access to the following tools:

{chr(10).join(tool_descriptions)}

User request: {prompt}

If you need to use a tool, format your response as:
TOOL_CALL: tool_name
PARAMETERS: {{json_parameters}}

If you need to use multiple tools, format each one separately.
Otherwise, provide a direct response without using any tools.
"""
        
        return formatted_prompt
    
    def _get_tool_calling_system_prompt(self) -> str:
        """Get system prompt for tool calling."""
        return """You are a helpful assistant that can use tools to accomplish tasks. 
When you need to use a tool, format your response exactly as:
TOOL_CALL: tool_name
PARAMETERS: {"param1": "value1", "param2": "value2"}

Make sure the parameters are valid JSON. If you don't need to use any tools, 
respond normally with helpful information."""
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from model response."""
        tool_calls = []
        lines = response.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('TOOL_CALL:'):
                tool_name = line.replace('TOOL_CALL:', '').strip()
                
                # Look for parameters on next line
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('PARAMETERS:'):
                    params_str = lines[i + 1].replace('PARAMETERS:', '').strip()
                    try:
                        parameters = json.loads(params_str)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse tool parameters: {e}")
                        parameters = {}
                    
                    tool_calls.append({
                        "name": tool_name,
                        "parameters": parameters
                    })
                    i += 2
                else:
                    tool_calls.append({
                        "name": tool_name,
                        "parameters": {}
                    })
                    i += 1
            else:
                i += 1
        
        return tool_calls
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dict containing model information
        """
        try:
            if not self.http_available:
                return {
                    "success": False,
                    "error": "HTTP client not available"
                }
            
            # Check cache first
            if model_name in self.model_info_cache:
                return {
                    "success": True,
                    "model_info": self.model_info_cache[model_name],
                    "cached": True
                }
            
            payload = {"name": model_name}
            
            if HTTPX_AVAILABLE and self.client:
                response = await self.client.post(
                    f"{self.base_url}/api/show",
                    json=payload
                )
            else:
                response = requests.post(
                    f"{self.base_url}/api/show",
                    json=payload,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                if HTTPX_AVAILABLE and self.client:
                    model_info = response.json()
                else:
                    model_info = response.json()
                
                # Cache the information
                self.model_info_cache[model_name] = model_info
                
                return {
                    "success": True,
                    "model_info": model_info,
                    "cached": False
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close the HTTP client."""
        if HTTPX_AVAILABLE and self.client:
            await self.client.aclose()
            self.logger.info("Ollama client closed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of Ollama client."""
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "http_available": self.http_available,
            "available_models": self.available_models,
            "model_count": len(self.available_models),
            "cached_models": len(self.model_info_cache),
            "last_health_check": self.last_health_check,
            "default_models": self.default_models
        }
    
    def get_recommended_model(self, task_type: str = "chat") -> str:
        """
        Get recommended model for task type.
        
        Args:
            task_type: Type of task (chat, code, tools)
        
        Returns:
            Recommended model name
        """
        if task_type in self.default_models:
            recommended = self.default_models[task_type]
            
            # Check if recommended model is available
            if recommended in self.available_models:
                return recommended
        
        # Fallback to first available model
        if self.available_models:
            return self.available_models[0]
        
        # Ultimate fallback
        return self.default_models.get(task_type, "llama3.2:latest")
    
    def set_default_model(self, task_type: str, model_name: str):
        """
        Set default model for task type.
        
        Args:
            task_type: Type of task (chat, code, tools)
            model_name: Model name to set as default
        """
        self.default_models[task_type] = model_name
        self.logger.info(f"Default model for {task_type} set to {model_name}")


# Convenience functions for common operations
async def quick_chat(prompt: str, model: str = None, base_url: str = "http://localhost:11434") -> str:
    """
    Quick chat function for simple interactions.
    
    Args:
        prompt: Input prompt
        model: Model to use (auto-detected if None)
        base_url: Ollama base URL
    
    Returns:
        Model response text
    """
    client = OllamaClient(base_url)
    
    try:
        # Health check first
        health = await client.health_check()
        if not health["healthy"]:
            return "Ollama service not available"
        
        # Use recommended model if none specified
        if not model:
            model = client.get_recommended_model("chat")
        
        # Generate response
        result = await client.generate_response(model, prompt)
        
        if result["success"]:
            return result["response"]
        else:
            return f"Error: {result['error']}"
            
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        await client.close()


async def quick_tool_call(prompt: str, tools: List[Dict[str, Any]], model: str = None, base_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """
    Quick tool calling function.
    
    Args:
        prompt: Input prompt
        tools: List of tool definitions
        model: Model to use (auto-detected if None)
        base_url: Ollama base URL
    
    Returns:
        List of tool calls
    """
    client = OllamaClient(base_url)
    
    try:
        # Health check first
        health = await client.health_check()
        if not health["healthy"]:
            return []
        
        # Use recommended model if none specified
        if not model:
            model = client.get_recommended_model("tools")
        
        # Execute tool calling
        result = await client.tool_calling(model, prompt, tools)
        
        if result["success"]:
            return result["tool_calls"]
        else:
            return []
            
    except Exception:
        return []
    finally:
        await client.close()