"""OpenCode SST API client for session management and agent orchestration.

This module provides a comprehensive client for interacting with OpenCode SST
server endpoints. It handles session creation, agent execution, shell commands,
and code analysis through OpenCode's REST API.

The client enables leveraging OpenCode's built-in capabilities for session
management, real-time event streaming, and isolated execution environments
instead of implementing these features from scratch.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel

from core.config import OpenCodeConfig

logger = logging.getLogger(__name__)


class OpenCodeSession(BaseModel):
    """Represents an OpenCode SST session.

    Attributes:
        session_id: Unique identifier for the session.
        parent_id: Optional parent session ID for hierarchical sessions.
        status: Current status of the session.
        created_at: Session creation timestamp.
        metadata: Additional session metadata.
    """

    session_id: str
    parent_id: str | None = None
    status: str = "active"
    created_at: str
    metadata: dict[str, Any] = {}


class OpenCodeShellResult(BaseModel):
    """Result from executing a shell command in OpenCode session.

    Attributes:
        command: The shell command that was executed.
        exit_code: Command exit status code.
        stdout: Standard output from the command.
        stderr: Standard error output from the command.
        execution_time: Time taken to execute the command in seconds.
    """

    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float


class OpenCodeClient:
    """Async client for OpenCode SST server API interactions.

    This client provides comprehensive access to OpenCode SST functionality
    including session management, agent orchestration, shell execution, and
    code analysis. It replaces custom implementations with OpenCode's
    proven infrastructure.

    Attributes:
        config: OpenCode configuration settings.
        client: Async HTTP client for API requests.
        active_sessions: Dictionary of currently active sessions.
    """

    def __init__(self, config: OpenCodeConfig) -> None:
        """Initialize the OpenCode SST client.

        Args:
            config: OpenCode configuration including server details and settings.
        """
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.server_url,
            timeout=httpx.Timeout(config.session_timeout_seconds),
        )
        self.active_sessions: dict[str, OpenCodeSession] = {}

        logger.info(f"Initialized OpenCode client for {config.server_url}")

    async def __aenter__(self) -> OpenCodeClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client and cleanup active sessions."""
        # Clean up active sessions
        for session_id in list(self.active_sessions.keys()):
            try:
                await self.delete_session(session_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup session {session_id}: {e}")

        await self.client.aclose()
        logger.info("OpenCode client closed")

    async def create_session(
        self, parent_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> OpenCodeSession:
        """Create a new OpenCode SST session.

        Args:
            parent_id: Optional parent session for hierarchical organization.
            metadata: Additional metadata to associate with the session.

        Returns:
            OpenCodeSession object representing the created session.

        Raises:
            httpx.HTTPError: If session creation fails.
        """
        payload = {}
        if parent_id:
            payload["parent_id"] = parent_id
        if metadata:
            payload["metadata"] = metadata

        try:
            logger.debug(f"Creating session with payload: {payload}")
            logger.debug(f"Sending POST to: {self.client.base_url}/session")
            response = await self.client.post("/session", json=payload)
            logger.debug(f"Session creation response status: {response.status_code}")
            logger.debug(f"Session creation response text: {response.text}")
            response.raise_for_status()
        except Exception as e:
            logger.error(
                f"Failed to create session. URL: {self.client.base_url}/session"
            )
            logger.error(f"Payload: {payload}")
            logger.error(f"Error: {e}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            raise

        session_data = response.json()
        session = OpenCodeSession(
            session_id=session_data["id"],
            parent_id=parent_id,
            created_at=session_data.get("created_at", ""),
            metadata=metadata or {},
        )

        self.active_sessions[session.session_id] = session
        logger.info(f"Created OpenCode session: {session.session_id}")

        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete an OpenCode SST session.

        Args:
            session_id: ID of the session to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            response = await self.client.delete(f"/session/{session_id}")
            response.raise_for_status()

            self.active_sessions.pop(session_id, None)
            logger.info(f"Deleted OpenCode session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    async def execute_shell_command(
        self, session_id: str, command: str, timeout: int | None = None
    ) -> OpenCodeShellResult:
        """Execute a shell command within an OpenCode session.

        Args:
            session_id: ID of the session to execute the command in.
            command: Shell command to execute.
            timeout: Optional timeout override for this command.

        Returns:
            OpenCodeShellResult containing command output and status.

        Raises:
            httpx.HTTPError: If command execution request fails.
        """
        payload = {"command": command}
        if timeout:
            payload["timeout"] = timeout

        response = await self.client.post(f"/session/{session_id}/shell", json=payload)
        response.raise_for_status()

        result_data = response.json()

        result = OpenCodeShellResult(
            command=command,
            exit_code=result_data.get("exit_code", 0),
            stdout=result_data.get("stdout", ""),
            stderr=result_data.get("stderr", ""),
            execution_time=result_data.get("execution_time", 0.0),
        )

        logger.debug(f"Executed shell command in session {session_id}: {command}")
        return result

    async def find_in_files(self, pattern: str) -> list[dict[str, any]]:
        """Find text patterns in files using OpenCode's find API.

        Uses OpenCode's /find endpoint to search for text patterns across files.
        This is useful for finding code dependencies and usage patterns.

        Args:
            pattern: Text pattern to search for (supports regex).

        Returns:
            List of match objects with path, lines, line numbers, and submatches.
        """
        endpoint = f"{self.base_url}/find"
        params = {"pattern": pattern}

        try:
            response = await self.session.get(endpoint, params=params)
            response.raise_for_status()

            results = response.json()
            logger.debug(f"Found {len(results)} matches for pattern: {pattern}")
            return results

        except Exception as e:
            logger.error(f"Failed to find pattern '{pattern}': {e}")
            return []

    async def find_symbols(self, query: str) -> list[dict[str, any]]:
        """Find workspace symbols using OpenCode's symbol search API.

        Uses OpenCode's /find/symbol endpoint to search for symbols (functions, classes, etc.)
        across the workspace. Useful for finding definitions and references.

        Args:
            query: Symbol search query.

        Returns:
            List of symbol objects with location and definition information.
        """
        endpoint = f"{self.base_url}/find/symbol"
        params = {"query": query}

        try:
            response = await self.session.get(endpoint, params=params)
            response.raise_for_status()

            results = response.json()
            logger.debug(f"Found {len(results)} symbols for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Failed to find symbols for '{query}': {e}")
            return []

    async def search_files(
        self, session_id: str, query: str, file_patterns: list[str] | None = None
    ) -> list[str]:
        """Search for files using OpenCode's file search capability.

        Args:
            session_id: ID of the session to search in.
            query: Search query for file discovery.
            file_patterns: Optional file patterns to filter results.

        Returns:
            List of file paths matching the search criteria.
        """
        payload = {"query": query}
        if file_patterns:
            payload["patterns"] = file_patterns

        response = await self.client.post(
            f"/session/{session_id}/files/search", json=payload
        )
        response.raise_for_status()

        result = response.json()
        files = result.get("files", [])

        logger.debug(f"Found {len(files)} files matching query: {query}")
        return files

    async def get_file_symbols(
        self, session_id: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Get code symbols from a file using OpenCode's analysis.

        Args:
            session_id: ID of the session containing the file.
            file_path: Path to the file to analyze.

        Returns:
            List of symbol information including functions, classes, etc.
        """
        response = await self.client.get(
            f"/session/{session_id}/files/{file_path}/symbols"
        )
        response.raise_for_status()

        result = response.json()
        symbols = result.get("symbols", [])

        logger.debug(f"Found {len(symbols)} symbols in {file_path}")
        return symbols

    async def read_file(self, session_id: str, file_path: str) -> str | None:
        """Read the content of a file in the session.

        Args:
            session_id: ID of the session containing the file.
            file_path: Path to the file to read.

        Returns:
            File content as string, or None if reading fails.
        """
        try:
            response = await self.client.get(f"/session/{session_id}/files/{file_path}")
            response.raise_for_status()

            result = response.json()
            content = result.get("content", "")

            logger.debug(f"Read {len(content)} characters from {file_path}")
            return content

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

    async def write_file(
        self, session_id: str, file_path: str, content: str
    ) -> bool:
        """Write content to a file in the session.

        Args:
            session_id: ID of the session to write the file in.
            file_path: Path to the file to write.
            content: Content to write to the file.

        Returns:
            True if the file was written successfully, False otherwise.
        """
        try:
            # Try direct API endpoint first
            payload = {"content": content}
            response = await self.client.put(
                f"/session/{session_id}/files/{file_path}", json=payload
            )
            
            # If direct endpoint doesn't exist, fall back to shell command
            if response.status_code == 404:
                logger.debug(
                    f"Direct write endpoint not found, using shell command for {file_path}"
                )
                return await self._write_file_via_shell(session_id, file_path, content)
            
            response.raise_for_status()
            logger.debug(f"Wrote {len(content)} characters to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            # Fall back to shell command if direct API fails
            logger.debug(f"Falling back to shell command for {file_path}")
            return await self._write_file_via_shell(session_id, file_path, content)

    async def _write_file_via_shell(
        self, session_id: str, file_path: str, content: str
    ) -> bool:
        """Write file using shell commands as fallback.
        
        Args:
            session_id: ID of the session to write the file in.
            file_path: Path to the file to write.
            content: Content to write to the file.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            from pathlib import Path
            dir_path = str(Path(file_path).parent)
            if dir_path != ".":
                mkdir_result = await self.execute_shell_command(
                    session_id, f"mkdir -p '{dir_path}'"
                )
                if mkdir_result.exit_code != 0:
                    logger.error(f"Failed to create directory {dir_path}")
                    return False
            
            # Use a here-document to safely write content with proper escaping
            escaped_content = content.replace("'", "'\"'\"'")
            write_command = f"cat > '{file_path}' << 'EOF'\n{escaped_content}\nEOF"
            
            result = await self.execute_shell_command(session_id, write_command)
            
            if result.exit_code == 0:
                logger.debug(f"Successfully wrote {len(content)} characters to {file_path}")
                return True
            else:
                logger.error(
                    f"Shell write failed for {file_path}: {result.stderr}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Shell write failed for {file_path}: {e}")
            return False

    async def delete_file(self, session_id: str, file_path: str) -> bool:
        """Delete a file in the session.

        Args:
            session_id: ID of the session containing the file.
            file_path: Path to the file to delete.

        Returns:
            True if the file was deleted successfully, False otherwise.
        """
        try:
            # Try direct API endpoint first
            response = await self.client.delete(
                f"/session/{session_id}/files/{file_path}"
            )
            
            # If direct endpoint doesn't exist, fall back to shell command
            if response.status_code == 404:
                logger.debug(
                    f"Direct delete endpoint not found, using shell command for {file_path}"
                )
                return await self._delete_file_via_shell(session_id, file_path)
            
            response.raise_for_status()
            logger.debug(f"Deleted file {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            # Fall back to shell command if direct API fails
            logger.debug(f"Falling back to shell command for {file_path}")
            return await self._delete_file_via_shell(session_id, file_path)

    async def _delete_file_via_shell(
        self, session_id: str, file_path: str
    ) -> bool:
        """Delete file using shell commands as fallback.
        
        Args:
            session_id: ID of the session to delete the file in.
            file_path: Path to the file to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            result = await self.execute_shell_command(
                session_id, f"rm -f '{file_path}'"
            )
            
            if result.exit_code == 0:
                logger.debug(f"Successfully deleted file {file_path}")
                return True
            else:
                logger.error(f"Shell delete failed for {file_path}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Shell delete failed for {file_path}: {e}")
            return False

    async def send_prompt(
        self,
        session_id: str,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        agent_id: str | None = None,
        response_format: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a prompt to an LLM through OpenCode's provider system.

        OpenCode manages the LLM provider authentication and routing,
        eliminating the need for direct API key management.

        Args:
            session_id: ID of the session to send the prompt to.
            prompt: The prompt text to send to the LLM.
            model: Model name (e.g., "claude-3-5-sonnet-20241022").
            temperature: Sampling temperature for response generation.
            max_tokens: Maximum tokens in the response.
            system_prompt: Optional system prompt for the LLM.
            agent_id: Optional specific agent ID for tracking.
            response_format: Format for the response ("json_object" for JSON).
            json_schema: JSON schema to enforce structured output format.

        Returns:
            LLM response containing the generated content and metadata.
        """
        # Build parts array with proper format
        parts = [{"type": "text", "text": prompt}]

        # Build payload with correct OpenCode API format
        payload = {"parts": parts}

        # Handle model parameter - convert string to OpenCode format
        if model:
            # Parse model name to extract provider and model ID
            if "claude" in model.lower():
                payload["model"] = {"providerID": "anthropic", "modelID": model}
            elif "gpt" in model.lower() or "openai" in model.lower():
                payload["model"] = {"providerID": "openai", "modelID": model}
            else:
                # Default to anthropic for unknown models
                payload["model"] = {"providerID": "anthropic", "modelID": model}

        # Add other parameters if supported by OpenCode API
        # Note: OpenCode may not support all these parameters directly
        # We'll include them in case they are supported
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if system_prompt:
            payload["system"] = system_prompt
        if agent_id:
            payload["agent_id"] = agent_id
        if response_format:
            payload["response_format"] = response_format
        if json_schema:
            payload["json_schema"] = json_schema

        response = await self.client.post(
            f"/session/{session_id}/message", json=payload
        )
        response.raise_for_status()

        result = response.json()
        logger.debug(
            f"Sent prompt to session {session_id} using model {model or 'default'}"
        )

        return result

    async def list_sessions(self) -> list[OpenCodeSession]:
        """List all active OpenCode sessions.

        Returns:
            List of OpenCodeSession objects for all active sessions.
        """
        response = await self.client.get("/sessions")
        response.raise_for_status()

        sessions_data = response.json()
        sessions = []

        for session_data in sessions_data.get("sessions", []):
            session = OpenCodeSession(
                session_id=session_data["id"],
                parent_id=session_data.get("parent_id"),
                status=session_data.get("status", "active"),
                created_at=session_data.get("created_at", ""),
                metadata=session_data.get("metadata", {}),
            )
            sessions.append(session)

        logger.debug(f"Listed {len(sessions)} active sessions")
        return sessions

    async def initialize_session_for_repository(
        self, repository_path: str, problem_description: str
    ) -> OpenCodeSession:
        """Initialize an OpenCode session for a specific repository.

        Creates a session and sets up the working directory to point to the
        target repository for code analysis and patch generation.

        Args:
            repository_path: Path to the repository to work with.
            problem_description: Description of the problem to solve.

        Returns:
            Initialized OpenCodeSession ready for patch generation work.
        """
        session = await self.create_session(
            metadata={
                "repository_path": repository_path,
                "problem_description": problem_description,
                "purpose": "patch_generation",
            }
        )

        # Change to repository directory (only if shell execution is enabled)
        if self.config.enable_shell_execution:
            await self.execute_shell_command(
                session.session_id, f"cd {repository_path}"
            )
        else:
            logger.debug("Skipping shell command (shell execution disabled)")

        logger.info(
            f"Initialized session {session.session_id} for repository: {repository_path}"
        )
        return session

    async def run_tests_in_session(
        self, session_id: str, test_command: str = "pytest", timeout: int | None = None
    ) -> OpenCodeShellResult:
        """Run tests within an OpenCode session.

        Args:
            session_id: ID of the session to run tests in.
            test_command: Test command to execute (default: pytest).
            timeout: Optional timeout for test execution.

        Returns:
            OpenCodeShellResult containing test execution results.
        """
        result = await self.execute_shell_command(
            session_id, test_command, timeout or self.config.session_timeout_seconds
        )

        logger.info(
            f"Test execution in session {session_id}: exit_code={result.exit_code}"
        )
        return result
