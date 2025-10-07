"""Individual AI agent implementation for autonomous patch generation.

This module implements a Claude-powered agent that works within the OpenCode SST
framework to generate code patches. Each agent can have specialized roles and
behaviors to encourage diverse solution approaches to the same problem.

The agent takes problem descriptions and code context as input, then uses
Claude's reasoning capabilities to propose specific code changes that could
address the identified issues.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from core.config import OpenCodeConfig
from core.path_utils import to_repo_relative
from core.role_manager import RoleManager
from core.types import AgentConfig, CodeContext, PatchCandidate
from opencode_client import OpenCodeClient
from operations import (
    FileOperationsService,
    OpenCodeBackend,
    LocalBackend,
    FileOperationsConfig,
    FileOperationError,
    SecurityViolationError,
)

logger = logging.getLogger(__name__)


class OpenCodeAgent:
    """Autonomous AI agent for generating code patch candidates through OpenCode SST.

    This class represents a single AI agent that operates within the OpenCode SST
    framework to generate patch proposals for code issues. OpenCode manages the
    LLM provider authentication and routing, eliminating direct API management.

    Each agent can have different specializations, model parameters, and prompting
    strategies to encourage solution diversity while leveraging OpenCode's session
    management and provider integration.

    Attributes:
        agent_config: Configuration parameters for this specific agent.
        opencode_config: OpenCode SST framework configuration.
        opencode_client: OpenCode client for session and LLM communication.
        session_id: Active OpenCode session ID for this agent.
        file_ops_service: Optional file operations service for direct file I/O.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        opencode_config: OpenCodeConfig,
        role_manager: RoleManager | None = None,
    ) -> None:
        """Initialize the OpenCode-powered patch generation agent.

        Args:
            agent_config: Individual agent configuration including model settings,
                specialized role, and behavioral parameters.
            opencode_config: OpenCode SST framework configuration for orchestration.
            role_manager: Optional role manager for loading role definitions from files.
                If None, a default instance will be created.
        """
        self.agent_config = agent_config
        self.opencode_config = opencode_config
        self.opencode_client = (
            OpenCodeClient(opencode_config) if opencode_config.enabled else None
        )
        self.session_id: str | None = None
        self.role_manager = role_manager or RoleManager()
        self.file_ops_service: FileOperationsService | None = None
        self.repository_path: Path | None = None

        logger.info(f"Initialized OpenCode agent {agent_config.agent_id}")

    async def generate_solution(
        self,
        problem_description: str,
        relevant_contexts: list[CodeContext],
    ) -> list[PatchCandidate]:
        """Generate a complete solution that may span multiple files.

        This method allows agents to propose coordinated changes across multiple files
        to solve complex problems that require cross-file modifications. It replaces
        the per-file patch generation approach with solution-based generation.

        Args:
            problem_description: Human-readable description of the issue to fix.
            relevant_contexts: List of code context chunks relevant to the problem.

        Returns:
            List of PatchCandidate objects representing a cohesive multi-file solution.
            Empty list if the agent was unable to generate a valid solution.

        Raises:
            Exception: If OpenCode communication fails or response parsing errors.
        """
        if not self.opencode_client or not self.session_id:
            logger.error(
                f"Agent {self.agent_config.agent_id} has no active OpenCode session"
            )
            return []

        try:
            # Prepare context for the agent
            context_text = self._format_contexts(relevant_contexts)

            # Create specialized prompt based on agent role
            system_prompt = self._create_solution_system_prompt()
            user_prompt = self._create_solution_user_prompt(
                problem_description, context_text
            )

            # Define JSON schema for structured solution output
            solution_schema = {
                "type": "object",
                "properties": {
                    "solution_description": {
                        "type": "string",
                        "description": "Overall description of the solution approach",
                    },
                    "patches": {
                        "type": "array",
                        "description": "List of patches that together form the complete solution",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Target file path for this patch",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The exact replacement code for the specific lines identified by line_start and line_end",
                                },
                                "line_start": {
                                    "type": "integer",
                                    "description": "Starting line number (1-indexed, as shown in the provided context) of the first line to replace",
                                    "minimum": 1,
                                },
                                "line_end": {
                                    "type": "integer",
                                    "description": "Ending line number (1-indexed, as shown in the provided context) of the last line to replace (inclusive)",
                                    "minimum": 1,
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of what this specific patch does",
                                },
                            },
                            "required": [
                                "file_path",
                                "content",
                                "line_start",
                                "line_end",
                                "description",
                            ],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    },
                    "confidence_score": {
                        "type": "number",
                        "description": "Agent confidence in the overall solution (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["solution_description", "patches", "confidence_score"],
                "additionalProperties": False,
            }

            # Generate solution using OpenCode's LLM provider management with structured output
            response = await self.opencode_client.send_prompt(
                session_id=self.session_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.agent_config.model_name,
                temperature=self.agent_config.temperature,
                max_tokens=self.agent_config.max_tokens,
                agent_id=self.agent_config.agent_id,
                response_format="json_object",
                json_schema=solution_schema,
            )

            # Parse response to extract solution patches
            patches = await self._parse_solution_response(response)

            if patches:
                logger.info(
                    f"Agent {self.agent_config.agent_id} generated solution with {len(patches)} patches"
                )
            else:
                logger.warning(
                    f"Agent {self.agent_config.agent_id} failed to generate valid solution"
                )

            return patches

        except Exception as e:
            logger.error(
                f"Agent {self.agent_config.agent_id} failed to generate solution: {e}"
            )
            return []

    async def generate_patch(
        self,
        problem_description: str,
        relevant_contexts: list[CodeContext],
        target_file: str,
    ) -> PatchCandidate | None:
        """Generate a patch candidate to address a specific code problem.

        This method orchestrates the entire patch generation process for this agent
        using OpenCode's LLM provider management. It formats the provided code context,
        creates specialized prompts based on the agent's role, queries the LLM through
        OpenCode, and parses the response into a structured patch candidate.

        Args:
            problem_description: Human-readable description of the issue to fix.
            relevant_contexts: List of code context chunks relevant to the problem.
            target_file: Path to the file where the patch should be applied.

        Returns:
            A PatchCandidate object containing the proposed fix, or None if the
            agent was unable to generate a valid solution.

        Raises:
            Exception: If OpenCode communication fails or response parsing errors.
        """
        if not self.opencode_client or not self.session_id:
            logger.error(
                f"Agent {self.agent_config.agent_id} has no active OpenCode session"
            )
            return None

        try:
            # Prepare context for the agent
            context_text = self._format_contexts(relevant_contexts)

            # Create specialized prompt based on agent role
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_solution_user_prompt(
                problem_description, context_text
            )

            # Define JSON schema for structured solution output
            solution_schema = {
                "type": "object",
                "properties": {
                    "solution_description": {
                        "type": "string",
                        "description": "Overall description of the solution approach",
                    },
                    "patches": {
                        "type": "array",
                        "description": "List of patches that together form the complete solution",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Target file path for this patch",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The exact replacement code for the specific lines identified by line_start and line_end",
                                },
                                "line_start": {
                                    "type": "integer",
                                    "description": "Starting line number (1-indexed, as shown in the provided context) of the first line to replace",
                                    "minimum": 1,
                                },
                                "line_end": {
                                    "type": "integer",
                                    "description": "Ending line number (1-indexed, as shown in the provided context) of the last line to replace (inclusive)",
                                    "minimum": 1,
                                },
                                "confidence_score": {
                                    "type": "number",
                                    "description": "Agent confidence (0.0-1.0)",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Brief description of this patch",
                                },
                            },
                            "required": [
                                "file_path",
                                "content",
                                "line_start",
                                "line_end",
                                "confidence_score",
                                "description",
                            ],
                        },
                    },
                },
                "required": ["solution_description", "patches"],
                "additionalProperties": False,
            }

            # Generate patch using OpenCode's LLM provider management with structured output
            response = await self.opencode_client.send_prompt(
                session_id=self.session_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.agent_config.model_name,
                temperature=self.agent_config.temperature,
                max_tokens=self.agent_config.max_tokens,
                agent_id=self.agent_config.agent_id,
                response_format="json_object",
                json_schema=solution_schema,
            )

            # Parse response to extract solution patches
            patches = await self._parse_solution_response(response)
            # For backward compatibility, return the first patch targeting the specified file
            if patches:
                target_patch = next(
                    (p for p in patches if p.file_path == target_file), patches[0]
                )
                logger.info(
                    f"Agent {self.agent_config.agent_id} generated solution with {len(patches)} patches"
                )
                return target_patch
            else:
                logger.warning(
                    f"Agent {self.agent_config.agent_id} failed to generate valid solution"
                )
                return None

        except Exception as e:
            logger.error(
                f"Agent {self.agent_config.agent_id} failed to generate patch: {e}"
            )
            return None

    def set_session_id(self, session_id: str) -> None:
        """Set the OpenCode session ID for this agent.

        Args:
            session_id: OpenCode session ID to associate with this agent.
        """
        self.session_id = session_id
        logger.debug(
            f"Agent {self.agent_config.agent_id} assigned to session {session_id}"
        )
    
    def set_repository_path(self, repository_path: str | Path) -> None:
        """Set the repository path for this agent.
        
        Args:
            repository_path: Path to the repository root directory.
        """
        self.repository_path = Path(repository_path)
        logger.debug(
            f"Agent {self.agent_config.agent_id} set repository path: {self.repository_path}"
        )
    
    def set_repository_path(self, repository_path: str | Path) -> None:
        """Set the repository path for this agent.
        
        Args:
            repository_path: Path to the repository root directory.
        """
        self.repository_path = Path(repository_path)
        logger.debug(
            f"Agent {self.agent_config.agent_id} set repository path: {self.repository_path}"
        )

    def initialize_file_operations(
        self, repo_path: str, enable_direct_ops: bool = False
    ) -> None:
        """Initialize file operations service for direct file I/O.

        Args:
            repo_path: Path to the repository root directory.
            enable_direct_ops: Whether to enable direct file operations.
        """
        if not enable_direct_ops or not self.opencode_config.enable_direct_file_ops:
            logger.debug(
                f"Agent {self.agent_config.agent_id}: Direct file ops disabled"
            )
            return

        try:
            # Choose backend based on OpenCode availability
            if self.opencode_client and self.session_id:
                backend = OpenCodeBackend(self.opencode_client, self.session_id)
                logger.info(
                    f"Agent {self.agent_config.agent_id}: Using OpenCode file backend"
                )
            else:
                from pathlib import Path

                backend = LocalBackend(Path(repo_path))
                logger.info(
                    f"Agent {self.agent_config.agent_id}: Using local file backend"
                )

            # Initialize file operations service with safety constraints
            self.file_ops_service = FileOperationsService(
                repo_root=repo_path,
                backend=backend,
                config=FileOperationsConfig(),
            )

            logger.info(
                f"Agent {self.agent_config.agent_id}: File operations service initialized"
            )

        except Exception as e:
            logger.error(
                f"Agent {self.agent_config.agent_id}: "
                f"Failed to initialize file operations: {e}"
            )
            self.file_ops_service = None

    def _create_system_prompt(self) -> str:
        """Create system prompt based on agent configuration."""
        base_prompt = (
            self.agent_config.system_prompt or "You are a skilled software engineer."
        )

        # Get role-specific addition from role manager
        role_addition = self.role_manager.get_role_prompt_addition(
            self.agent_config.specialized_role
        )

        return f"""
{base_prompt}

{role_addition}

Your task is to generate a code patch that fixes the described problem.

Analyze the problem carefully and provide a precise fix that:
1. Addresses the root cause of the issue
2. Maintains code quality and style consistency
3. Handles edge cases appropriately
4. Follows best practices for the programming language

You will respond with a structured JSON object containing:
- content: The exact replacement code
- line_start: Starting line number (1-indexed, as shown in the context)
- line_end: Ending line number (1-indexed, as shown in the context)
- confidence_score: Your confidence in this solution (0.0-1.0)
- description: Clear explanation of what the fix accomplishes

The JSON schema is enforced, so ensure all required fields are provided.
        """.strip()

    def _format_contexts(self, contexts: list[CodeContext]) -> str:
        """Format code contexts for inclusion in prompt."""
        if not contexts:
            return "No relevant context available."

        formatted_contexts = []
        for i, context in enumerate(contexts):
            # Add line numbers to help agents identify correct line ranges
            # Use 1-indexed numbering (human-readable) to match system prompt
            lines = context.content.split("\n")
            numbered_content = "\n".join(
                f"{idx+1:2d}|{line}" for idx, line in enumerate(lines)
            )

            formatted_contexts.append(
                f"""
Context {i + 1} - {context.file_path} ({context.language}):
```{context.language}
{numbered_content}
```

Functions: {", ".join(context.relevant_functions) if context.relevant_functions else "None"}
Dependencies: {", ".join(context.dependencies) if context.dependencies else "None"}
            """.strip()
            )

        return "\n\n".join(formatted_contexts)

    def _parse_opencode_response(
        self,
        response: dict[str, any],
        target_file: str,
    ) -> PatchCandidate | None:
        """Parse OpenCode's structured JSON response to extract patch information.

        With structured output enabled, the response should already be valid JSON
        conforming to the schema, making parsing more reliable.
        """
        try:
            # For structured output, try direct JSON parsing first
            if isinstance(response, dict) and all(
                field in response
                for field in [
                    "content",
                    "line_start",
                    "line_end",
                    "confidence_score",
                    "description",
                ]
            ):
                # Response is already structured JSON
                patch_data = response
                logger.debug("Using direct structured JSON response")
            else:
                # Fallback to content extraction for backward compatibility
                content = (
                    response.get("content")
                    or response.get("text")
                    or response.get("response", "")
                )
                if not content:
                    logger.error("No content found in OpenCode response")
                    return None

                # For structured output, content should be valid JSON
                if isinstance(content, str):
                    try:
                        patch_data = json.loads(content)
                        logger.debug("Parsed JSON from content string")
                    except json.JSONDecodeError:
                        # Fallback to regex extraction for non-structured responses
                        json_match = re.search(
                            r"```json\s*(\{.*?\})\s*```", content, re.DOTALL
                        )
                        if json_match:
                            patch_data = json.loads(json_match.group(1))
                            logger.debug("Extracted JSON from markdown code block")
                        else:
                            # Look for JSON object in the response
                            json_match = re.search(
                                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL
                            )
                            if json_match:
                                patch_data = json.loads(json_match.group(0))
                                logger.debug("Extracted JSON from response text")
                            else:
                                logger.error("No valid JSON found in OpenCode response")
                                return None
                else:
                    patch_data = content

            # Validate required fields (schema should guarantee this, but check for robustness)
            required_fields = [
                "content",
                "line_start",
                "line_end",
                "confidence_score",
                "description",
            ]
            for field in required_fields:
                if field not in patch_data:
                    logger.error(f"Missing required field in patch response: {field}")
                    return None

            # Create PatchCandidate with normalized file path
            normalized_file_path = (
                to_repo_relative(target_file, self.repository_path) 
                if self.repository_path and Path(target_file).is_absolute()
                else target_file
            )
            
            patch = PatchCandidate(
                content=patch_data["content"],
                description=patch_data["description"],
                agent_id=self.agent_config.agent_id,
                file_path=normalized_file_path,
                line_start=int(patch_data["line_start"]),
                line_end=int(patch_data["line_end"]),
                confidence_score=float(patch_data["confidence_score"]),
                metadata={
                    "model": self.agent_config.model_name,
                    "temperature": self.agent_config.temperature,
                    "specialized_role": self.agent_config.specialized_role,
                    "opencode_session": self.session_id,
                    "raw_response": content,
                    "opencode_response": response,
                },
            )

            return patch

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse OpenCode response: {e}")
            logger.debug(f"Raw response: {response}")
            return None

    def _parse_patch_response(
        self,
        response: any,  # Kept for backward compatibility
        target_file: str,
    ) -> PatchCandidate | None:
        """Legacy method for backward compatibility."""
        logger.warning(
            "Using deprecated _parse_patch_response, use _parse_opencode_response instead"
        )
        return None

    async def validate_patch_syntax(
        self,
        patch: PatchCandidate,
        language: str,
    ) -> bool:
        """Validate that the patch has correct syntax for the target language."""
        try:
            if language == "python":
                import ast

                ast.parse(patch.content)
                return True
            elif language in ["javascript", "typescript"]:
                # Basic validation - could be enhanced with actual JS parser
                return not any(
                    syntax_error in patch.content
                    for syntax_error in ["SyntaxError", "unexpected token"]
                )
            # Add more language-specific validation as needed
            return True

        except Exception as e:
            logger.warning(f"Syntax validation failed for patch {patch.id}: {e}")
            return False

    def _create_solution_system_prompt(self) -> str:
        """Create system prompt for solution-based generation."""
        base_prompt = (
            self.agent_config.system_prompt or "You are a skilled software engineer."
        )

        # Get role-specific addition from role manager
        role_addition = self.role_manager.get_role_prompt_addition(
            self.agent_config.specialized_role
        )

        return f"""
{base_prompt}

{role_addition}

Your task is to generate a complete solution that fixes the described problem.
This solution may require changes to multiple files that work together cohesively.

Analyze the problem carefully and provide a comprehensive solution that:
1. Addresses the root cause of the issue across all necessary files
2. Maintains code quality and style consistency across all changes
3. Ensures all files work together properly after the changes
4. Handles edge cases appropriately
5. Follows best practices for the programming language

Respond with a JSON object in this format:
{{
  "solution_description": "Overall description of your solution approach",
  "patches": [
    {{
      "file_path": "The file to modify",
      "content": "The exact replacement code for the specific lines",
      "line_start": 0,
      "line_end": 10,
      "description": "What this specific patch accomplishes"
    }}
  ],
  "confidence_score": 0.85
}}

CRITICAL DOCSTRING TARGETING RULES:
- When improving docstrings: Replace ONLY the existing docstring lines (triple quotes), NOT the function definition or body
- Use the numbered line references to identify the EXACT docstring lines to replace
- NEVER include 'def function_name():' in your replacement content
- NEVER include 'return' statements or function body code in docstring patches
- If a function has a simple docstring, replace ONLY that line
- Your replacement content should start and end with triple quotes
- Preserve the original indentation level of the docstring

Example: If you see:
  7|def add(a, b):
  8|    '''Add two numbers.'''
  9|    return a + b

To replace the docstring, target line_start: 8, line_end: 8, content with proper docstring only

The JSON schema is enforced, so ensure all required fields are provided.
        """.strip()

    def _create_solution_user_prompt(
        self,
        problem_description: str,
        context_text: str,
    ) -> str:
        """Create user prompt for solution-based generation."""
        return f"""
Problem Description:
{problem_description}

Relevant Code Context:
{context_text}

Please analyze this problem comprehensively and provide a complete solution that may
span multiple files if necessary. Consider all the code contexts provided and think
about how changes in one file might affect others.

The code contexts above include line numbers (e.g., "8|    def function():"). Use these
line numbers to identify the EXACT lines that need to be changed. Be very precise with
your line_start and line_end values.

Your solution should be cohesive and ensure all files work together properly after
the changes are applied. If the problem can be solved with changes to a single file,
that's fine too - provide the most appropriate solution for the specific problem.

Remember: only replace the specific lines that need to be changed. For docstring
improvements, replace only the docstring lines, not the entire function.

Respond with a JSON object containing your complete solution as specified in the system prompt.
        """.strip()

    async def _parse_solution_response(
        self,
        response: dict[str, any],
    ) -> list[PatchCandidate]:
        """Parse OpenCode's structured JSON response to extract solution patches."""
        try:
            # Extract content from OpenCode response
            solution_data = self._extract_and_parse_solution_content(response)
            if not solution_data:
                return []

            # Check if this is a direct operations response
            if solution_data.get("approach") == "direct_operations":
                return await self._handle_direct_operations(solution_data)
            
            # Handle traditional patch-based response
            return self._handle_patch_based_response(solution_data)
            
        except Exception as e:
            logger.error(f"Failed to parse OpenCode solution response: {e}")
            logger.debug(f"Raw response: {response}")
            return []

    def _extract_and_parse_solution_content(
        self, response: dict[str, any]
    ) -> dict[str, any] | None:
        """Extract and parse solution content from OpenCode response."""
        try:
            # For structured output, try direct JSON parsing first
            if isinstance(response, dict) and ("patches" in response or "operations" in response):
                # Response is already structured JSON
                solution_data = response
                logger.debug("Using direct structured JSON response for solution")
                return solution_data
            else:
                # Extract content from OpenCode response format
                content = self._extract_content_from_opencode_response(response)
                if not content:
                    logger.error("No content found in OpenCode solution response")
                    return None

                # For structured output, content should be valid JSON
                if isinstance(content, str):
                    try:
                        solution_data = json.loads(content)
                        logger.debug("Parsed JSON from content string")
                        return solution_data
                    except json.JSONDecodeError:
                        # Fallback to regex extraction for non-structured responses
                        json_match = re.search(
                            r"```json\s*(\{.*?\})\s*```", content, re.DOTALL
                        )
                        if json_match:
                            solution_data = json.loads(json_match.group(1))
                            logger.debug("Extracted JSON from markdown code block")
                            return solution_data
                        else:
                            # Look for JSON object in the response
                            json_match = re.search(
                                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL
                            )
                            if json_match:
                                solution_data = json.loads(json_match.group(0))
                                logger.debug("Extracted JSON from response text")
                                return solution_data
                            else:
                                logger.error(
                                    "No valid JSON found in OpenCode solution response"
                                )
                                return None
                else:
                    return content
                    
        except Exception as e:
            logger.error(f"Failed to extract solution content: {e}")
            return None

    async def _handle_direct_operations(
        self, solution_data: dict[str, any]
    ) -> list[PatchCandidate]:
        """Handle direct file operations from agent response."""
        if not self.file_ops_service:
            logger.warning(
                "Direct operations requested but file operations service not available. "
                "Falling back to patch-based approach."
            )
            # Convert to patch format for compatibility
            return self._convert_operations_to_patches(solution_data)
        
        try:
            operations = solution_data.get("operations", [])
            solution_description = solution_data.get(
                "solution_description", "Direct file operations"
            )
            confidence_score = float(solution_data.get("confidence_score", 0.5))
            
            patch_candidates = []
            
            for i, operation in enumerate(operations):
                op_type = operation.get("type")
                file_path = operation.get("file_path")
                description = operation.get("description", "File operation")
                
                if not file_path:
                    logger.error(f"Missing file_path in operation {i}")
                    continue
                    
                try:
                    if op_type == "write_file":
                        content = operation.get("content", "")
                        await self.file_ops_service.write_file(file_path, content)
                        
                        # Create patch candidate for tracking with normalized file path
                        normalized_file_path = (
                            to_repo_relative(file_path, self.repository_path) 
                            if self.repository_path and Path(file_path).is_absolute()
                            else file_path
                        )
                        
                        patch = PatchCandidate(
                            content=content,
                            description=f"{solution_description} - {description}",
                            agent_id=self.agent_config.agent_id,
                            file_path=normalized_file_path,
                            line_start=0,
                            line_end=-1,  # Indicates full file replacement
                            confidence_score=confidence_score,
                            metadata={
                                "model": self.agent_config.model_name,
                                "temperature": self.agent_config.temperature,
                                "specialized_role": self.agent_config.specialized_role,
                                "opencode_session": self.session_id,
                                "operation_type": "direct_write",
                                "direct_operation": True,
                            },
                        )
                        patch_candidates.append(patch)
                        
                    elif op_type == "delete_file":
                        await self.file_ops_service.delete_file(file_path)
                        
                        # Create patch candidate for tracking with normalized file path
                        normalized_file_path = (
                            to_repo_relative(file_path, self.repository_path) 
                            if self.repository_path and Path(file_path).is_absolute()
                            else file_path
                        )
                        
                        patch = PatchCandidate(
                            content="",
                            description=f"{solution_description} - {description}",
                            agent_id=self.agent_config.agent_id,
                            file_path=normalized_file_path,
                            line_start=0,
                            line_end=-1,
                            confidence_score=confidence_score,
                            metadata={
                                "model": self.agent_config.model_name,
                                "temperature": self.agent_config.temperature,
                                "specialized_role": self.agent_config.specialized_role,
                                "opencode_session": self.session_id,
                                "operation_type": "direct_delete",
                                "direct_operation": True,
                            },
                        )
                        patch_candidates.append(patch)
                        
                    else:
                        logger.warning(f"Unknown operation type: {op_type}")
                        
                except (FileOperationError, SecurityViolationError) as e:
                    logger.error(f"File operation failed for {file_path}: {e}")
                    # Continue with other operations
                    
            logger.info(
                f"Successfully executed {len(patch_candidates)} direct operations"
            )
            return patch_candidates
            
        except Exception as e:
            logger.error(f"Failed to handle direct operations: {e}")
            return []
    
    def _handle_patch_based_response(
        self, solution_data: dict[str, any]
    ) -> list[PatchCandidate]:
        """Handle traditional patch-based response (existing logic)."""
        try:
            # Validate required fields
            if "patches" not in solution_data:
                logger.error("Missing 'patches' field in solution response")
                return []

            if "confidence_score" not in solution_data:
                logger.error("Missing 'confidence_score' field in solution response")
                return []

            # Extract solution metadata
            solution_description = solution_data.get(
                "solution_description", "No description provided"
            )
            overall_confidence = float(solution_data.get("confidence_score", 0.5))

            # Parse individual patches
            patch_candidates = []
            for i, patch_data in enumerate(solution_data["patches"]):
                # Validate required fields for each patch
                required_fields = [
                    "file_path",
                    "content",
                    "line_start",
                    "line_end",
                    "description",
                ]
                for field in required_fields:
                    if field not in patch_data:
                        logger.error(f"Missing required field in patch {i}: {field}")
                        continue

                # Create PatchCandidate with normalized file path
                file_path = patch_data["file_path"]
                normalized_file_path = (
                    to_repo_relative(file_path, self.repository_path) 
                    if self.repository_path and Path(file_path).is_absolute()
                    else file_path
                )
                
                patch = PatchCandidate(
                    content=patch_data["content"],
                    description=f"{solution_description} - {patch_data['description']}",
                    agent_id=self.agent_config.agent_id,
                    file_path=normalized_file_path,
                    line_start=int(patch_data["line_start"]),
                    line_end=int(patch_data["line_end"]),
                    confidence_score=overall_confidence,  # Use overall solution confidence
                    metadata={
                        "model": self.agent_config.model_name,
                        "temperature": self.agent_config.temperature,
                        "specialized_role": self.agent_config.specialized_role,
                        "opencode_session": self.session_id,
                        "solution_description": solution_description,
                        "solution_patch_index": i,
                        "total_patches_in_solution": len(solution_data["patches"]),
                        "direct_operation": False,
                    },
                )
                patch_candidates.append(patch)

            logger.info(
                f"Successfully parsed {len(patch_candidates)} patches from solution"
            )
            return patch_candidates
            
        except Exception as e:
            logger.error(f"Failed to handle patch-based response: {e}")
            return []
    
    def _convert_operations_to_patches(
        self, solution_data: dict[str, any]
    ) -> list[PatchCandidate]:
        """Convert direct operations to patch format for compatibility.
        
        Used as fallback when direct operations are requested but 
        file operations service is not available.
        """
        try:
            operations = solution_data.get("operations", [])
            solution_description = solution_data.get(
                "solution_description", "Converted from direct operations"
            )
            confidence_score = float(solution_data.get("confidence_score", 0.5))
            
            patch_candidates = []
            
            for i, operation in enumerate(operations):
                op_type = operation.get("type")
                file_path = operation.get("file_path")
                description = operation.get("description", "File operation")
                
                if not file_path:
                    logger.error(f"Missing file_path in operation {i}")
                    continue
                    
                if op_type == "write_file":
                    content = operation.get("content", "")
                    
                    # Create patch candidate for full file replacement with normalized file path
                    normalized_file_path = (
                        to_repo_relative(file_path, self.repository_path) 
                        if self.repository_path and Path(file_path).is_absolute()
                        else file_path
                    )
                    
                    patch = PatchCandidate(
                        content=content,
                        description=f"{solution_description} - {description}",
                        agent_id=self.agent_config.agent_id,
                        file_path=normalized_file_path,
                        line_start=0,
                        line_end=-1,  # Indicates full file replacement
                        confidence_score=confidence_score,
                        metadata={
                            "model": self.agent_config.model_name,
                            "temperature": self.agent_config.temperature,
                            "specialized_role": self.agent_config.specialized_role,
                            "opencode_session": self.session_id,
                            "operation_type": "converted_write",
                            "direct_operation": False,
                            "original_operation_type": op_type,
                        },
                    )
                    patch_candidates.append(patch)
                    
                elif op_type == "delete_file":
                    # Create patch candidate for file deletion with normalized file path
                    normalized_file_path = (
                        to_repo_relative(file_path, self.repository_path) 
                        if self.repository_path and Path(file_path).is_absolute()
                        else file_path
                    )
                    
                    patch = PatchCandidate(
                        content="",
                        description=f"{solution_description} - {description}",
                        agent_id=self.agent_config.agent_id,
                        file_path=normalized_file_path,
                        line_start=0,
                        line_end=-1,
                        confidence_score=confidence_score,
                        metadata={
                            "model": self.agent_config.model_name,
                            "temperature": self.agent_config.temperature,
                            "specialized_role": self.agent_config.specialized_role,
                            "opencode_session": self.session_id,
                            "operation_type": "converted_delete",
                            "direct_operation": False,
                            "original_operation_type": op_type,
                        },
                    )
                    patch_candidates.append(patch)
                    
            logger.info(
                f"Converted {len(patch_candidates)} operations to patches"
            )
            return patch_candidates
            
        except Exception as e:
            logger.error(f"Failed to convert operations to patches: {e}")
            return []

            # Validate required fields
            if "patches" not in solution_data:
                logger.error("Missing 'patches' field in solution response")
                return []

            if "confidence_score" not in solution_data:
                logger.error("Missing 'confidence_score' field in solution response")
                return []

            # Extract solution metadata
            solution_description = solution_data.get(
                "solution_description", "No description provided"
            )
            overall_confidence = float(solution_data.get("confidence_score", 0.5))

            # Parse individual patches
            patch_candidates = []
            for i, patch_data in enumerate(solution_data["patches"]):
                # Validate required fields for each patch
                required_fields = [
                    "file_path",
                    "content",
                    "line_start",
                    "line_end",
                    "description",
                ]
                for field in required_fields:
                    if field not in patch_data:
                        logger.error(f"Missing required field in patch {i}: {field}")
                        continue

                # Create PatchCandidate
                patch = PatchCandidate(
                    content=patch_data["content"],
                    description=f"{solution_description} - {patch_data['description']}",
                    agent_id=self.agent_config.agent_id,
                    file_path=patch_data["file_path"],
                    line_start=int(patch_data["line_start"]),
                    line_end=int(patch_data["line_end"]),
                    confidence_score=overall_confidence,  # Use overall solution confidence
                    metadata={
                        "model": self.agent_config.model_name,
                        "temperature": self.agent_config.temperature,
                        "specialized_role": self.agent_config.specialized_role,
                        "opencode_session": self.session_id,
                        "solution_description": solution_description,
                        "solution_patch_index": i,
                        "total_patches_in_solution": len(solution_data["patches"]),
                        "raw_response": content,
                        "opencode_response": response,
                    },
                )
                patch_candidates.append(patch)

            logger.info(
                f"Successfully parsed {len(patch_candidates)} patches from solution"
            )
            return patch_candidates

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse OpenCode solution response: {e}")
            logger.debug(f"Raw response: {response}")
            return []

    def _extract_content_from_opencode_response(self, response: dict[str, any]) -> str:
        """Extract text content from OpenCode response format.

        OpenCode returns responses with this structure:
        {
            "info": {...},
            "parts": [
                {"type": "text", "text": "actual content here", ...},
                ...
            ]
        }
        """
        try:
            # Extract text from all text parts
            if isinstance(response, dict) and "parts" in response:
                text_content = []
                for part in response["parts"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            text_content.append(text)

                combined_text = "\n".join(text_content).strip()
                if combined_text:
                    logger.debug(
                        f"Extracted {len(combined_text)} characters from OpenCode parts"
                    )
                    return combined_text

            # Fallback to old format
            content = (
                response.get("content")
                or response.get("text")
                or response.get("response", "")
            )
            if content:
                logger.debug("Using fallback content extraction")
                return content

            logger.warning("No text content found in OpenCode response")
            return ""

        except Exception as e:
            logger.error(f"Failed to extract content from OpenCode response: {e}")
            return ""

    def get_agent_stats(self) -> dict[str, any]:
        """Get statistics about this agent's performance."""
        return {
            "agent_id": self.agent_config.agent_id,
            "model": self.agent_config.model_name,
            "specialized_role": self.agent_config.specialized_role,
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
        }
