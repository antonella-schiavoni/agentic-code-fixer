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

from core.config import OpenCodeConfig
from core.types import AgentConfig, CodeContext, PatchCandidate
from opencode_client import OpenCodeClient

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
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        opencode_config: OpenCodeConfig,
    ) -> None:
        """Initialize the OpenCode-powered patch generation agent.

        Args:
            agent_config: Individual agent configuration including model settings,
                specialized role, and behavioral parameters.
            opencode_config: OpenCode SST framework configuration for orchestration.
        """
        self.agent_config = agent_config
        self.opencode_config = opencode_config
        self.opencode_client = OpenCodeClient(opencode_config) if opencode_config.enabled else None
        self.session_id: str | None = None

        logger.info(f"Initialized OpenCode agent {agent_config.agent_id}")

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
            logger.error(f"Agent {self.agent_config.agent_id} has no active OpenCode session")
            return None

        try:
            # Prepare context for the agent
            context_text = self._format_contexts(relevant_contexts)

            # Create specialized prompt based on agent role
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(
                problem_description, context_text, target_file
            )

            # Generate patch using OpenCode's LLM provider management
            response = await self.opencode_client.send_prompt(
                session_id=self.session_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.agent_config.model_name,
                temperature=self.agent_config.temperature,
                max_tokens=self.agent_config.max_tokens,
                agent_id=self.agent_config.agent_id
            )

            # Parse response to extract patch
            patch = self._parse_opencode_response(response, target_file)

            if patch:
                logger.info(f"Agent {self.agent_config.agent_id} generated patch for {target_file}")
            else:
                logger.warning(f"Agent {self.agent_config.agent_id} failed to generate valid patch")

            return patch

        except Exception as e:
            logger.error(f"Agent {self.agent_config.agent_id} failed to generate patch: {e}")
            return None

    def set_session_id(self, session_id: str) -> None:
        """Set the OpenCode session ID for this agent.

        Args:
            session_id: OpenCode session ID to associate with this agent.
        """
        self.session_id = session_id
        logger.debug(f"Agent {self.agent_config.agent_id} assigned to session {session_id}")

    def _create_system_prompt(self) -> str:
        """Create system prompt based on agent configuration."""
        base_prompt = self.agent_config.system_prompt or "You are a skilled software engineer."

        role_specific_additions = {
            "security": """
                Pay special attention to security vulnerabilities and ensure any fixes
                don't introduce new security issues. Focus on input validation,
                authentication, authorization, and data sanitization.
            """,
            "performance": """
                Focus on optimizing code performance while maintaining correctness.
                Consider algorithm efficiency, memory usage, and runtime complexity.
                Avoid solutions that might degrade performance.
            """,
            "general": """
                Provide well-structured, maintainable solutions that follow best practices
                for the given programming language. Ensure code readability and proper error handling.
            """,
        }

        role_addition = role_specific_additions.get(
            self.agent_config.specialized_role, role_specific_additions["general"]
        )

        return f"""
{base_prompt}

{role_addition}

Your task is to generate a code patch that fixes the described problem.
You must provide:
1. The exact code to replace the problematic section
2. Clear line numbers for where the patch should be applied
3. A confidence score (0.0-1.0) for your solution
4. A brief description of what the patch does

Format your response as JSON with these fields:
- content: The replacement code
- line_start: Starting line number (0-indexed)
- line_end: Ending line number (0-indexed)
- confidence_score: Your confidence (0.0-1.0)
- description: Brief description of the fix

Example:
```json
{{
  "content": "def fixed_function():\\n    return 'fixed'",
  "line_start": 10,
  "line_end": 12,
  "confidence_score": 0.85,
  "description": "Fixed function logic and return value"
}}
```
        """.strip()

    def _create_user_prompt(
        self,
        problem_description: str,
        context_text: str,
        target_file: str,
    ) -> str:
        """Create user prompt with problem and context."""
        return f"""
Problem Description:
{problem_description}

Target File: {target_file}

Relevant Code Context:
{context_text}

Please analyze the problem and provide a patch to fix it. Focus on the specific
issue described while ensuring the solution is robust and doesn't break existing functionality.

Respond with a JSON object containing the patch information as specified in the system prompt.
        """.strip()

    def _format_contexts(self, contexts: list[CodeContext]) -> str:
        """Format code contexts for inclusion in prompt."""
        if not contexts:
            return "No relevant context available."

        formatted_contexts = []
        for i, context in enumerate(contexts):
            formatted_contexts.append(f"""
Context {i + 1} - {context.file_path} ({context.language}):
```{context.language}
{context.content}
```

Functions: {', '.join(context.relevant_functions) if context.relevant_functions else 'None'}
Dependencies: {', '.join(context.dependencies) if context.dependencies else 'None'}
            """.strip())

        return "\n\n".join(formatted_contexts)

    def _parse_opencode_response(
        self,
        response: dict[str, any],
        target_file: str,
    ) -> PatchCandidate | None:
        """Parse OpenCode's LLM response to extract patch information."""
        try:
            # OpenCode response format may vary, adapt as needed
            # Assuming response has 'content' or 'text' field with LLM output
            content = response.get("content") or response.get("text") or response.get("response", "")
            if not content:
                logger.error("No content found in OpenCode response")
                return None

            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                patch_data = json.loads(json_match.group(1))
            else:
                # Look for JSON object in the response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    patch_data = json.loads(json_match.group(0))
                else:
                    logger.error("No valid JSON found in OpenCode response")
                    return None

            # Validate required fields
            required_fields = ["content", "line_start", "line_end", "confidence_score", "description"]
            for field in required_fields:
                if field not in patch_data:
                    logger.error(f"Missing required field in patch response: {field}")
                    return None

            # Create PatchCandidate
            patch = PatchCandidate(
                content=patch_data["content"],
                description=patch_data["description"],
                agent_id=self.agent_config.agent_id,
                file_path=target_file,
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
        logger.warning("Using deprecated _parse_patch_response, use _parse_opencode_response instead")
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

    def get_agent_stats(self) -> dict[str, any]:
        """Get statistics about this agent's performance."""
        return {
            "agent_id": self.agent_config.agent_id,
            "model": self.agent_config.model_name,
            "specialized_role": self.agent_config.specialized_role,
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
        }