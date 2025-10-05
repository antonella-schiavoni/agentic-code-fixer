"""Individual AI agent implementation for autonomous patch generation.

This module implements a Claude-powered agent that works within the OpenCode SST
framework to generate code patches. Each agent can have specialized roles and
behaviors to encourage diverse solution approaches to the same problem.

The agent takes problem descriptions and code context as input, then uses
Claude's reasoning capabilities to propose specific code changes that could
address the identified issues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional

import anthropic

from core.config import OpenCodeConfig
from core.types import AgentConfig, CodeContext, PatchCandidate

logger = logging.getLogger(__name__)


class OpenCodeAgent:
    """Autonomous AI agent for generating code patch candidates using Claude.

    This class represents a single AI agent that operates within the OpenCode SST
    framework to generate patch proposals for code issues. Each agent can have
    different specializations, model parameters, and prompting strategies to
    encourage solution diversity.

    The agent operates asynchronously and can work in parallel with other agents
    on the same problem, contributing to a pool of candidate solutions that are
    later evaluated and ranked.

    Attributes:
        agent_config: Configuration parameters for this specific agent.
        opencode_config: OpenCode SST framework configuration.
        client: Async Anthropic client for Claude API communication.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        opencode_config: OpenCodeConfig,
        claude_client: anthropic.AsyncAnthropic,
    ) -> None:
        """Initialize the Claude-powered patch generation agent.

        Args:
            agent_config: Individual agent configuration including model settings,
                specialized role, and behavioral parameters.
            opencode_config: OpenCode SST framework configuration for orchestration.
            claude_client: Pre-configured async Anthropic client for API calls.
        """
        self.agent_config = agent_config
        self.opencode_config = opencode_config
        self.client = claude_client

        logger.info(f"Initialized Claude agent {agent_config.agent_id}")

    async def generate_patch(
        self,
        problem_description: str,
        relevant_contexts: List[CodeContext],
        target_file: str,
    ) -> Optional[PatchCandidate]:
        """Generate a patch candidate to address a specific code problem.

        This method orchestrates the entire patch generation process for this agent.
        It formats the provided code context, creates specialized prompts based on
        the agent's role, queries Claude for a solution, and parses the response
        into a structured patch candidate.

        The agent uses its specialized role (e.g., security, performance, general)
        to focus on different aspects of the problem and propose diverse solutions.

        Args:
            problem_description: Human-readable description of the issue to fix.
            relevant_contexts: List of code context chunks relevant to the problem.
            target_file: Path to the file where the patch should be applied.

        Returns:
            A PatchCandidate object containing the proposed fix, or None if the
            agent was unable to generate a valid solution.

        Raises:
            Exception: If Claude API communication fails or response parsing errors.
        """
        try:
            # Prepare context for the agent
            context_text = self._format_contexts(relevant_contexts)

            # Create specialized prompt based on agent role
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(
                problem_description, context_text, target_file
            )

            # Generate patch using Claude

            #TODO: Force the agent to return a JSON object (also maybe add a schema)
            response = await self.client.messages.create(
                model=self.agent_config.model_name,
                max_tokens=self.agent_config.max_tokens,
                temperature=self.agent_config.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )

            # Parse response to extract patch
            patch = self._parse_patch_response(response, target_file)

            if patch:
                logger.info(f"Agent {self.agent_config.agent_id} generated patch for {target_file}")
            else:
                logger.warning(f"Agent {self.agent_config.agent_id} failed to generate valid patch")

            return patch

        except Exception as e:
            logger.error(f"Agent {self.agent_config.agent_id} failed to generate patch: {e}")
            return None

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

    def _format_contexts(self, contexts: List[CodeContext]) -> str:
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

    def _parse_patch_response(
        self,
        response: anthropic.types.Message,
        target_file: str,
    ) -> Optional[PatchCandidate]:
        """Parse the agent's response to extract patch information."""
        try:
            content = response.content[0].text if response.content else ""
            if not content:
                return None

            # Try to extract JSON from the response
            #TODO: This shouldnt be needed, we should force the agent to return a JSON object
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                patch_data = json.loads(json_match.group(1))
            else:
                # Look for JSON object in the response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    patch_data = json.loads(json_match.group(0))
                else:
                    logger.error("No valid JSON found in response")
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
                    "raw_response": content,
                },
            )

            return patch

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse patch response: {e}")
            logger.debug(f"Raw response: {content}")
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

    def get_agent_stats(self) -> Dict[str, any]:
        """Get statistics about this agent's performance."""
        return {
            "agent_id": self.agent_config.agent_id,
            "model": self.agent_config.model_name,
            "specialized_role": self.agent_config.specialized_role,
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
        }