"""Orchestration system for coordinating multiple AI agents in parallel.

This module implements the core orchestration logic for managing multiple
Claude-powered agents working together on patch generation. It leverages
OpenCode SST sessions for proper agent isolation, resource management,
and coordinated execution instead of custom orchestration.

The orchestrator creates isolated OpenCode sessions for each agent, enabling
proper resource management and leveraging OpenCode's built-in capabilities
for session management and execution coordination.
"""

from __future__ import annotations

import asyncio
import logging

from agents.opencode_agent import OpenCodeAgent
from core.config import Config
from core.types import AgentConfig, CodeContext, PatchCandidate
from indexing import CodeIndexer
from opencode_client import OpenCodeClient, OpenCodeSession

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Coordinates parallel execution of multiple AI agents for patch generation.

    This class manages the lifecycle and execution of multiple specialized AI agents
    working together to generate diverse patch candidates. It leverages OpenCode SST
    sessions for proper agent isolation and resource management, replacing custom
    orchestration with OpenCode's proven infrastructure.

    Each agent operates within its own OpenCode session, enabling proper isolation,
    resource tracking, and coordinated execution through OpenCode's session management.
    LLM provider authentication is handled by OpenCode's auth system.

    Attributes:
        config: Main system configuration containing agent definitions.
        opencode_config: OpenCode SST specific configuration parameters.
        agents: List of initialized OpenCode agents ready for execution.
        opencode_client: OpenCode SST client for session management.
        active_sessions: Dictionary of active OpenCode sessions by agent ID.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the agent orchestrator with system configuration.

        Sets up the OpenCode client, initializes all configured agents, and
        prepares the session-based orchestration infrastructure. LLM provider
        authentication is managed through OpenCode's auth system.

        Args:
            config: Complete system configuration including agent definitions
                and orchestration parameters.
        """
        self.config = config
        self.opencode_config = config.opencode
        self.agents: list[OpenCodeAgent] = []
        self.active_sessions: dict[str, OpenCodeSession] = {}

        # Initialize OpenCode client if enabled
        if self.opencode_config.enabled:
            self.opencode_client = OpenCodeClient(self.opencode_config)
        else:
            self.opencode_client = None

        # Initialize agents
        self._initialize_agents()

        logger.info(f"Initialized orchestrator with {len(self.agents)} agents using OpenCode SST")

    def _initialize_agents(self) -> None:
        """Initialize all configured agents."""
        for agent_config in self.config.agents:
            agent = OpenCodeAgent(
                agent_config=agent_config,
                opencode_config=self.opencode_config,
            )
            self.agents.append(agent)

    async def generate_patches(
        self,
        problem_description: str,
        code_indexer: CodeIndexer,
        target_files: list[str] | None = None,
    ) -> list[PatchCandidate]:
        """Generate patch candidates using all agents with OpenCode session management."""
        logger.info("Starting patch generation with all agents")

        # Initialize OpenCode sessions if enabled
        if self.opencode_config.use_sessions and self.opencode_client:
            await self._initialize_agent_sessions(
                problem_description=problem_description,
                repository_path=self.config.repository_path
            )

        try:
            # Get relevant code contexts
            relevant_contexts = code_indexer.search_relevant_context(
                problem_description=problem_description,
                top_k=10  # TODO: We may need to increase this number
            )

            if not relevant_contexts:
                logger.warning("No relevant code contexts found")
                return []

            # Determine target files
            if not target_files:
                target_files = list(set(ctx.file_path for ctx in relevant_contexts))

            logger.info(f"Generating patches for {len(target_files)} files")

            # Generate patches for each file using all agents
            all_patches = []
            tasks = []

            # TODO: Evaluate if this logic makes sense. Each agent generates a solution for a single file.
            for target_file in target_files:
                # Get contexts specific to this file
                file_contexts = [ctx for ctx in relevant_contexts if ctx.file_path == target_file]

                # Create tasks for each agent to generate patches for this file
                for agent in self.agents:
                    task = self._generate_patch_with_agent(
                        agent=agent,
                        problem_description=problem_description,
                        relevant_contexts=file_contexts,
                        target_file=target_file,
                    )
                    tasks.append(task)

            # Execute all tasks with concurrency limit
            max_parallel = self.opencode_config.max_parallel_sessions
            semaphore = asyncio.Semaphore(max_parallel)

            async def bounded_task(task):
                async with semaphore:
                    return await task

            # Run tasks with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*[bounded_task(task) for task in tasks], return_exceptions=True),
                    timeout=self.opencode_config.session_timeout_seconds
                )

                # Collect successful patches
                for result in results:
                    if isinstance(result, PatchCandidate):
                        all_patches.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Agent task failed: {result}")

            except TimeoutError:
                logger.error(f"Patch generation timed out after {self.opencode_config.session_timeout_seconds}s")

            # Sort patches by confidence score
            all_patches.sort(key=lambda p: p.confidence_score, reverse=True)

            # Limit to requested number of candidates
            final_patches = all_patches[:self.config.num_candidate_solutions]

            logger.info(f"Generated {len(final_patches)} patch candidates")
            return final_patches

        finally:
            # Clean up OpenCode sessions
            await self._cleanup_agent_sessions()

    async def _generate_patch_with_agent(
        self,
        agent: OpenCodeAgent,
        problem_description: str,
        relevant_contexts: list[CodeContext],
        target_file: str,
    ) -> PatchCandidate | None:
        """Generate a patch using a specific agent."""
        try:
            return await agent.generate_patch(
                problem_description=problem_description,
                relevant_contexts=relevant_contexts,
                target_file=target_file,
            )
        except Exception as e:
            logger.error(f"Agent {agent.agent_config.agent_id} failed: {e}")
            return None

    async def generate_diverse_patches(
        self,
        problem_description: str,
        code_indexer: CodeIndexer,
        target_files: list[str] | None = None,
        diversity_factor: float = 0.3,  # noqa: ARG002
    ) -> list[PatchCandidate]:
        """Generate diverse patch candidates by varying agent parameters."""
        logger.info("Generating diverse patches with parameter variation")

        # Generate base patches
        base_patches = await self.generate_patches(
            problem_description=problem_description,
            code_indexer=code_indexer,
            target_files=target_files,
        )

        # Create variations with different parameters
        diverse_patches = []
        diverse_patches.extend(base_patches)

        # Generate additional patches with varied temperature
        for agent in self.agents:
            # Create agent variants with different temperatures
            for temp_offset in [-0.2, 0.2]:
                new_temp = max(0.0, min(2.0, agent.agent_config.temperature + temp_offset))
                if new_temp != agent.agent_config.temperature:
                    # Create variant agent config
                    variant_config = AgentConfig(
                        agent_id=f"{agent.agent_config.agent_id}_temp_{new_temp}",
                        model_name=agent.agent_config.model_name,
                        temperature=new_temp,
                        max_tokens=agent.agent_config.max_tokens,
                        system_prompt=agent.agent_config.system_prompt,
                        specialized_role=agent.agent_config.specialized_role,
                    )

                    variant_agent = OpenCodeAgent(
                        agent_config=variant_config,
                        opencode_config=self.opencode_config,
                        claude_client=self.claude_client,
                    )

                    # Generate patches with variant
                    relevant_contexts = code_indexer.search_relevant_context(
                        problem_description=problem_description,
                        top_k=5
                    )

                    if target_files:
                        files_to_process = target_files
                    else:
                        files_to_process = list(set(ctx.file_path for ctx in relevant_contexts))

                    for target_file in files_to_process[:2]:  # Limit to avoid too many variants
                        file_contexts = [ctx for ctx in relevant_contexts if ctx.file_path == target_file]
                        variant_patch = await variant_agent.generate_patch(
                            problem_description=problem_description,
                            relevant_contexts=file_contexts,
                            target_file=target_file,
                        )
                        if variant_patch:
                            diverse_patches.append(variant_patch)

        # Remove duplicates and sort by confidence
        unique_patches = self._deduplicate_patches(diverse_patches)
        unique_patches.sort(key=lambda p: p.confidence_score, reverse=True)

        # Return top candidates
        final_diverse_patches = unique_patches[:self.config.num_candidate_solutions]

        logger.info(f"Generated {len(final_diverse_patches)} diverse patch candidates")
        return final_diverse_patches

    def _deduplicate_patches(self, patches: list[PatchCandidate]) -> list[PatchCandidate]:
        """Remove duplicate patches based on content similarity."""
        unique_patches = []
        seen_contents = set()

        for patch in patches:
            # Create a signature for the patch
            signature = f"{patch.file_path}:{patch.line_start}-{patch.line_end}:{hash(patch.content)}"

            if signature not in seen_contents:
                seen_contents.add(signature)
                unique_patches.append(patch)

        return unique_patches

    async def validate_patches(self, patches: list[PatchCandidate]) -> list[PatchCandidate]:
        """Validate patch candidates for syntax and basic correctness."""
        validated_patches = []

        for patch in patches:
            # Determine language from file extension
            file_ext = patch.file_path.split('.')[-1].lower()
            language_map = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'java': 'java',
                'cpp': 'cpp',
                'c': 'c',
            }
            language = language_map.get(file_ext, 'unknown')

            # Find the agent that generated this patch
            agent = next((a for a in self.agents if a.agent_config.agent_id == patch.agent_id), None)

            if agent:
                is_valid = await agent.validate_patch_syntax(patch, language)
                if is_valid:
                    validated_patches.append(patch)
                else:
                    logger.warning(f"Patch {patch.id} failed syntax validation")
            else:
                # If agent not found, assume valid
                validated_patches.append(patch)

        logger.info(f"Validated {len(validated_patches)} out of {len(patches)} patches")
        return validated_patches

    async def _initialize_agent_sessions(
        self,
        problem_description: str,
        repository_path: str
    ) -> None:
        """Initialize OpenCode sessions for each agent.

        Args:
            problem_description: Description of the problem to solve.
            repository_path: Path to the repository being worked on.
        """
        if not self.opencode_client:
            return

        for agent in self.agents:
            try:
                session = await self.opencode_client.initialize_session_for_repository(
                    repository_path=repository_path,
                    problem_description=problem_description
                )
                session.metadata.update({
                    "agent_id": agent.agent_config.agent_id,
                    "specialized_role": agent.agent_config.specialized_role
                })
                self.active_sessions[agent.agent_config.agent_id] = session

                # Assign session ID to the agent
                agent.set_session_id(session.session_id)

                logger.info(f"Initialized session {session.session_id} for agent {agent.agent_config.agent_id}")

            except Exception as e:
                logger.error(f"Failed to initialize session for agent {agent.agent_config.agent_id}: {e}")

    async def _cleanup_agent_sessions(self) -> None:
        """Clean up all active OpenCode sessions."""
        if not self.opencode_client or not self.active_sessions:
            return

        for agent_id, session in self.active_sessions.items():
            try:
                await self.opencode_client.delete_session(session.session_id)
                logger.info(f"Cleaned up session for agent {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session for agent {agent_id}: {e}")

        self.active_sessions.clear()

    def get_orchestrator_stats(self) -> dict[str, any]:
        """Get statistics about the orchestrator and its agents."""
        agent_stats = [agent.get_agent_stats() for agent in self.agents]

        stats = {
            "total_agents": len(self.agents),
            "target_candidate_solutions": self.config.num_candidate_solutions,
            "agents": agent_stats,
            "opencode_integration": {
                "sessions_enabled": self.opencode_config.use_sessions,
                "max_parallel_sessions": self.opencode_config.max_parallel_sessions,
                "session_timeout_seconds": self.opencode_config.session_timeout_seconds,
                "active_sessions": len(self.active_sessions),
                "shell_execution_enabled": self.opencode_config.enable_shell_execution,
                "code_analysis_enabled": self.opencode_config.enable_code_analysis,
            }
        }

        return stats