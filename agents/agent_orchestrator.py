"""Orchestration system for coordinating multiple AI agents in parallel.

This module implements the core orchestration logic for managing multiple
Claude-powered agents working together on patch generation. It handles agent
initialization, parallel execution, result aggregation, and resource management
within the OpenCode SST framework.

The orchestrator enables diverse solution generation by running multiple agents
with different specializations simultaneously, then collecting and managing
their patch candidates for downstream evaluation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import anthropic

from core.config import Config, OpenCodeConfig
from core.types import AgentConfig, CodeContext, PatchCandidate
from indexing import CodeIndexer
from agents.opencode_agent import OpenCodeAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Coordinates parallel execution of multiple AI agents for patch generation.

    This class manages the lifecycle and execution of multiple specialized AI agents
    working together to generate diverse patch candidates. It handles agent creation,
    parallel task execution, resource limiting, and result aggregation within the
    OpenCode SST framework.

    The orchestrator supports different agent specializations and ensures efficient
    resource utilization while maximizing solution diversity through parallel execution.

    Attributes:
        config: Main system configuration containing agent definitions.
        opencode_config: OpenCode SST specific configuration parameters.
        agents: List of initialized OpenCode agents ready for execution.
        claude_client: Shared Anthropic client for all agents.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the agent orchestrator with system configuration.

        Sets up the Anthropic client, initializes all configured agents, and
        prepares the orchestration infrastructure for parallel patch generation.

        Args:
            config: Complete system configuration including agent definitions,
                API credentials, and orchestration parameters.
        """
        self.config = config
        self.opencode_config = config.opencode
        self.agents: List[OpenCodeAgent] = []

        # Initialize Claude client
        self.claude_client = anthropic.AsyncAnthropic(
            api_key=config.claude_api_key
        )

        # Initialize agents
        self._initialize_agents()

        logger.info(f"Initialized orchestrator with {len(self.agents)} agents")

    def _initialize_agents(self) -> None:
        """Initialize all configured agents."""
        for agent_config in self.config.agents:
            agent = OpenCodeAgent(
                agent_config=agent_config,
                opencode_config=self.opencode_config,
                claude_client=self.claude_client,
            )
            self.agents.append(agent)

    async def generate_patches(
        self,
        problem_description: str,
        code_indexer: CodeIndexer,
        target_files: Optional[List[str]] = None,
    ) -> List[PatchCandidate]:
        """Generate patch candidates using all agents."""
        logger.info("Starting patch generation with all agents")

        # Get relevant code contexts
        relevant_contexts = code_indexer.search_relevant_context(
            problem_description=problem_description,
            top_k=10 #TODO: We may need to increase this number
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

        #TODO: Evaluate if this logic makes sense. Each agent generates a solution for a single file.
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
        semaphore = asyncio.Semaphore(self.opencode_config.max_parallel_agents)

        async def bounded_task(task):
            async with semaphore:
                return await task

        # Run tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[bounded_task(task) for task in tasks], return_exceptions=True),
                timeout=self.opencode_config.timeout_seconds
            )

            # Collect successful patches
            for result in results:
                if isinstance(result, PatchCandidate):
                    all_patches.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Agent task failed: {result}")

        except asyncio.TimeoutError:
            logger.error(f"Patch generation timed out after {self.opencode_config.timeout_seconds}s")

        # Sort patches by confidence score
        all_patches.sort(key=lambda p: p.confidence_score, reverse=True)

        # Limit to requested number of candidates
        final_patches = all_patches[:self.config.num_patch_candidates]

        logger.info(f"Generated {len(final_patches)} patch candidates")
        return final_patches

    async def _generate_patch_with_agent(
        self,
        agent: OpenCodeAgent,
        problem_description: str,
        relevant_contexts: List[CodeContext],
        target_file: str,
    ) -> Optional[PatchCandidate]:
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
        target_files: Optional[List[str]] = None,
        diversity_factor: float = 0.3,
    ) -> List[PatchCandidate]:
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
        final_diverse_patches = unique_patches[:self.config.num_patch_candidates]

        logger.info(f"Generated {len(final_diverse_patches)} diverse patch candidates")
        return final_diverse_patches

    def _deduplicate_patches(self, patches: List[PatchCandidate]) -> List[PatchCandidate]:
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

    async def validate_patches(self, patches: List[PatchCandidate]) -> List[PatchCandidate]:
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

    def get_orchestrator_stats(self) -> Dict[str, any]:
        """Get statistics about the orchestrator and its agents."""
        agent_stats = [agent.get_agent_stats() for agent in self.agents]

        return {
            "total_agents": len(self.agents),
            "max_parallel_agents": self.opencode_config.max_parallel_agents,
            "timeout_seconds": self.opencode_config.timeout_seconds,
            "target_patch_candidates": self.config.num_patch_candidates,
            "agents": agent_stats,
        }