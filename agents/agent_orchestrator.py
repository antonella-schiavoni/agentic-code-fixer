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
from core.lsp_analyzer import TargetType
from core.patch_validator import LSPPatchValidator, ValidationResult
from core.role_manager import RoleManager
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

    def __init__(self, config: Config, code_indexer: CodeIndexer) -> None:
        """Initialize the agent orchestrator with system configuration and code indexer.

        Sets up the OpenCode client, initializes all configured agents, and
        prepares the session-based orchestration infrastructure. LLM provider
        authentication is managed through OpenCode's auth system.

        Args:
            config: Complete system configuration including agent definitions
                and orchestration parameters.
            code_indexer: Code indexer instance for context retrieval during
                patch generation. Stored as instance variable to avoid repeated
                parameter passing.
        """
        self.config = config
        self.opencode_config = config.opencode
        self._code_indexer = code_indexer
        self._indices_dirty = False
        self.agents: list[OpenCodeAgent] = []
        self.active_sessions: dict[str, OpenCodeSession] = {}
        self.role_manager = RoleManager()
        self._background_tasks: set = set()  # Store background tasks to prevent GC

        # Initialize OpenCode client if enabled
        if self.opencode_config.enabled:
            self.opencode_client = OpenCodeClient(self.opencode_config)
            self.patch_validator = LSPPatchValidator(self.opencode_client)
        else:
            self.opencode_client = None
            self.patch_validator = None

        # Initialize agents
        self._initialize_agents()

        logger.info(
            f"Initialized orchestrator with {len(self.agents)} agents using OpenCode SST"
        )

    def _initialize_agents(self) -> None:
        """Initialize all configured agents."""
        for agent_config in self.config.agents:
            agent = OpenCodeAgent(
                agent_config=agent_config,
                opencode_config=self.opencode_config,
                role_manager=self.role_manager,
            )
            self.agents.append(agent)

    async def generate_patches(
        self,
        problem_description: str,
    ) -> list[PatchCandidate]:
        """Generate patch candidates using all agents with OpenCode session management.

        Uses the internal code indexer for context retrieval during patch generation.
        """
        logger.info("Starting patch generation with all agents")

        # Initialize OpenCode sessions if enabled
        if self.opencode_config.use_sessions and self.opencode_client:
            await self._initialize_agent_sessions(
                problem_description=problem_description,
                repository_path=self.config.repository_path,
            )

        try:
            # Get relevant code contexts with enhanced filtering
            relevant_contexts = self._code_indexer.search_relevant_context(
                problem_description=problem_description,
                top_k=10,  # TODO: We may need to increase this number
                # Enhanced metadata filtering is now available:
                # language_filter="python" - filter by specific language
                # function_filter="parse_config" - find specific functions
                # dependency_filter="fastapi" - find code using specific imports
                # content_size_range=(100, 2000) - filter by code size
                # languages=["python", "typescript"] - multiple languages
                # file_patterns=["/api/", "/core/"] - filter by path patterns
            )

            if not relevant_contexts:
                logger.warning("No relevant code contexts found")
                return []

            # Determine target files from relevant contexts
            target_files = list(set(ctx.file_path for ctx in relevant_contexts))

            logger.info(f"Generating patches for {len(target_files)} files")

            # Generate solutions using all agents (solution-based approach)
            all_patches = []
            tasks = []

            # Each agent generates a complete solution that may span multiple files
            for agent in self.agents:
                task = self._generate_solution_with_agent(
                    agent=agent,
                    problem_description=problem_description,
                    relevant_contexts=relevant_contexts,  # Pass all contexts, not per-file
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
                    asyncio.gather(
                        *[bounded_task(task) for task in tasks], return_exceptions=True
                    ),
                    timeout=self.opencode_config.session_timeout_seconds,
                )

                # Collect successful patches from solutions
                for result in results:
                    if isinstance(result, list):
                        # Result is a list of patches from solution-based generation
                        all_patches.extend(result)
                        if result:
                            logger.info(
                                f"Agent generated solution with {len(result)} patches"
                            )
                    elif isinstance(result, PatchCandidate):
                        # Backward compatibility for single patch results
                        all_patches.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Agent task failed: {result}")
                    elif result is None or (
                        isinstance(result, list) and len(result) == 0
                    ):
                        logger.warning("Agent returned no patches")

            except TimeoutError:
                logger.error(
                    f"Patch generation timed out after {self.opencode_config.session_timeout_seconds}s"
                )

            # Sort patches by confidence score
            all_patches.sort(key=lambda p: p.confidence_score, reverse=True)

            # Limit to requested number of candidates
            final_patches = all_patches[: self.config.num_candidate_solutions]

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

    async def _generate_solution_with_agent(
        self,
        agent: OpenCodeAgent,
        problem_description: str,
        relevant_contexts: list[CodeContext],
    ) -> list[PatchCandidate]:
        """Generate a complete solution using a specific agent.

        This method uses the new solution-based approach where agents can generate
        multiple patches that work together cohesively across multiple files.

        Args:
            agent: The agent to use for solution generation.
            problem_description: Description of the problem to solve.
            relevant_contexts: All relevant code contexts for the problem.

        Returns:
            List of patches representing a complete solution, or empty list if failed.
        """
        try:
            return await agent.generate_solution(
                problem_description=problem_description,
                relevant_contexts=relevant_contexts,
            )
        except Exception as e:
            logger.error(
                f"Agent {agent.agent_config.agent_id} failed to generate solution: {e}"
            )
            return []

    async def generate_diverse_patches(
        self,
        problem_description: str,
    ) -> list[PatchCandidate]:
        """Generate diverse patch candidates by varying agent parameters.

        Uses the internal code indexer for context retrieval during patch generation.
        """
        logger.info("Generating diverse patches with parameter variation")

        # Generate base patches
        base_patches = await self.generate_patches(
            problem_description=problem_description,
        )

        # Create variations with different parameters
        diverse_patches = []
        diverse_patches.extend(base_patches)

        # Generate additional patches with varied temperature
        for agent in self.agents:
            # Create agent variants with different temperatures
            for temp_offset in [-0.2, 0.2]:
                new_temp = max(
                    0.0, min(2.0, agent.agent_config.temperature + temp_offset)
                )
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
                        role_manager=self.role_manager,
                    )

                    # Generate patches with variant
                    relevant_contexts = self._code_indexer.search_relevant_context(
                        problem_description=problem_description, top_k=5
                    )

                    files_to_process = list(
                        set(ctx.file_path for ctx in relevant_contexts)
                    )

                    for target_file in files_to_process[
                        :2
                    ]:  # Limit to avoid too many variants
                        file_contexts = [
                            ctx
                            for ctx in relevant_contexts
                            if ctx.file_path == target_file
                        ]
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
        final_diverse_patches = unique_patches[: self.config.num_candidate_solutions]

        logger.info(f"Generated {len(final_diverse_patches)} diverse patch candidates")
        return final_diverse_patches

    def _deduplicate_patches(
        self, patches: list[PatchCandidate]
    ) -> list[PatchCandidate]:
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

    async def validate_patches(
        self, patches: list[PatchCandidate]
    ) -> list[PatchCandidate]:
        """Validate patch candidates using both LSP semantic validation and syntax checks."""
        validated_patches = []

        # First, perform LSP semantic validation if available
        if self.patch_validator and self.opencode_client:
            try:
                # Get the first active session for validation
                session_id = None
                for session in self.active_sessions.values():
                    session_id = session.session_id
                    break

                if session_id:
                    logger.info(
                        f"Performing LSP semantic validation on {len(patches)} patches"
                    )
                    validation_reports = await self.patch_validator.validate_patches(
                        session_id, patches
                    )

                    # Process validation results
                    corrected_patches = []
                    for _i, (patch, report) in enumerate(
                        zip(patches, validation_reports, strict=False)
                    ):
                        if report.result == ValidationResult.VALID:
                            validated_patches.append(patch)
                            logger.info(f"✓ Patch {patch.id} passed LSP validation")
                        elif report.suggested_range:
                            # Try to create a corrected patch
                            inferred_target_type = (
                                self._infer_target_type_from_description(
                                    patch.description
                                )
                            )
                            corrected_patch = (
                                await self.patch_validator.suggest_corrected_patch(
                                    session_id, patch, inferred_target_type
                                )
                            )
                            if corrected_patch:
                                corrected_patches.append(corrected_patch)
                                logger.info(
                                    f"🔧 Created corrected patch for {patch.id}"
                                )
                            else:
                                logger.warning(
                                    f"❌ Patch {patch.id} failed LSP validation: {report.message}"
                                )
                        else:
                            logger.warning(
                                f"❌ Patch {patch.id} failed LSP validation: {report.message}"
                            )

                    # Add corrected patches to the validated set
                    validated_patches.extend(corrected_patches)

                    logger.info(
                        f"LSP validation: {len(validated_patches)} valid/corrected out of {len(patches)} total patches"
                    )
                else:
                    logger.warning(
                        "No active OpenCode session available for LSP validation"
                    )

            except Exception as e:
                logger.error(f"LSP validation failed: {e}")
                logger.warning("Falling back to basic syntax validation")

        # Fall back to basic syntax validation for patches that haven't been validated yet
        remaining_patches = patches if not validated_patches else []

        for patch in remaining_patches:
            # Determine language from file extension
            file_ext = patch.file_path.split(".")[-1].lower()
            language_map = {
                "py": "python",
                "js": "javascript",
                "ts": "typescript",
                "java": "java",
                "cpp": "cpp",
                "c": "c",
            }
            language = language_map.get(file_ext, "unknown")

            # Find the agent that generated this patch
            agent = next(
                (a for a in self.agents if a.agent_config.agent_id == patch.agent_id),
                None,
            )

            if agent:
                is_valid = await agent.validate_patch_syntax(patch, language)
                if is_valid:
                    validated_patches.append(patch)
                else:
                    logger.warning(f"Patch {patch.id} failed syntax validation")
            else:
                # If agent not found, assume valid
                validated_patches.append(patch)

        logger.info(
            f"Final validation result: {len(validated_patches)} out of {len(patches)} patches passed"
        )
        return validated_patches

    def _infer_target_type_from_description(self, description: str) -> TargetType:
        """Infer target type from patch description for validation."""
        description_lower = description.lower()

        if any(
            keyword in description_lower
            for keyword in ["docstring", "documentation", "doc", "description"]
        ):
            return TargetType.DOCSTRING
        elif any(
            keyword in description_lower
            for keyword in ["implementation", "logic", "bug", "fix", "body"]
        ):
            return TargetType.FUNCTION_BODY
        else:
            return TargetType.ENTIRE_FUNCTION

    async def _initialize_agent_sessions(
        self, problem_description: str, repository_path: str
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
                    problem_description=problem_description,
                )
                session.metadata.update(
                    {
                        "agent_id": agent.agent_config.agent_id,
                        "specialized_role": agent.agent_config.specialized_role,
                    }
                )
                self.active_sessions[agent.agent_config.agent_id] = session

                # Assign session ID to the agent
                agent.set_session_id(session.session_id)

                logger.info(
                    f"Initialized session {session.session_id} for agent {agent.agent_config.agent_id}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize session for agent {agent.agent_config.agent_id}: {e}"
                )
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception details: {e!s}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")

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

    async def _update_indices_if_needed(self) -> None:
        """Update vector database indices if changes have occurred.

        Checks the internal dirty flag and updates the code indexer's vector
        database if any code modifications were made during patch operations.
        Resets the dirty flag after successful update.
        """
        if not self._indices_dirty:
            return

        try:
            # Check if the indexer has an update method
            if hasattr(self._code_indexer, "update_repository"):
                await self._code_indexer.update_repository(
                    repo_path=self.config.repository_path,
                    exclude_patterns=getattr(self.config, "exclude_patterns", []),
                )
                logger.info("Updated vector database indices after code changes")
            elif hasattr(self._code_indexer, "reindex_repository"):
                # Fallback to full reindexing if update not available
                await self._code_indexer.reindex_repository(
                    repo_path=self.config.repository_path,
                    exclude_patterns=getattr(self.config, "exclude_patterns", []),
                )
                logger.info("Re-indexed vector database after code changes")
            else:
                logger.warning("Code indexer does not support updating indices")

            self._indices_dirty = False

        except Exception as e:
            logger.error(f"Failed to update vector database indices: {e}")
            # Keep the dirty flag set for potential retry

    def flush_indices(self) -> None:
        """Public method to trigger index updates if needed.

        This method can be called externally to ensure any pending
        index updates are applied.
        """
        import asyncio

        # Create task and store reference to avoid garbage collection
        task = asyncio.create_task(self._update_indices_if_needed())
        self._background_tasks.add(task)

        # Remove task from set when it's done to avoid memory leak
        task.add_done_callback(self._background_tasks.discard)

    def mark_indices_dirty(self) -> None:
        """Mark indices as needing update due to code changes.

        Should be called whenever code modifications are persisted
        that would affect the vector database.
        """
        self._indices_dirty = True
        logger.debug("Marked vector database indices as dirty")

    def get_targeted_context(
        self,
        problem_description: str,
        focus_area: str | None = None,
        language_preference: str | None = None,
        top_k: int = 10,
    ) -> list[CodeContext]:
        """Get targeted code context using enhanced metadata filtering.

        This method provides intelligent context retrieval by analyzing the
        problem description and applying appropriate filters.

        Args:
            problem_description: Description of the problem to solve.
            focus_area: Area to focus on (e.g., "api", "core", "tests").
            language_preference: Preferred programming language.
            top_k: Maximum number of contexts to return.

        Returns:
            List of targeted CodeContext objects.
        """
        # Build filters based on focus area
        file_patterns = None
        if focus_area:
            file_patterns = [f"/{focus_area}/", f"{focus_area}_", f".{focus_area}"]

        # Smart language detection from problem description
        languages = None
        if language_preference:
            languages = [language_preference]
        elif any(
            keyword in problem_description.lower()
            for keyword in ["python", "py", "django", "flask", "fastapi"]
        ):
            languages = ["python"]
        elif any(
            keyword in problem_description.lower()
            for keyword in ["javascript", "js", "typescript", "ts", "react", "node"]
        ):
            languages = ["javascript", "typescript"]
        elif any(
            keyword in problem_description.lower()
            for keyword in ["java", "spring", "junit"]
        ):
            languages = ["java"]

        # Smart function detection from problem description
        function_filter = None
        if "function " in problem_description.lower():
            # Try to extract function name from description
            import re

            func_match = re.search(
                r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                problem_description,
                re.IGNORECASE,
            )
            if func_match:
                function_filter = func_match.group(1)

        # Use enhanced search with intelligent filters
        return self._code_indexer.search_relevant_context(
            problem_description=problem_description,
            top_k=top_k,
            languages=languages,
            file_patterns=file_patterns,
            function_filter=function_filter,
        )

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
            },
        }

        return stats
