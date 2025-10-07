"""Configuration management and validation for the Agentic Code Fixer system.

This module provides comprehensive configuration handling for all components
of the patch generation pipeline. It defines Pydantic models for type-safe
configuration validation and supports YAML configuration files.

The configuration system is designed with sensible defaults while allowing
fine-grained control over every aspect of the system's behavior, from agent
parameters to evaluation methods and testing procedures.

Key configuration areas include:
- AI agent setup and specialization
- Vector database and embedding configuration
- Patch evaluation methodology
- Test execution and validation
- Logging and experiment tracking
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from core.types import AgentConfig, EvaluationMethod


class VectorDBConfig(BaseModel):
    """Configuration for the vector database used in code indexing and retrieval.

    This configuration controls how source code is processed, embedded, and
    stored for semantic similarity search. The vector database enables efficient
    retrieval of relevant code contexts based on problem descriptions.

    Attributes:
        provider: Vector database backend ('chromadb' is currently supported).
        collection_name: Name of the collection to store code embeddings.
        persist_directory: Local directory for persistent storage of the database.
        embedding_model: HuggingFace model name for generating code embeddings.
        chunk_size: Maximum characters per code chunk for embedding.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
    """

    provider: str = "chromadb"
    collection_name: str = "code_embeddings"
    persist_directory: str = "./data/vectordb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50


class OpenCodeConfig(BaseModel):
    """Configuration for OpenCode SST integration and agent orchestration.

    This configuration controls the integration with OpenCode SST framework
    for managing multiple AI agents in parallel during patch generation.
    OpenCode SST provides the infrastructure for coordinating autonomous
    agents working on the same problem through session-based workflows.

    OpenCode manages LLM provider authentication and API calls, eliminating
    the need for direct LLM API management in our codebase.

    Attributes:
        enabled: Whether to use OpenCode SST for agent orchestration.
        server_host: OpenCode SST server hostname.
        server_port: OpenCode SST server port.
        base_url: Optional custom OpenCode SST service endpoint.
        use_sessions: Whether to use OpenCode's session management for agents.
        session_timeout_seconds: Maximum time allowed for session execution.
        max_parallel_sessions: Maximum number of concurrent OpenCode sessions.
        enable_shell_execution: Whether to use OpenCode's shell execution for tests.
        enable_code_analysis: Whether to use OpenCode's built-in code analysis.
        enable_event_streaming: Whether to enable OpenCode's real-time event streaming.
        enable_direct_file_ops: Whether to enable AI agents to write files directly.
        provider_name: LLM provider to use ("anthropic", "openai", "opencode").
        use_provider_auth: Whether to use OpenCode's provider authentication system.
    """

    enabled: bool = True
    server_host: str = "127.0.0.1"
    server_port: int = 4096
    base_url: str | None = None
    use_sessions: bool = True
    session_timeout_seconds: int = 600
    max_parallel_sessions: int = 4
    enable_shell_execution: bool = True
    enable_code_analysis: bool = True
    enable_event_streaming: bool = True
    enable_direct_file_ops: bool = False  # Enable direct file operations via agents
    provider_name: str = "anthropic"
    use_provider_auth: bool = True

    @property
    def server_url(self) -> str:
        """Get the complete OpenCode SST server URL."""
        if self.base_url:
            return self.base_url
        return f"http://{self.server_host}:{self.server_port}"


class EvaluationConfig(BaseModel):
    """Configuration for patch evaluation and ranking algorithms.

    This configuration controls how patch candidates are compared and ranked
    to determine the best solution. It supports both AB testing and ELO
    tournament methods with customizable parameters for different evaluation
    strategies.

    Attributes:
        method: Evaluation algorithm (AB_TESTING or ELO_TOURNAMENT).
        model_name: Language model to use for patch comparisons.
        temperature: Sampling temperature for evaluation consistency.
        max_tokens: Maximum response length for evaluation reasoning.
        elo_k_factor: K-factor for ELO rating updates (affects rating volatility).
        min_comparisons_per_patch: Minimum comparisons needed per patch.
        confidence_threshold: Minimum confidence required for evaluation results.
    """
    
    model_config = {"protected_namespaces": ()}

    method: EvaluationMethod = EvaluationMethod.ELO_TOURNAMENT
    model_name: str
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = 2048
    elo_k_factor: int = 32
    min_comparisons_per_patch: int = 3
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class TestingConfig(BaseModel):
    """Configuration for testing patches."""

    test_command: str = "pytest"
    test_timeout_seconds: int = 300
    pre_test_commands: list[str] = Field(default_factory=list)
    post_test_commands: list[str] = Field(default_factory=list)
    required_coverage: float | None = None
    fail_on_regression: bool = True


class LoggingConfig(BaseModel):
    """Configuration for logging and reporting."""

    level: str = "INFO"
    output_dir: str = "./experiments"
    log_file: str = "agentic_code_fixer.log"
    save_patches: bool = True
    save_evaluations: bool = True
    save_test_results: bool = True
    console_output: bool = True


class Config(BaseModel):
    """Main configuration for Agentic Code Fixer."""

    # Repository settings
    repository_path: str = Field(description="Path to the target repository")
    problem_description: str = Field(
        default="", 
        description="Description of the bug/issue to fix (overridden by --input CLI parameter)"
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*.pyc", "__pycache__", ".git", "node_modules"]
    )
    apply_patch_to_repository: bool = Field(
        default=True, 
        description="Whether to apply successful patches to the original repository"
    )

    # Agent configuration
    agents: list[AgentConfig] = Field(description="List of agent configurations")
    num_candidate_solutions: int = Field(default=10, gt=0)

    # Component configurations
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    opencode: OpenCodeConfig = Field(default_factory=OpenCodeConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Note: LLM API keys are now managed through OpenCode's auth system
    # Run `opencode auth login` to configure your LLM provider

    @field_validator("repository_path")
    @classmethod
    def validate_repository_path(cls, v: str) -> str:
        """Validate that repository path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Repository path is not a directory: {v}")
        return str(path.resolve())

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, v: list[AgentConfig]) -> list[AgentConfig]:
        """Validate that at least one agent is configured."""
        if not v:
            raise ValueError("At least one agent must be configured")
        return v

    def get_output_dir(self) -> Path:
        """Get the output directory for this experiment."""
        return Path(self.logging.output_dir)

    def ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.get_output_dir().mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix.lower() not in [".yaml", ".yml"]:
        raise ValueError(
            f"Unsupported config file format: {config_path.suffix}. "
            "Only YAML format (.yaml or .yml) is supported."
        )

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(**data)


def create_default_config(
    repository_path: str,
    problem_description: str,
    model_name: str,
    output_path: str | Path = "config.yaml",
) -> Config:
    """Create a default configuration file with the specified model name.

    Args:
        repository_path: Path to the target repository.
        problem_description: Description of the problem to fix.
        model_name: Claude model name to use for agents and evaluation.
        output_path: Path where the configuration file should be saved.

    Returns:
        Config object with the specified parameters.
    """
    default_agents = [
        AgentConfig(
            agent_id="general_fixer",
            model_name=model_name,
            temperature=0.7,
            system_prompt="You are a skilled software engineer focused on fixing bugs.",
            specialized_role="general",
        ),
        AgentConfig(
            agent_id="security_expert",
            model_name=model_name,
            temperature=0.5,
            system_prompt="You are a security expert focused on secure code fixes.",
            specialized_role="security",
        ),
        AgentConfig(
            agent_id="performance_optimizer",
            model_name=model_name,
            temperature=0.6,
            system_prompt="You are a performance expert focused on efficient solutions.",
            specialized_role="performance",
        ),
    ]

    config = Config(
        repository_path=repository_path,
        problem_description=problem_description,
        agents=default_agents,
        evaluation=EvaluationConfig(model_name=model_name),
    )

    # Save to file
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert to dict and save as YAML
        yaml.dump(config.model_dump(), f, default_flow_style=False, indent=2)

    return config
