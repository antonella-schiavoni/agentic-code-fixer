# Agentic Code Fixer

An automated code patch generation and evaluation system that leverages **OpenCode SST** and autonomous AI agents to fix bugs in software repositories. The system uses OpenCode's LLM provider management for secure, scalable AI interactions.

## Features

- **OpenCode SST Integration**: Leverages OpenCode's session management and LLM provider system for secure, scalable AI operations
- **Autonomous AI Agents**: Uses multiple specialized agents with different roles to generate diverse patch candidates
- **Multi-Provider Support**: Works with Claude, OpenAI, or OpenCode Zen through OpenCode's unified interface
- **Advanced Evaluation**: Implements AB testing and ELO tournament ranking to select the best patches
- **Vector-Based Code Search**: Indexes codebases using embeddings for intelligent context retrieval
- **Session-Based Isolation**: Each agent operates in its own OpenCode session for better resource management
- **Comprehensive Testing**: Automatically applies patches and runs tests to verify fixes
- **Rich Reporting**: Generates detailed experiment reports and statistics
- **Modular Architecture**: Clean, extensible design with clear component separation

## Architecture

The system consists of several key components integrated with **OpenCode SST**:

1. **OpenCode SST Integration**: Manages LLM provider authentication, sessions, and API routing
2. **Configuration Module**: Manages runtime settings and OpenCode provider configuration
3. **Indexing Module**: Scans codebases and generates vector embeddings for context retrieval
4. **Agent Orchestrator**: Coordinates multiple AI agents through OpenCode sessions
5. **Patch Manager**: Stores, labels, and tracks patch candidates with session metadata
6. **Evaluation Module**: Compares patches using LLM reasoning through OpenCode sessions
7. **Patch Applicator**: Applies patches and optionally runs tests through OpenCode shell execution
8. **Reporting System**: Logs experiments and generates comprehensive reports
9. **Coordinator**: Orchestrates the entire end-to-end workflow with session management

### OpenCode SST Benefits

- **Secure Authentication**: API keys managed through `opencode auth login`
- **Provider Flexibility**: Easy switching between Claude, OpenAI, or OpenCode Zen
- **Session Isolation**: Each agent operates in its own isolated session
- **Resource Management**: Built-in session lifecycle and resource tracking
- **Shell Integration**: Optional shell command execution through OpenCode sessions

## Installation

### Prerequisites

1. **Install OpenCode SST**: The system requires OpenCode SST for LLM provider management
   ```bash
   # Install OpenCode SST (follow their installation instructions)
   # https://opencode.ai/docs/installation
   ```

2. **Clone and Install Agentic Code Fixer**:
   ```bash
   # Clone the repository
   git clone https://github.com/your-org/agentic-code-fixer.git
   cd agentic-code-fixer

   # Install with uv (recommended)
   uv add .

   # Or install in development mode
   uv add -e .
   ```

## Quick Start

### 1. Configure OpenCode SST with your LLM provider

```bash
# Set up your LLM provider through OpenCode
opencode auth login

# Select your provider:
# - "anthropic" for Claude
# - "openai" for GPT models
# - "opencode" for OpenCode Zen

# Enter your API key when prompted
```

### 2. Start OpenCode SST server (if not already running)

```bash
# Start OpenCode server (runs on localhost:4096 by default)
opencode server start
```

### 3. Create a configuration file

```bash
agentic-code-fixer create-config \
  /path/to/your/repository \
  "Fix the authentication bug in the login module" \
  claude-3-5-sonnet-20241022 \
  --output-path config.yaml
```

### 4. Run the experiment

```bash
agentic-code-fixer run config.yaml
```

### 4. Generate a report

```bash
agentic-code-fixer report ./experiments/your-experiment-id
```

## Configuration

Configuration files use YAML format with OpenCode SST integration. Here's a basic example:

```yaml
# Repository settings
repository_path: "/path/to/your/repository"
problem_description: "Fix the bug in the authentication module"
target_files:
  - "src/auth.py"
  - "src/login.py"

# OpenCode SST configuration
opencode:
  enabled: true
  provider_name: "anthropic"  # or "openai", "opencode"
  use_provider_auth: true
  server_host: "127.0.0.1"
  server_port: 4096
  use_sessions: true
  enable_shell_execution: true

# Agent configurations
agents:
  - agent_id: "general_fixer"
    model_name: "claude-3-5-sonnet-20241022"
    temperature: 0.7
    specialized_role: "general"

  - agent_id: "security_expert"
    model_name: "claude-3-5-sonnet-20241022"
    temperature: 0.5
    specialized_role: "security"

# Evaluation configuration
evaluation:
  method: "ab_testing"  # or "elo_tournament"
  model_name: "claude-3-5-sonnet-20241022"
  confidence_threshold: 0.7

# Testing configuration
testing:
  test_command: "pytest"
  test_timeout_seconds: 300
```

### OpenCode Configuration Options

- **provider_name**: LLM provider ("anthropic", "openai", "opencode")
- **use_provider_auth**: Use OpenCode's authentication system (recommended: true)
- **use_sessions**: Enable session-based agent isolation (recommended: true)
- **enable_shell_execution**: Allow OpenCode to execute shell commands for testing
- **server_host/server_port**: OpenCode SST server connection details

### Multiple Provider Support

```yaml
# Use Claude through Anthropic
opencode:
  provider_name: "anthropic"

# Use GPT models through OpenAI
opencode:
  provider_name: "openai"

# Use OpenCode Zen (curated models)
opencode:
  provider_name: "opencode"
```

## Usage Examples

### Basic Bug Fix

```bash
# Create config for a simple Python bug
agentic-code-fixer create-config \
  ./my-python-project \
  "Fix the null pointer exception in user validation"

# Run the experiment
agentic-code-fixer run config.yaml
```

### Complex Multi-Agent Setup

```bash
# Use the advanced configuration template
cp agentpatchai/examples/advanced_config.yaml my_advanced_config.yaml

# Edit the config file for your specific repository
# Then run the experiment
agentic-code-fixer run my_advanced_config.yaml --output-dir ./my-experiments
```

### JavaScript/Node.js Projects

```bash
# Use the JavaScript-specific configuration
cp agentpatchai/examples/javascript_config.yaml js_config.yaml

# Customize for your Node.js project
# Run the experiment
agentic-code-fixer run js_config.yaml
```

## CLI Commands

### `run`
Run a complete patch generation experiment:
```bash
agentic-code-fixer run config.yaml [--output-dir ./experiments]
```

### `create-config`
Create a default configuration file:
```bash
agentic-code-fixer create-config REPO_PATH PROBLEM_DESCRIPTION [--output-path config.yaml]
```

### `report`
Generate reports from experiment data:
```bash
agentic-code-fixer report EXPERIMENT_DIR [--format markdown|json] [--output-file report.md]
```

### `validate-config`
Validate a configuration file:
```bash
agentic-code-fixer validate-config config.yaml
```

### `list-experiments`
List all experiments in a directory:
```bash
agentic-code-fixer list-experiments [--output-dir ./experiments]
```

### `baseline-test`
Run baseline tests on a repository:
```bash
agentic-code-fixer baseline-test REPO_PATH [--test-command pytest]
```

## Agent Specialization

The system supports different agent roles for diverse patch generation:

- **General**: Well-rounded fixes following best practices
- **Security**: Focus on security vulnerabilities and safe coding
- **Performance**: Optimization and efficiency improvements
- **Concurrency**: Threading and async programming fixes
- **Reliability**: Fault-tolerant and robust solutions

## Evaluation Methods

### AB Testing
Pairwise comparison of patches using Claude's evaluation capabilities:
- Direct head-to-head comparisons
- Confidence-weighted results
- Best patch selection based on win rates

### ELO Tournament
Chess-style ranking system for patch quality:
- Patches gain/lose rating points based on comparisons
- Converges to stable rankings over multiple rounds
- Suitable for large numbers of patch candidates

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run ruff format .
uv run ruff check . --fix
```

### Type Checking

```bash
uv run pyright
```

### Development Setup

```bash
# Install development dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff pyright

# Install pre-commit hooks
uv run pre-commit install
```

## Output Structure

Each experiment creates a structured output directory:

```
experiments/
└── experiment-id-timestamp/
    ├── experiment.json          # Complete experiment data
    ├── agentic_code_fixer.log   # Detailed logs
    ├── patches/                 # Individual patch files
    │   ├── patches.json
    │   ├── patch_abc123.json
    │   └── patches_summary.json
    ├── test_env/               # Test environment (if preserved)
    └── reports/                # Generated reports
```

## API Reference

### Core Classes

- `AgenticCodeFixer`: Main coordinator class
- `Config`: Configuration management
- `AgentOrchestrator`: Manages multiple AI agents
- `PatchEvaluator`: Handles patch comparison and selection
- `CodeIndexer`: Vector-based code search and retrieval

### Example Programmatic Usage

```python
from agentic_code_fixer import AgenticCodeFixer
from core import load_config

# Load configuration (includes OpenCode SST settings)
config = load_config("config.yaml")

# Initialize the system (connects to OpenCode SST)
fixer = AgenticCodeFixer(config)

# Run experiment (agents work through OpenCode sessions)
experiment_metadata = await fixer.run_experiment()

# Generate report
report_generator = fixer.generate_report()
report_generator.save_report("experiment_report.md")
```

### Advanced OpenCode Integration

```python
from opencode_client import OpenCodeClient
from core.config import OpenCodeConfig

# Direct OpenCode client usage
opencode_config = OpenCodeConfig(
    provider_name="anthropic",
    use_sessions=True
)

async with OpenCodeClient(opencode_config) as client:
    # Create a session
    session = await client.create_session(
        metadata={"purpose": "custom_patch_generation"}
    )

    # Send prompts directly to LLM through OpenCode
    response = await client.send_prompt(
        session_id=session.session_id,
        prompt="Analyze this code for bugs...",
        model="claude-3-5-sonnet-20241022"
    )

    # Execute shell commands in session
    result = await client.execute_shell_command(
        session_id=session.session_id,
        command="pytest tests/"
    )
```

## Why OpenCode SST Integration?

### Security Benefits
- **No API Keys in Code**: Credentials managed securely by OpenCode
- **Centralized Authentication**: Single auth point for all LLM providers
- **Session Isolation**: Each agent operates in its own secure environment

### Scalability Benefits
- **Resource Management**: OpenCode handles session lifecycle and cleanup
- **Provider Abstraction**: Switch between Claude, OpenAI, and others seamlessly
- **Concurrent Sessions**: Built-in support for parallel agent execution

### Developer Experience
- **Unified Interface**: Same API for all LLM providers
- **Built-in Logging**: Comprehensive session and request logging
- **Shell Integration**: Execute tests and commands within sessions
- **Real-time Monitoring**: Track session status and resource usage

### Production Ready
- **Battle Tested**: OpenCode SST is used in production environments
- **Error Handling**: Robust error recovery and retry mechanisms
- **Performance Optimized**: Efficient connection pooling and request routing

## Troubleshooting

### Common Issues

#### 1. **OpenCode SST Connection Issues**
```bash
# Check if OpenCode server is running
curl http://localhost:4096/health

# Start OpenCode server if not running
opencode server start

# Check OpenCode authentication
opencode auth status
```

#### 2. **Provider Authentication Issues**
```bash
# Re-authenticate with your LLM provider
opencode auth login

# Verify provider configuration
opencode providers list
```

#### 3. **Session Management Issues**
- Ensure `use_sessions: true` in your configuration
- Check OpenCode server logs for session errors
- Restart OpenCode server if sessions are stuck

#### 4. **Legacy Issues**
- **Test Failures**: Check that test commands work in your repository
- **Memory Issues**: Reduce `num_candidate_solutions` for large codebases
- **Timeout Errors**: Increase `session_timeout_seconds` in OpenCode configuration

### Debug Mode

Enable debug logging for troubleshooting:

```yaml
logging:
  level: "DEBUG"
  console_output: true

opencode:
  enabled: true
  use_sessions: true  # Enable for better debugging
```

### OpenCode SST Logs

Check OpenCode server logs for detailed session information:
```bash
# View OpenCode server logs
opencode logs

# Monitor sessions in real-time
opencode sessions list --watch
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding guidelines
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Agentic Code Fixer in your research, please cite:

```bibtex
@software{agentic_code_fixer,
  title={Agentic Code Fixer: Automated Patch Generation with AI Agents},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/agentic-code-fixer}
}
```
