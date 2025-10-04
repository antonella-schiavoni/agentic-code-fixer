# Agentic Code Fixer

An automated code patch generation and evaluation system that leverages autonomous AI agents and advanced evaluation techniques to fix bugs in software repositories.

## Features

- **Autonomous AI Agents**: Uses multiple Claude Code agents with specialized roles to generate diverse patch candidates
- **Advanced Evaluation**: Implements AB testing and ELO tournament ranking to select the best patches
- **Vector-Based Code Search**: Indexes codebases using embeddings for intelligent context retrieval
- **Comprehensive Testing**: Automatically applies patches and runs tests to verify fixes
- **Rich Reporting**: Generates detailed experiment reports and statistics
- **Modular Architecture**: Clean, extensible design with clear component separation

## Architecture

The system consists of several key components:

1. **Configuration Module**: Manages runtime settings and parameters
2. **Indexing Module**: Scans codebases and generates vector embeddings
3. **Agent Orchestrator**: Coordinates multiple AI agents for patch generation
4. **Patch Manager**: Stores, labels, and tracks patch candidates
5. **Evaluation Module**: Compares patches using AB testing or ELO tournaments
6. **Patch Applicator**: Applies patches and runs tests
7. **Reporting System**: Logs experiments and generates comprehensive reports
8. **Coordinator**: Orchestrates the entire end-to-end workflow

## Installation

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

### 1. Set up your API key

```bash
export ANTHROPIC_API_KEY="your-claude-api-key"
```

### 2. Create a configuration file

```bash
agentic-code-fixer create-config \
  /path/to/your/repository \
  "Fix the authentication bug in the login module" \
  --output-path config.yaml
```

### 3. Run the experiment

```bash
agentic-code-fixer run config.yaml
```

### 4. Generate a report

```bash
agentic-code-fixer report ./experiments/your-experiment-id
```

## Configuration

Configuration files use YAML format. Here's a basic example:

```yaml
# Repository settings
repository_path: "/path/to/your/repository"
problem_description: "Fix the bug in the authentication module"
target_files:
  - "src/auth.py"
  - "src/login.py"

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

# Evaluation method
evaluation:
  method: "ab_testing"  # or "elo_tournament"
  confidence_threshold: 0.7

# Testing configuration
testing:
  test_command: "pytest"
  test_timeout_seconds: 300
```

See the `agentpatchai/examples/` directory for more configuration examples.

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
from agentpatchai import AgenticCodeFixer
from agentpatchai.core import load_config

# Load configuration
config = load_config("config.yaml")

# Initialize the system
fixer = AgenticCodeFixer(config)

# Run experiment
experiment_metadata = await fixer.run_experiment()

# Generate report
report_generator = fixer.generate_report()
report_generator.save_report("experiment_report.md")
```

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure `ANTHROPIC_API_KEY` is set correctly
2. **Test Failures**: Check that test commands work in your repository
3. **Memory Issues**: Reduce `num_patch_candidates` for large codebases
4. **Timeout Errors**: Increase `timeout_seconds` in the configuration

### Debug Mode

Enable debug logging for troubleshooting:

```yaml
logging:
  level: "DEBUG"
  console_output: true
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
