# Agent Roles

This directory contains role definitions for specialized AI agents. Each role provides specific prompting to focus agents on particular types of code issues and expertise areas.

## How It Works

Each YAML file defines a role with:
- **name**: Unique identifier for the role
- **description**: Human-readable description of the role's purpose
- **prompt_addition**: Additional prompt text that guides the agent's behavior
- **category**: Optional category for organization
- **priority**: Optional priority level (high, medium, low)
- **tags**: Optional list of tags for filtering and discovery

## Available Roles

Use the CLI to list all available roles:

```bash
# List all roles
uv run python cli.py list-roles

# Show detailed information
uv run python cli.py list-roles --verbose

# Filter by category
uv run python cli.py list-roles --category security
```

## Adding New Roles

To add a new role, create a YAML file in this directory:

```yaml
# Example: roles/my_new_role.yaml
name: "my_new_role"
description: "Agent focused on my specific expertise area"
prompt_addition: |
  Focus on my specific area of expertise. Consider specific patterns,
  best practices, and common issues in this domain. Pay attention to
  domain-specific requirements and optimization opportunities.

category: "my_category"
priority: "medium"
tags: ["tag1", "tag2", "tag3"]
```

## Usage in Configuration

Reference roles in your agent configuration:

```yaml
agents:
  - agent_id: "my_specialist"
    model_name: "claude-sonnet-4-5-20250929"
    temperature: 0.7
    max_tokens: 2048
    system_prompt: "You are a specialist in my domain."
    specialized_role: "my_new_role"  # References the role file
```

## Scalability

This system scales to hundreds of roles without code changes:
- Each role is a separate YAML file
- Roles are loaded automatically at runtime
- No need to modify source code to add new roles
- Community contributions can easily add new role definitions

## Built-in Roles

The system includes these default roles:

- **general**: General-purpose development
- **security**: Security vulnerabilities and secure coding
- **performance**: Code optimization and performance
- **concurrency**: Threading and parallel programming
- **reliability**: Error handling and fault tolerance
- **framework**: Framework-specific patterns
- **database**: SQL and database operations
- **ai_ml**: AI/ML and data science code
- **testing**: Test automation and quality assurance

Each role provides specialized prompting to help agents focus on domain-specific best practices and common issues.