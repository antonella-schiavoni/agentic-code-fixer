"""Role management system for scalable agent role definitions.

This module provides functionality to load and manage agent role definitions
from external YAML files, making the system scalable to hundreds of roles
without code changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RoleDefinition(BaseModel):
    """Definition of an agent role loaded from a YAML file.

    Attributes:
        name: Unique identifier for the role.
        description: Human-readable description of the role's purpose.
        prompt_addition: Additional prompt text to add for this role.
        category: Optional category for organizing roles.
        priority: Optional priority level (high, medium, low).
        tags: Optional list of tags for filtering and discovery.
    """

    name: str
    description: str
    prompt_addition: str
    category: str | None = None
    priority: str | None = None
    tags: list[str] = Field(default_factory=list)


class RoleManager:
    """Manager for loading and providing agent role definitions.

    This class handles the discovery and loading of role definitions from
    YAML files, providing a scalable way to manage hundreds of agent roles
    without hardcoding them in the source code.

    Attributes:
        roles_directory: Path to the directory containing role definition files.
        roles: Dictionary of loaded role definitions by name.
        default_role: Name of the default role to use when none is specified.
    """

    def __init__(self, roles_directory: str | Path = "roles", default_role: str = "general"):
        """Initialize the role manager.

        Args:
            roles_directory: Directory containing role definition YAML files.
            default_role: Name of the default role to use as fallback.
        """
        self.roles_directory = Path(roles_directory)
        self.default_role = default_role
        self.roles: dict[str, RoleDefinition] = {}
        self._load_roles()

    def _load_roles(self) -> None:
        """Load all role definitions from the roles directory."""
        if not self.roles_directory.exists():
            logger.warning(f"Roles directory does not exist: {self.roles_directory}")
            self._create_fallback_roles()
            return

        yaml_files = list(self.roles_directory.glob("*.yaml")) + list(self.roles_directory.glob("*.yml"))

        if not yaml_files:
            logger.warning(f"No YAML files found in roles directory: {self.roles_directory}")
            self._create_fallback_roles()
            return

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    role_data = yaml.safe_load(f)

                role_def = RoleDefinition(**role_data)
                self.roles[role_def.name] = role_def
                logger.debug(f"Loaded role definition: {role_def.name}")

            except Exception as e:
                logger.error(f"Failed to load role definition from {yaml_file}: {e}")

        logger.info(f"Loaded {len(self.roles)} role definitions")

    def _create_fallback_roles(self) -> None:
        """Create fallback role definitions if no external files are found."""
        fallback_roles = {
            "general": RoleDefinition(
                name="general",
                description="General-purpose agent for standard code fixes",
                prompt_addition=(
                    "Provide well-structured, maintainable solutions that follow best practices "
                    "for the given programming language. Ensure code readability and proper error handling."
                ),
                category="general",
                priority="medium",
                tags=["general", "best-practices"]
            ),
            "security": RoleDefinition(
                name="security",
                description="Agent focused on security vulnerabilities and secure coding practices",
                prompt_addition=(
                    "Pay special attention to security vulnerabilities and ensure any fixes "
                    "don't introduce new security issues. Focus on input validation, "
                    "authentication, authorization, and data sanitization."
                ),
                category="security",
                priority="high",
                tags=["security", "vulnerabilities"]
            ),
            "performance": RoleDefinition(
                name="performance",
                description="Agent focused on code performance and optimization",
                prompt_addition=(
                    "Focus on optimizing code performance while maintaining correctness. "
                    "Consider algorithm efficiency, memory usage, and runtime complexity. "
                    "Avoid solutions that might degrade performance."
                ),
                category="optimization",
                priority="medium",
                tags=["performance", "optimization"]
            ),
        }

        self.roles.update(fallback_roles)
        logger.info("Created fallback role definitions")

    def get_role_prompt_addition(self, role_name: str | None) -> str:
        """Get the prompt addition text for a specific role.

        Args:
            role_name: Name of the role, or None for default role.

        Returns:
            Prompt addition text for the specified role.
        """
        if not role_name:
            role_name = self.default_role

        role_def = self.roles.get(role_name)
        if not role_def:
            logger.warning(f"Role '{role_name}' not found, using default role '{self.default_role}'")
            role_def = self.roles.get(self.default_role)

        if not role_def:
            logger.error(f"Default role '{self.default_role}' not found")
            return ""

        return role_def.prompt_addition

    def get_role_definition(self, role_name: str) -> RoleDefinition | None:
        """Get the complete role definition for a specific role.

        Args:
            role_name: Name of the role to retrieve.

        Returns:
            RoleDefinition object if found, None otherwise.
        """
        return self.roles.get(role_name)

    def list_available_roles(self) -> list[str]:
        """Get a list of all available role names.

        Returns:
            List of role names that can be used.
        """
        return list(self.roles.keys())

    def get_roles_by_category(self, category: str) -> list[RoleDefinition]:
        """Get all roles in a specific category.

        Args:
            category: Category name to filter by.

        Returns:
            List of role definitions in the specified category.
        """
        return [role for role in self.roles.values() if role.category == category]

    def get_roles_by_tag(self, tag: str) -> list[RoleDefinition]:
        """Get all roles that have a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of role definitions that have the specified tag.
        """
        return [role for role in self.roles.values() if tag in role.tags]

    def reload_roles(self) -> None:
        """Reload all role definitions from disk.

        This allows adding new roles without restarting the application.
        """
        self.roles.clear()
        self._load_roles()
        logger.info("Reloaded role definitions")

    def get_role_stats(self) -> dict[str, Any]:
        """Get statistics about loaded roles.

        Returns:
            Dictionary containing role statistics.
        """
        categories = {}
        priorities = {}

        for role in self.roles.values():
            if role.category:
                categories[role.category] = categories.get(role.category, 0) + 1
            if role.priority:
                priorities[role.priority] = priorities.get(role.priority, 0) + 1

        return {
            "total_roles": len(self.roles),
            "categories": categories,
            "priorities": priorities,
            "roles_directory": str(self.roles_directory),
            "default_role": self.default_role,
        }