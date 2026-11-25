"""Unit and integration tests for Phase 10: Project Detection Engine.

Tests cover:
1. Python project detection (requirements.txt, pyproject.toml, Pipfile, setup.py)
2. Node.js project detection (package.json, lockfiles)
3. React project identification from dependencies in package.json
4. Java Maven detection (pom.xml, Spring Boot)
5. Java Gradle detection (build.gradle, build.gradle.kts)
6. CMake C++ detection (CMakeLists.txt, C++ sources)
7. Flutter detection (pubspec.yaml with Flutter SDK) vs standalone Dart
8. .NET detection (.csproj, .sln)
9. Docker detection (Dockerfile, compose.yml coexisting with code)
10. Multi-technology project detection (polyglot)
11. Monorepo detection (pnpm-workspaces, lerna, workspaces field)
12. Empty directory handling
13. Invalid project root (non-existent path, file path)
14. Missing project markers (generic text files only)
15. Malformed manifest handling (broken JSON, TOML)
16. Permission/access error handling
17. Ignored dependency/build directories (node_modules, venv, etc.)
18. Symlink boundary protection
"""

import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from core.project_detector import (
    ProjectDetector,
    ProjectDetectionResult,
    DetectionStatus,
    DetectionConfidence,
)


@pytest.fixture
def detector() -> ProjectDetector:
    """Fixture providing a fresh ProjectDetector instance."""
    return ProjectDetector()


# ------------------------------------------------------------------------------
# 1. Python Project Detection
# ------------------------------------------------------------------------------
def test_python_requirements_detection(detector, tmp_path):
    """Detects Python project with requirements.txt."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("fastapi>=0.100.0\nuvicorn\npydantic\n", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert result.is_valid_directory is True
    assert result.detection_status == DetectionStatus.SUCCESS
    assert "Python" in result.detected_languages
    assert result.primary_language == "Python"
    assert "pip" in result.package_managers
    assert "FastAPI" in result.frameworks
    assert "requirements.txt" in result.project_markers
    assert "Python" in result.evidence


def test_python_pyproject_toml_poetry_detection(detector, tmp_path):
    """Detects Python project with pyproject.toml and Poetry build tool."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.poetry]
name = "demo-app"
version = "0.1.0"
description = "Sample app"

[tool.poetry.dependencies]
python = "^3.11"
django = "^5.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
        """,
        encoding="utf-8",
    )

    result = detector.detect(tmp_path)
    assert "Python" in result.detected_languages
    assert result.primary_language == "Python"
    assert "poetry" in result.build_tools
    assert "poetry" in result.package_managers
    assert "Django" in result.frameworks
    assert "pyproject.toml" in result.project_markers


def test_python_pipfile_detection(detector, tmp_path):
    """Detects Python project with Pipfile."""
    pipfile = tmp_path / "Pipfile"
    pipfile.write_text("[packages]\nflask = '*'\n", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "Python" in result.detected_languages
    assert "pipenv" in result.package_managers
    assert "Pipfile" in result.project_markers


def test_python_setup_py_detection(detector, tmp_path):
    """Detects Python project with setup.py."""
    setup_file = tmp_path / "setup.py"
    setup_file.write_text("from setuptools import setup\nsetup(name='sample')\n", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "Python" in result.detected_languages
    assert "setuptools" in result.build_tools
    assert "setup.py" in result.project_markers


# ------------------------------------------------------------------------------
# 2. JavaScript / TypeScript Project Detection
# ------------------------------------------------------------------------------
def test_nodejs_project_detection(detector, tmp_path):
    """Detects pure Node.js / JavaScript project with package.json and npm lockfile."""
    pkg = tmp_path / "package.json"
    pkg.write_text(
        json.dumps({
            "name": "node-backend",
            "version": "1.0.0",
            "dependencies": {
                "express": "^4.18.0"
            }
        }),
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "JavaScript" in result.detected_languages
    assert result.primary_language == "JavaScript"
    assert "npm" in result.package_managers
    assert "Express" in result.frameworks
    assert "package.json" in result.project_markers
    assert "package-lock.json" in result.project_markers


def test_typescript_project_detection(detector, tmp_path):
    """Detects TypeScript project via tsconfig.json and yarn lockfile."""
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({"name": "ts-lib", "devDependencies": {"typescript": "^5.0"}}), encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    (tmp_path / "yarn.lock").write_text("", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "TypeScript" in result.detected_languages
    assert result.primary_language == "TypeScript"
    assert "yarn" in result.package_managers
    assert "tsconfig.json" in result.project_markers


# ------------------------------------------------------------------------------
# 3. React Project Identification from Manifest Dependencies
# ------------------------------------------------------------------------------
def test_react_project_identification(detector, tmp_path):
    """Identifies React framework specifically from package dependencies, not just package.json."""
    pkg = tmp_path / "package.json"
    pkg.write_text(
        json.dumps({
            "name": "react-frontend",
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0"
            },
            "devDependencies": {
                "vite": "^4.0.0"
            }
        }),
        encoding="utf-8",
    )
    (tmp_path / "pnpm-lock.yaml").write_text("", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "React" in result.frameworks
    assert "Vite" in result.build_tools
    assert "pnpm" in result.package_managers


# ------------------------------------------------------------------------------
# 4. Java Maven Detection
# ------------------------------------------------------------------------------
def test_java_maven_detection(detector, tmp_path):
    """Detects Java Maven project with Spring Boot."""
    pom = tmp_path / "pom.xml"
    pom.write_text(
        """
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <dependencies>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
  </dependencies>
</project>
        """,
        encoding="utf-8",
    )

    result = detector.detect(tmp_path)
    assert "Java" in result.detected_languages
    assert result.primary_language == "Java"
    assert "Maven" in result.build_tools
    assert "Maven" in result.package_managers
    assert "Spring Boot" in result.frameworks
    assert "pom.xml" in result.project_markers


# ------------------------------------------------------------------------------
# 5. Java Gradle Detection
# ------------------------------------------------------------------------------
def test_java_gradle_detection(detector, tmp_path):
    """Detects Gradle Java/Kotlin project."""
    gradle_kts = tmp_path / "build.gradle.kts"
    gradle_kts.write_text(
        'plugins { java\n id("org.springframework.boot") version "3.2.0" }\n',
        encoding="utf-8",
    )

    result = detector.detect(tmp_path)
    assert "Java" in result.detected_languages
    assert "Kotlin" in result.detected_languages
    assert "Gradle" in result.build_tools
    assert "Gradle" in result.package_managers
    assert "Spring Boot" in result.frameworks
    assert "build.gradle.kts" in result.project_markers


# ------------------------------------------------------------------------------
# 6. CMake C++ Detection
# ------------------------------------------------------------------------------
def test_cmake_cpp_detection(detector, tmp_path):
    """Detects CMake C++ project."""
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\nproject(Engine)\n", encoding="utf-8")
    (tmp_path / "main.cpp").write_text('#include <iostream>\nint main() { return 0; }\n', encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "C++" in result.detected_languages
    assert "CMake" in result.build_tools
    assert "CMakeLists.txt" in result.project_markers


# ------------------------------------------------------------------------------
# 7. Flutter / Dart Detection
# ------------------------------------------------------------------------------
def test_flutter_project_detection(detector, tmp_path):
    """Detects Flutter specifically when manifest indicates Flutter."""
    pubspec = tmp_path / "pubspec.yaml"
    pubspec.write_text(
        """
name: my_flutter_app
description: A new Flutter project.
environment:
  sdk: '>=3.0.0 <4.0.0'
dependencies:
  flutter:
    sdk: flutter
flutter:
  uses-material-design: true
        """,
        encoding="utf-8",
    )

    result = detector.detect(tmp_path)
    assert "Dart" in result.detected_languages
    assert "Flutter" in result.frameworks
    assert "Flutter CLI" in result.build_tools
    assert "pub" in result.package_managers
    assert "pubspec.yaml" in result.project_markers


def test_pure_dart_project_detection(detector, tmp_path):
    """Detects Dart without Flutter when manifest has no Flutter SDK dependency."""
    pubspec = tmp_path / "pubspec.yaml"
    pubspec.write_text(
        """
name: dart_cli_tool
dependencies:
  args: ^2.4.0
        """,
        encoding="utf-8",
    )

    result = detector.detect(tmp_path)
    assert "Dart" in result.detected_languages
    assert "Flutter" not in result.frameworks
    assert "pub" in result.package_managers


# ------------------------------------------------------------------------------
# 8. .NET Detection
# ------------------------------------------------------------------------------
def test_dotnet_csharp_detection(detector, tmp_path):
    """Detects C# and .NET solution."""
    (tmp_path / "App.csproj").write_text("<Project Sdk=\"Microsoft.NET.Sdk\">\n</Project>\n", encoding="utf-8")
    (tmp_path / "App.sln").write_text("Microsoft Visual Studio Solution File", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "C#" in result.detected_languages
    assert "dotnet" in result.build_tools
    assert "NuGet" in result.package_managers
    assert "App.csproj" in result.project_markers
    assert "App.sln" in result.project_markers


# ------------------------------------------------------------------------------
# 9. Docker Detection (Coexisting with code)
# ------------------------------------------------------------------------------
def test_docker_detection(detector, tmp_path):
    """Detects Dockerfile and Docker Compose alongside a primary language."""
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    (tmp_path / "docker-compose.yml").write_text("version: '3.8'\nservices:\n", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "Python" in result.detected_languages
    assert "Docker" in result.infrastructure
    assert "Docker Compose" in result.infrastructure
    assert "Dockerfile" in result.project_markers
    assert "docker-compose.yml" in result.project_markers


# ------------------------------------------------------------------------------
# 10. Multi-Technology Project Detection
# ------------------------------------------------------------------------------
def test_multi_technology_polyglot_project(detector, tmp_path):
    """Verifies that multiple languages/frameworks are preserved without forcing a single type."""
    # Python backend
    (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    # TypeScript frontend
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"react": "^18.0"}}), encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert "Python" in result.detected_languages
    assert "JavaScript" in result.detected_languages or "TypeScript" in result.detected_languages
    assert "FastAPI" in result.frameworks
    assert "React" in result.frameworks
    assert len(result.detected_languages) >= 2


# ------------------------------------------------------------------------------
# 11. Monorepo Detection
# ------------------------------------------------------------------------------
def test_pnpm_monorepo_detection(detector, tmp_path):
    """Detects pnpm workspaces monorepo."""
    (tmp_path / "pnpm-workspace.yaml").write_text("packages:\n  - 'packages/*'\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert result.is_monorepo is True
    assert result.monorepo_type == "pnpm Workspaces"
    assert "pnpm-workspace.yaml" in result.project_markers


def test_turborepo_detection(detector, tmp_path):
    """Detects Turborepo monorepo configuration."""
    (tmp_path / "turbo.json").write_text("{}", encoding="utf-8")
    (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["apps/*"]}), encoding="utf-8")

    result = detector.detect(tmp_path)
    assert result.is_monorepo is True
    assert "turbo.json" in result.project_markers


# ------------------------------------------------------------------------------
# 12. Empty Directory Handling
# ------------------------------------------------------------------------------
def test_empty_directory_handling(detector, tmp_path):
    """Detects empty directory gracefully without crashing."""
    result = detector.detect(tmp_path)
    assert result.is_valid_directory is True
    assert result.detection_status == DetectionStatus.NO_TECHNOLOGIES_DETECTED
    assert result.primary_language is None
    assert result.detected_languages == []
    assert result.frameworks == []


# ------------------------------------------------------------------------------
# 13. Invalid Project Root
# ------------------------------------------------------------------------------
def test_nonexistent_directory(detector, tmp_path):
    """Handles non-existent directory path."""
    non_existent = tmp_path / "does_not_exist"
    result = detector.detect(non_existent)
    assert result.is_valid_directory is False
    assert result.detection_status == DetectionStatus.INVALID_DIRECTORY
    assert len(result.detection_warnings) > 0


def test_file_instead_of_directory(detector, tmp_path):
    """Handles pointing to a file instead of a directory."""
    file_path = tmp_path / "some_file.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = detector.detect(file_path)
    assert result.is_valid_directory is False
    assert result.detection_status == DetectionStatus.INVALID_DIRECTORY


# ------------------------------------------------------------------------------
# 14. Missing Project Markers (Generic text files only)
# ------------------------------------------------------------------------------
def test_generic_text_files_no_markers(detector, tmp_path):
    """Handles directory with plain text and documentation only."""
    (tmp_path / "notes.txt").write_text("some notes", encoding="utf-8")
    (tmp_path / "README.txt").write_text("readme", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert result.is_valid_directory is True
    assert result.detection_status == DetectionStatus.NO_TECHNOLOGIES_DETECTED
    assert result.detected_languages == []


# ------------------------------------------------------------------------------
# 15. Malformed Manifest Handling
# ------------------------------------------------------------------------------
def test_malformed_package_json_graceful(detector, tmp_path):
    """Verifies malformed JSON in package.json produces warning without raising an exception."""
    pkg = tmp_path / "package.json"
    pkg.write_text("{broken json: 123", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert result.is_valid_directory is True
    assert "JavaScript" in result.detected_languages
    assert any("Malformed package.json" in w for w in result.detection_warnings)


# ------------------------------------------------------------------------------
# 16. Permission/Access Error Handling
# ------------------------------------------------------------------------------
def test_permission_error_handling(detector, tmp_path):
    """Handles simulated PermissionError during directory access."""
    with patch("os.scandir", side_effect=PermissionError("Permission denied")):
        result = detector.detect(tmp_path)
        assert result.is_valid_directory is False
        assert result.detection_status == DetectionStatus.ACCESS_DENIED
        assert any("Permission denied" in w for w in result.detection_warnings)


# ------------------------------------------------------------------------------
# 17. Ignored Dependency and Build Directories
# ------------------------------------------------------------------------------
def test_ignored_directories_skipped(detector, tmp_path):
    """Ensures node_modules and .venv directories are completely ignored."""
    # Place a nested package.json inside node_modules that should NOT be detected
    node_modules = tmp_path / "node_modules" / "dependency-lib"
    node_modules.mkdir(parents=True, exist_ok=True)
    (node_modules / "package.json").write_text(json.dumps({"name": "dep-pkg"}), encoding="utf-8")

    # Place a nested requirements.txt inside .venv that should NOT be detected
    venv_dir = tmp_path / ".venv" / "lib"
    venv_dir.mkdir(parents=True, exist_ok=True)
    (venv_dir / "requirements.txt").write_text("hidden-pkg\n", encoding="utf-8")

    result = detector.detect(tmp_path)
    assert result.detection_status == DetectionStatus.NO_TECHNOLOGIES_DETECTED
    assert "node_modules/dependency-lib/package.json" not in result.project_markers
    assert ".venv/lib/requirements.txt" not in result.project_markers


# ------------------------------------------------------------------------------
# 18. Symlink Boundary Protection
# ------------------------------------------------------------------------------
def test_symlink_escaping_boundary_skipped(detector, tmp_path):
    """Verifies that symlinks pointing outside the project root are skipped."""
    outside_dir = tmp_path / "outside_root"
    outside_dir.mkdir()
    (outside_dir / "package.json").write_text('{"name": "outside"}', encoding="utf-8")

    inside_dir = tmp_path / "project"
    inside_dir.mkdir()

    symlink_path = inside_dir / "linked_subfolder"
    try:
        os.symlink(outside_dir, symlink_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported or permitted on this environment.")

    result = detector.detect(inside_dir)
    assert any("Skipping symlink" in w for w in result.detection_warnings)
    assert "package.json" not in result.project_markers
