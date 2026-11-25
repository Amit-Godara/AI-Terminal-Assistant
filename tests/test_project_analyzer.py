"""Unit and integration tests for Phase 11: Project Intelligence Engine.

Tests cover:
1. Python manifest analysis (pyproject.toml, requirements.txt, setup.py)
2. Python requirements extraction
3. JavaScript package metadata (name, version, main entry point)
4. npm script extraction (data only, never executed)
5. React dependency identification
6. Java Maven analysis (pom.xml)
7. Gradle analysis (build.gradle)
8. Flutter pubspec analysis (pubspec.yaml)
9. .NET project analysis (.csproj)
10. Docker configuration analysis (Dockerfile, docker-compose.yml)
11. README information extraction (description, install, run, test)
12. Project structure analysis (source, test, tree representation)
13. Mixed full-stack project analysis (polyglot)
14. Empty project handling
15. Malformed JSON handling
16. Malformed TOML handling
17. Malformed YAML handling
18. Missing manifests handling
19. File-size limit protection
20. Ignored directories skipped (node_modules, .venv)
21. Symlink boundary protection
22. Permission error handling
23. No filesystem mutation verification
24. No project command execution verification
25. Secret file protection (.env files never read)
"""

import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from core.project_analyzer import (
    ProjectAnalyzer,
    ProjectAnalysisResult,
)
from core.project_detector import DetectionStatus


@pytest.fixture
def analyzer() -> ProjectAnalyzer:
    """Fixture providing a fresh ProjectAnalyzer instance."""
    return ProjectAnalyzer()


# ------------------------------------------------------------------------------
# 1 & 2. Python Manifest & Requirements Analysis
# ------------------------------------------------------------------------------
def test_python_manifest_and_requirements_analysis(analyzer, tmp_path):
    """Verifies extraction of Python metadata, dependencies, and entry points."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "my-awesome-python-app"
version = "1.2.3"
description = "A production FastAPI service"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0"
]

[project.scripts]
my-cli = "my_app.cli:main"
        """,
        encoding="utf-8",
    )

    reqs = tmp_path / "requirements.txt"
    reqs.write_text("pytest>=7.4.0\npytest-cov\nhttpx\n", encoding="utf-8")

    result = analyzer.analyze(tmp_path)
    assert result.project_name == "my-awesome-python-app"
    assert result.project_version == "1.2.3"
    assert "A production FastAPI service" in (result.project_description or "")
    assert "fastapi" in result.dependencies
    assert "pytest" in result.dependencies
    assert any("my-cli" in ep for ep in result.entry_points)
    assert "pyproject.toml" in result.files_analyzed
    assert "requirements.txt" in result.files_analyzed


# ------------------------------------------------------------------------------
# 3, 4, 5. JavaScript / TypeScript & npm Script Extraction
# ------------------------------------------------------------------------------
def test_javascript_package_metadata_and_scripts(analyzer, tmp_path):
    """Verifies package metadata, exact script extraction, and React dependency identification."""
    pkg = tmp_path / "package.json"
    pkg.write_text(
        json.dumps({
            "name": "enterprise-web-dashboard",
            "version": "2.0.0",
            "description": "Next.js & React enterprise dashboard",
            "main": "dist/index.js",
            "scripts": {
                "dev": "next dev -p 3000",
                "build": "next build",
                "start": "next start",
                "test": "jest --coverage",
                "lint": "eslint src/"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "next": "^14.0.0"
            },
            "devDependencies": {
                "typescript": "^5.2.0",
                "@types/react": "^18.2.0"
            }
        }),
        encoding="utf-8",
    )

    result = analyzer.analyze(tmp_path)
    assert result.project_name == "enterprise-web-dashboard"
    assert result.project_version == "2.0.0"
    assert "dist/index.js" in result.entry_points
    assert "react" in result.dependencies
    assert "next" in result.dependencies
    assert "typescript" in result.development_dependencies

    # Scripts must be extracted with exact declared values
    assert result.available_scripts["dev"] == "next dev -p 3000"
    assert result.available_scripts["build"] == "next build"
    assert result.available_scripts["test"] == "jest --coverage"

    # Inferred test configuration from scripts
    assert result.test_configuration.get("npm_test_script") == "jest --coverage"


# ------------------------------------------------------------------------------
# 6. Java Maven Analysis
# ------------------------------------------------------------------------------
def test_java_maven_analysis(analyzer, tmp_path):
    """Verifies Maven pom.xml parsing for artifactId, version, and dependencies."""
    pom = tmp_path / "pom.xml"
    pom.write_text(
        """
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.company.service</groupId>
  <artifactId>user-auth-service</artifactId>
  <version>1.5.0-SNAPSHOT</version>
  <dependencies>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-security</artifactId>
    </dependency>
  </dependencies>
</project>
        """,
        encoding="utf-8",
    )

    result = analyzer.analyze(tmp_path)
    assert result.project_name == "user-auth-service"
    assert result.project_version == "1.5.0-SNAPSHOT"
    assert any("spring-boot-starter-security" in d for d in result.dependencies)
    assert result.build_configuration.get("java_build_tool") == "Maven"


# ------------------------------------------------------------------------------
# 7. Java Gradle Analysis
# ------------------------------------------------------------------------------
def test_java_gradle_analysis(analyzer, tmp_path):
    """Verifies Gradle build.gradle parsing for dependencies."""
    gradle_file = tmp_path / "build.gradle"
    gradle_file.write_text(
        """
plugins {
    id 'java'
}
dependencies {
    implementation 'com.google.guava:guava:32.0.0-jre'
    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
}
        """,
        encoding="utf-8",
    )

    result = analyzer.analyze(tmp_path)
    assert any("guava" in d for d in result.dependencies)
    assert result.build_configuration.get("java_build_tool") == "Gradle"


# ------------------------------------------------------------------------------
# 8. Flutter / Dart Analysis
# ------------------------------------------------------------------------------
def test_flutter_pubspec_analysis(analyzer, tmp_path):
    """Verifies Flutter pubspec.yaml analysis."""
    pubspec = tmp_path / "pubspec.yaml"
    pubspec.write_text(
        """
name: flutter_chat_app
version: 1.0.4+5
description: A clean Flutter messaging client
environment:
  sdk: '>=3.0.0 <4.0.0'
dependencies:
  flutter:
    sdk: flutter
  http: ^1.1.0
  provider: ^6.0.5
dev_dependencies:
  flutter_test:
    sdk: flutter
        """,
        encoding="utf-8",
    )

    result = analyzer.analyze(tmp_path)
    assert result.project_name == "flutter_chat_app"
    assert result.project_version == "1.0.4+5"
    assert "A clean Flutter messaging client" in (result.project_description or "")
    assert "http" in result.dependencies
    assert "provider" in result.dependencies
    assert result.build_configuration.get("flutter_project") is True


# ------------------------------------------------------------------------------
# 9. .NET Project Analysis
# ------------------------------------------------------------------------------
def test_dotnet_project_analysis(analyzer, tmp_path):
    """Verifies .NET .csproj target framework and package references."""
    csproj = tmp_path / "OrderProcessing.csproj"
    csproj.write_text(
        """
<Project Sdk="Microsoft.NET.Sdk.Web">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageReference Include="Serilog.AspNetCore" Version="7.0.0" />
  </ItemGroup>
</Project>
        """,
        encoding="utf-8",
    )

    result = analyzer.analyze(tmp_path)
    assert result.build_configuration.get("dotnet_target_framework") == "net8.0"
    assert any("Newtonsoft.Json" in d for d in result.dependencies)
    assert any("Serilog.AspNetCore" in d for d in result.dependencies)


# ------------------------------------------------------------------------------
# 10. Docker Configuration Analysis
# ------------------------------------------------------------------------------
def test_docker_configuration_analysis(analyzer, tmp_path):
    """Verifies Dockerfile and docker-compose service extraction."""
    (tmp_path / "Dockerfile").write_text("FROM node:20-alpine\nEXPOSE 3000 8080\nCMD [\"npm\", \"start\"]\n", encoding="utf-8")
    (tmp_path / "docker-compose.yml").write_text(
        """
version: '3.8'
services:
  web:
    build: .
    ports:
      - "3000:3000"
  redis:
    image: redis:7-alpine
  postgres:
    image: postgres:15
        """,
        encoding="utf-8",
    )

    result = analyzer.analyze(tmp_path)
    infra = result.infrastructure_configuration
    assert "node:20-alpine" in infra.get("dockerfile_base_images", [])
    assert set(infra.get("docker_compose_services", [])) == {"web", "redis", "postgres"}


# ------------------------------------------------------------------------------
# 11. README Information Extraction
# ------------------------------------------------------------------------------
def test_readme_information_extraction(analyzer, tmp_path):
    """Verifies extraction of description, installation, run, and test sections from README."""
    readme = tmp_path / "README.md"
    readme.write_text(
        """
# Cloud File Synchronizer

Cloud File Synchronizer is a robust tool designed for encrypted cross-cloud replication.

## Prerequisites
Ensure Python 3.11+ is installed.

## Installation
Run `pip install -r requirements.txt` to install all dependencies.

## Usage & Development
Start the sync daemon by running `python -m sync_daemon --config config.yaml`.

## Testing
Run the unit test suite using `pytest tests/ -v`.
        """,
        encoding="utf-8",
    )

    result = analyzer.analyze(tmp_path)
    assert "Cloud File Synchronizer is a robust tool" in (result.project_description or "")
    assert "pip install -r requirements.txt" in (result.documented_installation_instructions or "")
    assert "python -m sync_daemon" in (result.documented_run_instructions or "")
    assert "pytest tests/ -v" in (result.documented_test_instructions or "")
    assert "README.md" in result.files_analyzed


# ------------------------------------------------------------------------------
# 12. Project Structure Analysis
# ------------------------------------------------------------------------------
def test_project_structure_analysis(analyzer, tmp_path):
    """Verifies identification of source, test, doc directories, and entry point files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.ts").write_text("console.log('hi');", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")

    result = analyzer.analyze(tmp_path)
    struct = result.project_structure
    assert "src" in struct.get("source_dirs", [])
    assert "tests" in struct.get("test_dirs", [])
    assert "docs" in struct.get("doc_dirs", [])
    assert any("app.py" in ep for ep in struct.get("entry_points", []))
    assert tmp_path.name in struct.get("tree_representation", "")


# ------------------------------------------------------------------------------
# 13. Mixed Full-Stack Project Analysis (Polyglot)
# ------------------------------------------------------------------------------
def test_mixed_fullstack_project_analysis(analyzer, tmp_path):
    """Verifies analysis of a full-stack project with both Python backend and Node frontend."""
    (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
    (tmp_path / "package.json").write_text(json.dumps({"name": "frontend-ui", "dependencies": {"vue": "^3.0"}}), encoding="utf-8")

    result = analyzer.analyze(tmp_path)
    assert "fastapi" in result.dependencies
    assert "vue" in result.dependencies
    assert len(result.detection_result.detected_languages) >= 2


# ------------------------------------------------------------------------------
# 14. Empty Project Handling
# ------------------------------------------------------------------------------
def test_empty_project_handling(analyzer, tmp_path):
    """Verifies that analyzing an empty directory succeeds safely without errors."""
    result = analyzer.analyze(tmp_path)
    assert result.detection_result.is_valid_directory is True
    assert result.dependencies == []
    assert result.available_scripts == {}
    assert result.project_name == tmp_path.name


# ------------------------------------------------------------------------------
# 15, 16, 17. Malformed Manifests Handling (JSON, TOML, YAML)
# ------------------------------------------------------------------------------
def test_malformed_manifests_handling(analyzer, tmp_path):
    """Verifies that malformed manifests produce warnings rather than crashing."""
    (tmp_path / "package.json").write_text("{broken json syntax!!", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[[broken toml syntax", encoding="utf-8")
    (tmp_path / "pubspec.yaml").write_text("name: valid_name\n  bad_indent: [", encoding="utf-8")

    result = analyzer.analyze(tmp_path)
    assert result.detection_result.is_valid_directory is True
    assert len(result.analysis_warnings) > 0


# ------------------------------------------------------------------------------
# 18. Missing Manifests Handling
# ------------------------------------------------------------------------------
def test_missing_manifests_handling(analyzer, tmp_path):
    """Verifies handling of directory containing only non-manifest text files."""
    (tmp_path / "document.txt").write_text("just some text", encoding="utf-8")
    result = analyzer.analyze(tmp_path)
    assert result.detection_result.detection_status == DetectionStatus.NO_TECHNOLOGIES_DETECTED
    assert result.dependencies == []
    assert result.files_analyzed == []


# ------------------------------------------------------------------------------
# 19. File-Size Limit Protection
# ------------------------------------------------------------------------------
def test_file_size_limit_protection(tmp_path):
    """Verifies that huge manifests exceeding max_file_size_bytes are safely skipped."""
    analyzer = ProjectAnalyzer(max_file_size_bytes=50)  # Tiny 50-byte limit
    huge_reqs = tmp_path / "requirements.txt"
    huge_reqs.write_text("a" * 200, encoding="utf-8")

    result = analyzer.analyze(tmp_path)
    assert any("exceeds limit" in w for w in result.analysis_warnings)
    assert result.dependencies == []


# ------------------------------------------------------------------------------
# 20. Ignored Directories Skipped
# ------------------------------------------------------------------------------
def test_ignored_directories_skipped(analyzer, tmp_path):
    """Verifies node_modules and .venv directories are completely skipped."""
    node_modules = tmp_path / "node_modules" / "express"
    node_modules.mkdir(parents=True, exist_ok=True)
    (node_modules / "package.json").write_text(json.dumps({"name": "express-internal"}), encoding="utf-8")

    result = analyzer.analyze(tmp_path)
    assert "node_modules" not in result.project_structure.get("source_dirs", [])
    assert result.project_name != "express-internal"


# ------------------------------------------------------------------------------
# 21. Symlink Boundary Protection
# ------------------------------------------------------------------------------
def test_symlink_boundary_protection(analyzer, tmp_path):
    """Verifies symlinks pointing outside the project root are not read."""
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    outside_file = outside_dir / "external_reqs.txt"
    outside_file.write_text("should_not_read_this==1.0.0", encoding="utf-8")

    inside_dir = tmp_path / "project_root"
    inside_dir.mkdir()
    link_path = inside_dir / "requirements.txt"

    try:
        os.symlink(outside_file, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not permitted in this environment.")

    result = analyzer.analyze(inside_dir)
    assert "should_not_read_this" not in result.dependencies


# ------------------------------------------------------------------------------
# 22. Permission Error Handling
# ------------------------------------------------------------------------------
def test_permission_error_graceful_handling(analyzer, tmp_path):
    """Verifies simulated permission error is caught and logged safely."""
    with patch("os.scandir", side_effect=PermissionError("Simulated access denial")):
        result = analyzer.analyze(tmp_path)
        assert result.detection_result.detection_status == DetectionStatus.ACCESS_DENIED
        assert any("Permission denied" in w for w in result.analysis_warnings)


# ------------------------------------------------------------------------------
# 23. No Filesystem Mutation Verification
# ------------------------------------------------------------------------------
def test_no_filesystem_mutation(analyzer, tmp_path):
    """Verifies that analyzer never creates, modifies, or deletes any files in the project."""
    (tmp_path / "test.txt").write_text("original content", encoding="utf-8")
    before_files = set(tmp_path.iterdir())
    before_stat = (tmp_path / "test.txt").stat().st_mtime

    _ = analyzer.analyze(tmp_path)

    after_files = set(tmp_path.iterdir())
    after_stat = (tmp_path / "test.txt").stat().st_mtime

    assert before_files == after_files
    assert before_stat == after_stat


# ------------------------------------------------------------------------------
# 24. No Command Execution Verification
# ------------------------------------------------------------------------------
def test_no_command_execution(analyzer, tmp_path):
    """Verifies that subprocess.run is NEVER called during project analysis."""
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"malicious": "rm -rf /"}}), encoding="utf-8")

    with patch("subprocess.run") as mock_subproc, patch("os.system") as mock_sys:
        _ = analyzer.analyze(tmp_path)
        mock_subproc.assert_not_called()
        mock_sys.assert_not_called()


# ------------------------------------------------------------------------------
# 25. Secret File Protection (.env files never read)
# ------------------------------------------------------------------------------
def test_secret_file_protection(analyzer, tmp_path):
    """Verifies that .env and secret files are NEVER opened or read."""
    env_file = tmp_path / ".env"
    env_file.write_text("API_SECRET_KEY=super_secret_value_12345\nDATABASE_PASSWORD=secret", encoding="utf-8")

    result = analyzer.analyze(tmp_path)
    assert ".env" not in result.files_analyzed
    # Confirm secret contents never appear in any evidence or warnings
    for warn in result.analysis_warnings:
        assert "super_secret_value" not in warn
    for ev_list in result.evidence.values():
        for ev in ev_list:
            assert "super_secret_value" not in ev
