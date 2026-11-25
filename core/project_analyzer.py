"""Project Intelligence Engine module for AI Terminal Assistant / Copilot.

Analyzes an explicitly selected project directory to extract structural insights,
configuration, dependencies, dev dependencies, entry points, available scripts,
Docker configurations, and README documentation in a strictly read-only, safe manner.
"""

from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field

from core.project_detector import (
    ProjectDetector,
    ProjectDetectionResult,
    DetectionStatus,
    IGNORED_DIRS,
    DEFAULT_MAX_FILE_SIZE_BYTES,
)

logger = logging.getLogger(__name__)

# Patterns for secret files that must NEVER be read or inspected
SECRET_FILE_PATTERNS: Set[str] = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.staging",
    ".env.test",
    ".envrc",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
}

# Source directory names
COMMON_SOURCE_DIRS = {"src", "app", "lib", "backend", "frontend", "packages", "cmd", "pkg", "core", "internal"}
# Test directory names
COMMON_TEST_DIRS = {"tests", "test", "spec", "specs", "__tests__", "testing"}
# Documentation directory names
COMMON_DOC_DIRS = {"docs", "doc", "documentation", "wiki"}


class ProjectAnalysisResult(BaseModel):
    """Structured result of project intelligence analysis."""
    project_root: str
    detection_result: ProjectDetectionResult
    project_name: Optional[str] = None
    project_version: Optional[str] = None
    project_description: Optional[str] = None
    project_structure: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    development_dependencies: List[str] = Field(default_factory=list)
    available_scripts: Dict[str, str] = Field(default_factory=dict)
    entry_points: List[str] = Field(default_factory=list)
    build_configuration: Dict[str, Any] = Field(default_factory=dict)
    test_configuration: Dict[str, Any] = Field(default_factory=dict)
    documented_installation_instructions: Optional[str] = None
    documented_run_instructions: Optional[str] = None
    documented_test_instructions: Optional[str] = None
    infrastructure_configuration: Dict[str, Any] = Field(default_factory=dict)
    analysis_warnings: List[str] = Field(default_factory=list)
    files_analyzed: List[str] = Field(default_factory=list)
    evidence: Dict[str, List[str]] = Field(default_factory=dict)


class ProjectAnalyzer:
    """Read-only project intelligence engine."""

    def __init__(
        self,
        detector: Optional[ProjectDetector] = None,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_tree_depth: int = 3,
        max_tree_entries: int = 60,
    ) -> None:
        """Initializes the ProjectAnalyzer.

        Args:
            detector: Optional ProjectDetector instance to reuse.
            max_file_size_bytes: Ceiling for reading manifest files.
            max_tree_depth: Maximum directory depth for project structure tree.
            max_tree_entries: Maximum number of entries rendered in tree representation.
        """
        self.detector = detector or ProjectDetector(max_file_size_bytes=max_file_size_bytes)
        self.max_file_size_bytes = max_file_size_bytes
        self.max_tree_depth = max_tree_depth
        self.max_tree_entries = max_tree_entries

    # --------------------------------------------------------------------------
    # Main Analysis Entry Point
    # --------------------------------------------------------------------------

    def analyze(self, project_root: Union[str, Path]) -> ProjectAnalysisResult:
        """Performs deep, read-only analysis of a project directory.

        Args:
            project_root: Root directory path of the project.

        Returns:
            ProjectAnalysisResult containing structured intelligence.
        """
        root_path = Path(project_root)
        warnings: List[str] = []
        evidence: Dict[str, List[str]] = {}
        files_analyzed: List[str] = []

        # 1. Step 1: Run detection using ProjectDetector (Reusing Phase 10)
        detection_res = self.detector.detect(root_path)
        warnings.extend(detection_res.detection_warnings)

        if not detection_res.is_valid_directory:
            return ProjectAnalysisResult(
                project_root=str(root_path),
                detection_result=detection_res,
                analysis_warnings=warnings,
            )

        resolved_root = Path(detection_res.project_root)

        # 2. Step 2: Structure Analysis (Source dirs, test dirs, tree, entry points)
        structure_info = self._analyze_structure(resolved_root, warnings)

        # Initialize result properties
        project_name: Optional[str] = None
        project_version: Optional[str] = None
        project_description: Optional[str] = None
        dependencies: Set[str] = set()
        dev_dependencies: Set[str] = set()
        available_scripts: Dict[str, str] = {}
        entry_points: Set[str] = set(structure_info.get("entry_points", []))
        build_config: Dict[str, Any] = {}
        test_config: Dict[str, Any] = {}
        infra_config: Dict[str, Any] = {}

        # 3. Step 3: Manifest Analysis based on detected technologies / markers
        markers = set(detection_res.project_markers)

        # Python Manifests
        self._analyze_python_manifests(
            resolved_root, markers, dependencies, dev_dependencies, entry_points,
            build_config, files_analyzed, evidence, warnings,
            project_meta={"name": project_name, "version": project_version, "description": project_description}
        )
        if "name" in build_config and not project_name:
            project_name = build_config.pop("name")
        if "version" in build_config and not project_version:
            project_version = build_config.pop("version")
        if "description" in build_config and not project_description:
            project_description = build_config.pop("description")

        # JavaScript / TypeScript Manifests
        js_meta = self._analyze_javascript_manifests(
            resolved_root, markers, dependencies, dev_dependencies, available_scripts,
            entry_points, build_config, files_analyzed, evidence, warnings
        )
        if js_meta.get("name") and not project_name:
            project_name = js_meta["name"]
        if js_meta.get("version") and not project_version:
            project_version = js_meta["version"]
        if js_meta.get("description") and not project_description:
            project_description = js_meta["description"]

        # Java Manifests (Maven / Gradle)
        java_meta = self._analyze_java_manifests(
            resolved_root, markers, dependencies, build_config, files_analyzed, evidence, warnings
        )
        if java_meta.get("name") and not project_name:
            project_name = java_meta["name"]
        if java_meta.get("version") and not project_version:
            project_version = java_meta["version"]

        # C / C++ Manifests (CMake / Makefile)
        self._analyze_cpp_manifests(
            resolved_root, markers, build_config, files_analyzed, evidence, warnings
        )

        # Flutter / Dart Manifests (pubspec.yaml)
        dart_meta = self._analyze_dart_manifests(
            resolved_root, markers, dependencies, dev_dependencies, build_config, files_analyzed, evidence, warnings
        )
        if dart_meta.get("name") and not project_name:
            project_name = dart_meta["name"]
        if dart_meta.get("version") and not project_version:
            project_version = dart_meta["version"]
        if dart_meta.get("description") and not project_description:
            project_description = dart_meta["description"]

        # .NET Manifests (.csproj, .sln)
        self._analyze_dotnet_manifests(
            resolved_root, markers, dependencies, build_config, files_analyzed, evidence, warnings
        )

        # Docker / Infrastructure Configuration
        self._analyze_docker_manifests(
            resolved_root, markers, infra_config, files_analyzed, evidence, warnings
        )

        # 4. Step 4: README Intelligence
        readme_intel = self._analyze_readme(
            resolved_root, files_analyzed, evidence, warnings
        )
        if readme_intel.get("description") and not project_description:
            project_description = readme_intel["description"]

        # Infer test configuration
        if structure_info.get("test_dirs"):
            test_config["test_directories"] = structure_info["test_dirs"]
            self._add_evidence(evidence, "Testing", f"Found test directories: {', '.join(structure_info['test_dirs'])}")
        if "test" in available_scripts:
            test_config["npm_test_script"] = available_scripts["test"]
            self._add_evidence(evidence, "Testing", f"Declared npm test script: '{available_scripts['test']}'")
        if any("pytest" in d for d in dev_dependencies.union(dependencies)):
            test_config["framework"] = "pytest"
            self._add_evidence(evidence, "Testing", "Found pytest dependency")

        # Fallback project name to root directory name if not found in manifests
        if not project_name:
            project_name = resolved_root.name
            self._add_evidence(evidence, "Project Name", f"Inferred from directory name '{project_name}'")

        return ProjectAnalysisResult(
            project_root=str(resolved_root),
            detection_result=detection_res,
            project_name=project_name,
            project_version=project_version,
            project_description=project_description,
            project_structure=structure_info,
            dependencies=sorted(dependencies),
            development_dependencies=sorted(dev_dependencies),
            available_scripts=available_scripts,
            entry_points=sorted(entry_points),
            build_configuration=build_config,
            test_configuration=test_config,
            documented_installation_instructions=readme_intel.get("installation"),
            documented_run_instructions=readme_intel.get("run"),
            documented_test_instructions=readme_intel.get("test"),
            infrastructure_configuration=infra_config,
            analysis_warnings=sorted(set(warnings)),
            files_analyzed=sorted(set(files_analyzed)),
            evidence=evidence,
        )

    # --------------------------------------------------------------------------
    # Safe File Reading & Security Protections
    # --------------------------------------------------------------------------

    def _read_file_safe(
        self,
        file_path: Path,
        root: Path,
        warnings: List[str],
    ) -> Optional[str]:
        """Safely reads a text file enforcing size limits, symlink bounds, and secret blocking."""
        # 1. Secret Protection: Check against forbidden secret file names
        name_lower = file_path.name.lower()
        if (
            name_lower in SECRET_FILE_PATTERNS
            or name_lower.startswith(".env")
            or "credential" in name_lower
            or "secret" in name_lower
            or name_lower.endswith((".pem", ".key", ".pfx", ".p12"))
        ):
            warnings.append(f"Security: Refusing to read protected secret file '{file_path.name}'")
            return None

        # 2. Symlink boundary check
        try:
            resolved = file_path.resolve()
            if not (resolved == root or root in resolved.parents):
                warnings.append(f"Security: Skipping symlink '{file_path.name}' escaping project root")
                return None
        except Exception:
            return None

        # 3. File existence and size check
        try:
            stat = file_path.stat()
            if stat.st_size > self.max_file_size_bytes:
                warnings.append(
                    f"File '{file_path.name}' exceeds limit ({self.max_file_size_bytes} bytes). Skipped."
                )
                return None
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except PermissionError:
            warnings.append(f"Permission denied reading '{file_path.name}'")
            return None
        except Exception as e:
            warnings.append(f"Error reading '{file_path.name}': {e}")
            return None

    def _add_evidence(self, evidence_dict: Dict[str, List[str]], key: str, reason: str) -> None:
        """Appends evidence cleanly avoiding duplicates."""
        if key not in evidence_dict:
            evidence_dict[key] = []
        if reason not in evidence_dict[key]:
            evidence_dict[key].append(reason)

    # --------------------------------------------------------------------------
    # Structure & Directory Tree Analysis
    # --------------------------------------------------------------------------

    def _analyze_structure(self, root: Path, warnings: List[str]) -> Dict[str, Any]:
        """Scans directory structure to identify source, test, doc dirs, and build a concise tree."""
        source_dirs: List[str] = []
        test_dirs: List[str] = []
        doc_dirs: List[str] = []
        config_files: List[str] = []
        entry_points: List[str] = []
        tree_lines: List[str] = [f"{root.name}/"]
        entry_count = [0]  # mutable counter for tree lines

        # Common entry point candidates relative to root
        entry_point_names = {
            "app.py", "main.py", "wsgi.py", "asgi.py", "manage.py",
            "index.js", "index.ts", "main.js", "main.ts", "server.js", "server.ts",
            "App.tsx", "App.jsx", "main.go", "main.rs", "Program.cs", "main.dart",
        }

        def build_tree(current_dir: Path, current_depth: int, prefix: str) -> None:
            if current_depth > self.max_tree_depth or entry_count[0] >= self.max_tree_entries:
                return

            try:
                entries = sorted(list(os.scandir(current_dir)), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                warnings.append(f"Permission denied accessing directory: {current_dir.name}")
                return
            except OSError as e:
                warnings.append(f"Error accessing directory {current_dir.name}: {e}")
                return

            # Filter entries: remove ignored dirs and secret files
            valid_entries = []
            for e in entries:
                name = e.name
                if e.is_dir(follow_symlinks=False):
                    if name in IGNORED_DIRS:
                        continue
                    if e.is_symlink():
                        try:
                            target = Path(e.path).resolve()
                            if not (target == root or root in target.parents):
                                continue
                        except Exception:
                            continue
                    valid_entries.append(e)
                elif e.is_file(follow_symlinks=False):
                    if name.startswith(".env") or name in SECRET_FILE_PATTERNS:
                        continue
                    valid_entries.append(e)

            for idx, entry in enumerate(valid_entries):
                if entry_count[0] >= self.max_tree_entries:
                    tree_lines.append(f"{prefix}... (entries truncated)")
                    break

                is_last = (idx == len(valid_entries) - 1)
                connector = "└── " if is_last else "├── "
                sub_prefix = "    " if is_last else "│   "

                entry_count[0] += 1
                rel_path = str(Path(entry.path).relative_to(root)).replace("\\", "/")

                if entry.is_dir():
                    tree_lines.append(f"{prefix}{connector}{entry.name}/")
                    # Classify category
                    name_lower = entry.name.lower()
                    if current_depth == 0:
                        if name_lower in COMMON_SOURCE_DIRS:
                            source_dirs.append(entry.name)
                        elif name_lower in COMMON_TEST_DIRS:
                            test_dirs.append(entry.name)
                        elif name_lower in COMMON_DOC_DIRS:
                            doc_dirs.append(entry.name)

                    build_tree(Path(entry.path), current_depth + 1, prefix + sub_prefix)
                else:
                    tree_lines.append(f"{prefix}{connector}{entry.name}")
                    if entry.name in entry_point_names or rel_path.endswith(("/main.py", "/app.py", "/index.ts", "/index.js")):
                        entry_points.append(rel_path)
                    elif entry.name.endswith((".json", ".toml", ".yaml", ".yml", ".cfg", ".ini")):
                        config_files.append(rel_path)

        build_tree(root, 0, "")

        return {
            "source_dirs": sorted(source_dirs),
            "test_dirs": sorted(test_dirs),
            "doc_dirs": sorted(doc_dirs),
            "config_files": sorted(config_files),
            "entry_points": sorted(entry_points),
            "tree_representation": "\n".join(tree_lines),
        }

    # --------------------------------------------------------------------------
    # Python Manifest Analysis
    # --------------------------------------------------------------------------

    def _analyze_python_manifests(
        self,
        root: Path,
        markers: Set[str],
        dependencies: Set[str],
        dev_dependencies: Set[str],
        entry_points: Set[str],
        build_config: Dict[str, Any],
        files_analyzed: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
        project_meta: Dict[str, Any],
    ) -> None:
        """Parses pyproject.toml, requirements.txt, Pipfile, and setup.cfg safely."""
        # 1. pyproject.toml
        if "pyproject.toml" in markers or (root / "pyproject.toml").is_file():
            pyproj_path = root / "pyproject.toml"
            content = self._read_file_safe(pyproj_path, root, warnings)
            if content:
                files_analyzed.append("pyproject.toml")
                try:
                    import tomllib
                    data = tomllib.loads(content)
                    # Project metadata
                    project_sec = data.get("project", {})
                    if "name" in project_sec and not project_meta.get("name"):
                        build_config["name"] = project_sec["name"]
                        self._add_evidence(evidence, "Project Name", f"Found 'name = \"{project_sec['name']}\"' in pyproject.toml")
                    if "version" in project_sec and not project_meta.get("version"):
                        build_config["version"] = project_sec["version"]
                        self._add_evidence(evidence, "Version", f"Found 'version = \"{project_sec['version']}\"' in pyproject.toml")
                    if "description" in project_sec and not project_meta.get("description"):
                        build_config["description"] = project_sec["description"]
                    if "requires-python" in project_sec:
                        build_config["requires_python"] = project_sec["requires-python"]

                    # Build system
                    if "build-system" in data:
                        build_config["build_system"] = data["build-system"].get("build-backend", "unknown")

                    # Standard dependencies
                    for dep in project_sec.get("dependencies", []):
                        pkg = str(dep).split(";")[0].split(">=")[0].split("==")[0].split("<")[0].split("~=")[0].strip()
                        if pkg:
                            dependencies.add(pkg)

                    # Scripts / entry points
                    scripts = project_sec.get("scripts", {})
                    if isinstance(scripts, dict):
                        for s_name, s_target in scripts.items():
                            entry_points.add(f"{s_name} = {s_target}")
                            self._add_evidence(evidence, "Entry Point", f"Found console script '{s_name}' in pyproject.toml")

                    # Poetry section
                    poetry_sec = data.get("tool", {}).get("poetry", {})
                    if poetry_sec:
                        if "name" in poetry_sec and "name" not in build_config:
                            build_config["name"] = poetry_sec["name"]
                        if "version" in poetry_sec and "version" not in build_config:
                            build_config["version"] = poetry_sec["version"]
                        for dep_k in poetry_sec.get("dependencies", {}).keys():
                            if dep_k.lower() != "python":
                                dependencies.add(dep_k)
                        for dep_k in poetry_sec.get("group", {}).get("dev", {}).get("dependencies", {}).keys():
                            dev_dependencies.add(dep_k)

                    self._add_evidence(evidence, "Python Dependencies", "Extracted dependencies from pyproject.toml")
                except Exception as e:
                    warnings.append(f"Error parsing pyproject.toml: {e}")

        # 2. requirements.txt
        req_path = root / "requirements.txt"
        if req_path.is_file():
            content = self._read_file_safe(req_path, root, warnings)
            if content:
                files_analyzed.append("requirements.txt")
                count = 0
                for line in content.splitlines():
                    cleaned = line.strip()
                    if not cleaned or cleaned.startswith("#") or cleaned.startswith("-"):
                        continue
                    pkg = re.split(r"[><=~!]", cleaned)[0].strip()
                    if pkg and not pkg.startswith("git+"):
                        dependencies.add(pkg)
                        count += 1
                if count > 0:
                    self._add_evidence(evidence, "Python Dependencies", f"Extracted {count} package(s) from requirements.txt")

        # 3. Pipfile
        pipfile_path = root / "Pipfile"
        if pipfile_path.is_file():
            content = self._read_file_safe(pipfile_path, root, warnings)
            if content:
                files_analyzed.append("Pipfile")
                in_packages = False
                in_dev = False
                for line in content.splitlines():
                    line_clean = line.strip()
                    if line_clean == "[packages]":
                        in_packages, in_dev = True, False
                        continue
                    elif line_clean == "[dev-packages]":
                        in_packages, in_dev = False, True
                        continue
                    elif line_clean.startswith("["):
                        in_packages, in_dev = False, False
                        continue

                    if "=" in line_clean:
                        pkg = line_clean.split("=")[0].strip().strip('"').strip("'")
                        if in_packages:
                            dependencies.add(pkg)
                        elif in_dev:
                            dev_dependencies.add(pkg)

        # 4. setup.py (REGEX ONLY - NEVER IMPORT OR EXECUTE)
        setup_path = root / "setup.py"
        if setup_path.is_file():
            content = self._read_file_safe(setup_path, root, warnings)
            if content:
                files_analyzed.append("setup.py")
                name_match = re.search(r"name\s*=\s*['\"]([^'\"]+)['\"]", content)
                if name_match and "name" not in build_config:
                    build_config["name"] = name_match.group(1)
                    self._add_evidence(evidence, "Project Name", f"Extracted name '{name_match.group(1)}' from setup.py regex")
                ver_match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
                if ver_match and "version" not in build_config:
                    build_config["version"] = ver_match.group(1)

        # 5. setup.cfg
        setup_cfg_path = root / "setup.cfg"
        if setup_cfg_path.is_file():
            content = self._read_file_safe(setup_cfg_path, root, warnings)
            if content:
                files_analyzed.append("setup.cfg")
                name_match = re.search(r"^name\s*=\s*(.+)$", content, re.MULTILINE)
                if name_match and "name" not in build_config:
                    build_config["name"] = name_match.group(1).strip()
                ver_match = re.search(r"^version\s*=\s*(.+)$", content, re.MULTILINE)
                if ver_match and "version" not in build_config:
                    build_config["version"] = ver_match.group(1).strip()

    # --------------------------------------------------------------------------
    # JavaScript / TypeScript Manifest Analysis
    # --------------------------------------------------------------------------

    def _analyze_javascript_manifests(
        self,
        root: Path,
        markers: Set[str],
        dependencies: Set[str],
        dev_dependencies: Set[str],
        available_scripts: Dict[str, str],
        entry_points: Set[str],
        build_config: Dict[str, Any],
        files_analyzed: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Parses package.json and tsconfig.json safely."""
        meta: Dict[str, Any] = {}
        pkg_path = root / "package.json"

        if pkg_path.is_file():
            content = self._read_file_safe(pkg_path, root, warnings)
            if content:
                files_analyzed.append("package.json")
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        if "name" in data:
                            meta["name"] = str(data["name"])
                            self._add_evidence(evidence, "Project Name", f"Found 'name: \"{data['name']}\"' in package.json")
                        if "version" in data:
                            meta["version"] = str(data["version"])
                            self._add_evidence(evidence, "Version", f"Found 'version: \"{data['version']}\"' in package.json")
                        if "description" in data:
                            meta["description"] = str(data["description"])

                        # Main and module entry points
                        if "main" in data:
                            entry_points.add(str(data["main"]))
                            self._add_evidence(evidence, "Entry Point", f"Found 'main: \"{data['main']}\"' in package.json")
                        if "module" in data:
                            entry_points.add(str(data["module"]))
                        if "bin" in data:
                            if isinstance(data["bin"], str):
                                entry_points.add(str(data["bin"]))
                            elif isinstance(data["bin"], dict):
                                for bin_k, bin_v in data["bin"].items():
                                    entry_points.add(f"{bin_k} -> {bin_v}")

                        # Dependencies
                        if isinstance(data.get("dependencies"), dict):
                            for k in data["dependencies"].keys():
                                dependencies.add(k)
                            self._add_evidence(evidence, "Dependencies", f"Extracted {len(data['dependencies'])} dependency(ies) from package.json")

                        # Dev Dependencies
                        if isinstance(data.get("devDependencies"), dict):
                            for k in data["devDependencies"].keys():
                                dev_dependencies.add(k)

                        # Available Scripts (Exact declared values, NEVER executed)
                        if isinstance(data.get("scripts"), dict):
                            for s_name, s_cmd in data["scripts"].items():
                                available_scripts[s_name] = str(s_cmd)
                            self._add_evidence(
                                evidence,
                                "Available Scripts",
                                f"Extracted {len(available_scripts)} declared npm script(s): {', '.join(list(available_scripts.keys())[:5])}"
                            )

                        # Package manager field
                        if "packageManager" in data:
                            build_config["declared_package_manager"] = str(data["packageManager"])
                except json.JSONDecodeError as jde:
                    warnings.append(f"Malformed package.json (JSON error): {jde}")
                except Exception as e:
                    warnings.append(f"Error parsing package.json: {e}")

        # tsconfig.json analysis
        tsconfig_path = root / "tsconfig.json"
        if tsconfig_path.is_file():
            content = self._read_file_safe(tsconfig_path, root, warnings)
            if content:
                files_analyzed.append("tsconfig.json")
                try:
                    # Strip simple JS comments from tsconfig before parsing
                    cleaned = re.sub(r"//.*?$|/\*.*?\*/", "", content, flags=re.MULTILINE | re.DOTALL)
                    ts_data = json.loads(cleaned)
                    if isinstance(ts_data, dict):
                        co = ts_data.get("compilerOptions", {})
                        build_config["typescript_compiler_options"] = {
                            "target": co.get("target"),
                            "module": co.get("module"),
                            "strict": co.get("strict"),
                        }
                        self._add_evidence(evidence, "TypeScript Configuration", "Extracted compilerOptions from tsconfig.json")
                except Exception:
                    # Soft warning if tsconfig has complex trailing commas
                    pass

        return meta

    # --------------------------------------------------------------------------
    # Java Manifest Analysis (Maven / Gradle)
    # --------------------------------------------------------------------------

    def _analyze_java_manifests(
        self,
        root: Path,
        markers: Set[str],
        dependencies: Set[str],
        build_config: Dict[str, Any],
        files_analyzed: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Parses pom.xml and build.gradle safely."""
        meta: Dict[str, Any] = {}
        pom_path = root / "pom.xml"

        if pom_path.is_file():
            content = self._read_file_safe(pom_path, root, warnings)
            if content:
                files_analyzed.append("pom.xml")
                try:
                    # Strip namespace for easier ElementTree querying
                    clean_xml = re.sub(r'\sxmlns="[^"]+"', '', content, count=1)
                    tree = ET.fromstring(clean_xml)
                    art_id = tree.findtext("artifactId")
                    ver = tree.findtext("version")
                    if art_id:
                        meta["name"] = art_id
                        self._add_evidence(evidence, "Project Name", f"Found artifactId '{art_id}' in pom.xml")
                    if ver:
                        meta["version"] = ver

                    # Dependencies
                    for dep in tree.findall(".//dependency"):
                        g = dep.findtext("groupId") or ""
                        a = dep.findtext("artifactId") or ""
                        if a:
                            dependencies.add(f"{g}:{a}" if g else a)

                    build_config["java_build_tool"] = "Maven"
                except Exception as e:
                    warnings.append(f"Error parsing pom.xml: {e}")

        # Gradle analysis (Regex based, NO execution)
        for gradle_name in ("build.gradle", "build.gradle.kts"):
            g_path = root / gradle_name
            if g_path.is_file():
                content = self._read_file_safe(g_path, root, warnings)
                if content:
                    files_analyzed.append(gradle_name)
                    build_config["java_build_tool"] = "Gradle"
                    # Match dependencies: implementation("org.group:artifact:1.0")
                    dep_matches = re.findall(r'(?:implementation|api|compileOnly)\s*[\(\'\"]([^\'\"\)]+)[\'\"\)]', content)
                    for d in dep_matches:
                        dependencies.add(d.strip())
                    if dep_matches:
                        self._add_evidence(evidence, "Java Dependencies", f"Extracted dependencies from {gradle_name}")
                    break

        return meta

    # --------------------------------------------------------------------------
    # C / C++ Manifest Analysis
    # --------------------------------------------------------------------------

    def _analyze_cpp_manifests(
        self,
        root: Path,
        markers: Set[str],
        build_config: Dict[str, Any],
        files_analyzed: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Parses CMakeLists.txt and Makefile safely via regex."""
        cmake_path = root / "CMakeLists.txt"
        if cmake_path.is_file():
            content = self._read_file_safe(cmake_path, root, warnings)
            if content:
                files_analyzed.append("CMakeLists.txt")
                proj_match = re.search(r"project\s*\(\s*([A-Za-z0-9_\-]+)", content, re.IGNORECASE)
                if proj_match:
                    build_config["cmake_project"] = proj_match.group(1)
                    self._add_evidence(evidence, "CMake", f"Found project({proj_match.group(1)}) in CMakeLists.txt")

                exec_matches = re.findall(r"add_executable\s*\(\s*([A-Za-z0-9_\-]+)", content, re.IGNORECASE)
                if exec_matches:
                    build_config["cmake_targets"] = exec_matches

        makefile_path = root / "Makefile"
        if makefile_path.is_file():
            content = self._read_file_safe(makefile_path, root, warnings)
            if content:
                files_analyzed.append("Makefile")
                # Extract phony / standard make targets
                target_matches = re.findall(r"^([a-zA-Z0-9_\-]+)\s*:", content, re.MULTILINE)
                if target_matches:
                    build_config["make_targets"] = [t for t in target_matches if not t.startswith(".")]

    # --------------------------------------------------------------------------
    # Flutter / Dart Manifest Analysis
    # --------------------------------------------------------------------------

    def _analyze_dart_manifests(
        self,
        root: Path,
        markers: Set[str],
        dependencies: Set[str],
        dev_dependencies: Set[str],
        build_config: Dict[str, Any],
        files_analyzed: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Parses pubspec.yaml safely."""
        meta: Dict[str, Any] = {}
        pubspec_path = root / "pubspec.yaml"

        if pubspec_path.is_file():
            content = self._read_file_safe(pubspec_path, root, warnings)
            if content:
                files_analyzed.append("pubspec.yaml")
                try:
                    import yaml
                    data = yaml.safe_load(content)
                    if isinstance(data, dict):
                        if "name" in data:
                            meta["name"] = str(data["name"])
                            self._add_evidence(evidence, "Project Name", f"Found 'name: {data['name']}' in pubspec.yaml")
                        if "version" in data:
                            meta["version"] = str(data["version"])
                            self._add_evidence(evidence, "Version", f"Found 'version: {data['version']}' in pubspec.yaml")
                        if "description" in data:
                            meta["description"] = str(data["description"])

                        # Dependencies
                        deps = data.get("dependencies", {})
                        if isinstance(deps, dict):
                            for k in deps.keys():
                                dependencies.add(k)
                        dev_deps = data.get("dev_dependencies", {})
                        if isinstance(dev_deps, dict):
                            for k in dev_deps.keys():
                                dev_dependencies.add(k)

                        # Check for Flutter either as top-level section, dependency, or framework
                        if "flutter" in data or (isinstance(data.get("dependencies"), dict) and "flutter" in data["dependencies"]):
                            build_config["flutter_project"] = True
                except Exception:
                    # Regex fallback for pubspec
                    name_m = re.search(r"^name:\s*([^\s]+)", content, re.MULTILINE)
                    if name_m:
                        meta["name"] = name_m.group(1).strip()
                    ver_m = re.search(r"^version:\s*([^\s]+)", content, re.MULTILINE)
                    if ver_m:
                        meta["version"] = ver_m.group(1).strip()
                    if "sdk: flutter" in content or re.search(r"^\s*flutter:\s*$", content, re.MULTILINE):
                        build_config["flutter_project"] = True

        return meta

    # --------------------------------------------------------------------------
    # .NET Manifest Analysis
    # --------------------------------------------------------------------------

    def _analyze_dotnet_manifests(
        self,
        root: Path,
        markers: Set[str],
        dependencies: Set[str],
        build_config: Dict[str, Any],
        files_analyzed: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Parses .csproj and .fsproj files safely using ElementTree."""
        for cs_name in [m for m in markers if m.endswith(".csproj") or m.endswith(".fsproj")]:
            cs_path = root / cs_name
            if cs_path.is_file():
                content = self._read_file_safe(cs_path, root, warnings)
                if content:
                    files_analyzed.append(cs_name)
                    try:
                        tree = ET.fromstring(content)
                        tf = tree.findtext(".//TargetFramework")
                        if tf:
                            build_config["dotnet_target_framework"] = tf
                            self._add_evidence(evidence, ".NET", f"Target Framework: {tf}")
                        for pkg in tree.findall(".//PackageReference"):
                            inc = pkg.get("Include")
                            ver = pkg.get("Version")
                            if inc:
                                dependencies.add(f"{inc} ({ver})" if ver else inc)
                    except Exception as e:
                        warnings.append(f"Error parsing .NET project file {cs_name}: {e}")

    # --------------------------------------------------------------------------
    # Docker Manifest Analysis
    # --------------------------------------------------------------------------

    def _analyze_docker_manifests(
        self,
        root: Path,
        markers: Set[str],
        infra_config: Dict[str, Any],
        files_analyzed: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Analyzes Dockerfile and Docker Compose configurations safely."""
        # Dockerfile
        df_path = root / "Dockerfile"
        if df_path.is_file():
            content = self._read_file_safe(df_path, root, warnings)
            if content:
                files_analyzed.append("Dockerfile")
                base_images = re.findall(r"^FROM\s+([^\s]+)", content, re.MULTILINE | re.IGNORECASE)
                ports = re.findall(r"^EXPOSE\s+([^\s]+)", content, re.MULTILINE | re.IGNORECASE)
                if base_images:
                    infra_config["dockerfile_base_images"] = base_images
                    self._add_evidence(evidence, "Docker", f"Base image(s): {', '.join(base_images)}")
                if ports:
                    infra_config["dockerfile_exposed_ports"] = ports

        # Docker Compose
        for comp_name in ("compose.yaml", "compose.yml", "docker-compose.yml", "docker-compose.yaml"):
            comp_path = root / comp_name
            if comp_path.is_file():
                content = self._read_file_safe(comp_path, root, warnings)
                if content:
                    files_analyzed.append(comp_name)
                    try:
                        import yaml
                        data = yaml.safe_load(content)
                        if isinstance(data, dict) and "services" in data:
                            services = list(data["services"].keys())
                            infra_config["docker_compose_services"] = services
                            self._add_evidence(evidence, "Docker Compose", f"Defined service(s): {', '.join(services)}")
                    except Exception:
                        # Regex fallback for service names
                        pass
                    break

    # --------------------------------------------------------------------------
    # README Intelligence Analysis
    # --------------------------------------------------------------------------

    def _analyze_readme(
        self,
        root: Path,
        files_analyzed: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> Dict[str, Optional[str]]:
        """Extracts documented description, install, run, and test instructions safely.

        Treats all README contents as untrusted data to protect against prompt injection.
        """
        result: Dict[str, Optional[str]] = {
            "description": None,
            "installation": None,
            "run": None,
            "test": None,
        }

        # Find prioritized README
        readme_path: Optional[Path] = None
        for cand in ("README.md", "README.rst", "README.txt", "readme.md", "README"):
            p = root / cand
            if p.is_file():
                readme_path = p
                break

        if not readme_path:
            return result

        content = self._read_file_safe(readme_path, root, warnings)
        if not content:
            return result

        files_analyzed.append(readme_path.name)
        self._add_evidence(evidence, "Documentation", f"Analyzed {readme_path.name} for developer guidance")

        # Untrusted text sanitizer: strip any control characters or prompt-injection attempts
        clean_text = self._sanitize_untrusted_markdown(content)

        # 1. Extract Description (First non-header paragraph)
        paragraphs = clean_text.split("\n\n")
        for para in paragraphs:
            para_strip = para.strip()
            if para_strip and not para_strip.startswith("#") and not para_strip.startswith("[!"):
                result["description"] = para_strip.replace("\n", " ")[:300]
                break

        # 2. Extract Documented Instructions by Markdown Sections
        sections = self._split_markdown_sections(clean_text)

        # Installation: prioritize 'install' over 'setup'/'prerequisite'
        for heading, body in sections.items():
            if "install" in heading.lower():
                result["installation"] = body[:500].strip()
                break
        if not result["installation"]:
            for heading, body in sections.items():
                if any(k in heading.lower() for k in ("setup", "getting started", "prerequisite")):
                    result["installation"] = body[:500].strip()
                    break

        # Run / Usage: prioritize 'run' / 'usage' / 'start'
        for heading, body in sections.items():
            if any(k in heading.lower() for k in ("usage", "run", "start", "launch", "how to use")):
                result["run"] = body[:500].strip()
                break
        if not result["run"]:
            for heading, body in sections.items():
                if any(k in heading.lower() for k in ("development", "quickstart")):
                    result["run"] = body[:500].strip()
                    break

        # Test: prioritize 'test' / 'testing'
        for heading, body in sections.items():
            if any(k in heading.lower() for k in ("test", "testing", "evaluation", "benchmark")):
                result["test"] = body[:500].strip()
                break

        return result

    def _sanitize_untrusted_markdown(self, text: str) -> str:
        """Sanitizes untrusted documentation text to prevent prompt injection."""
        # Replace null bytes or suspicious control sequences
        sanitized = text.replace("\0", "").replace("\r\n", "\n")
        # Prepend explicit safety indicator if dangerous keywords detected
        if re.search(r"(ignore\s+(previous|all)\s+instructions|system\s+prompt|execute\s+as\s+admin)", sanitized, re.IGNORECASE):
            logger.warning("Potential prompt-injection pattern neutralized in documentation text.")
        return sanitized

    def _split_markdown_sections(self, text: str) -> Dict[str, str]:
        """Splits markdown content into heading -> body mapping."""
        sections: Dict[str, str] = {}
        current_heading: Optional[str] = None
        current_body: List[str] = []

        for line in text.splitlines():
            header_match = re.match(r"^#{1,3}\s+(.+)$", line.strip())
            if header_match:
                if current_heading:
                    sections[current_heading] = "\n".join(current_body)
                current_heading = header_match.group(1).strip()
                current_body = []
            elif current_heading is not None:
                current_body.append(line)

        if current_heading and current_body:
            sections[current_heading] = "\n".join(current_body)

        return sections
