"""Project Detection Engine module for AI Terminal Assistant / Copilot.

Analyzes a target directory structure, project manifests, and configuration markers
to identify languages, frameworks, package managers, build tools, infrastructure,
and monorepo configurations in a strictly read-only, safe manner.
"""

from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Max file size to inspect (1 MB) to prevent memory exhaustion
DEFAULT_MAX_FILE_SIZE_BYTES = 1024 * 1024

# Ignored directory names that should never be traversed
IGNORED_DIRS: Set[str] = {
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".tox",
    "bin",
    "obj",
    "out",
    "coverage",
    ".mypy_cache",
    ".next",
    ".nuxt",
    ".output",
    ".cache",
}


class DetectionConfidence(str, Enum):
    """Confidence level of a specific technology detection."""
    HIGH = "HIGH"        # Explicit manifest or configuration found
    MEDIUM = "MEDIUM"    # Indirect dependency or strong file pattern
    LOW = "LOW"          # Heuristic or generic marker


class DetectionStatus(str, Enum):
    """Lifecycle status of the project detection scan."""
    SUCCESS = "SUCCESS"
    NO_TECHNOLOGIES_DETECTED = "NO_TECHNOLOGIES_DETECTED"
    INVALID_DIRECTORY = "INVALID_DIRECTORY"
    ACCESS_DENIED = "ACCESS_DENIED"


class ProjectDetectionResult(BaseModel):
    """Structured result of project detection analysis."""
    project_root: str
    is_valid_directory: bool
    detection_status: DetectionStatus
    primary_language: Optional[str] = None
    detected_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    build_tools: List[str] = Field(default_factory=list)
    package_managers: List[str] = Field(default_factory=list)
    infrastructure: List[str] = Field(default_factory=list)
    project_markers: List[str] = Field(default_factory=list)
    evidence: Dict[str, List[str]] = Field(default_factory=dict)
    detection_warnings: List[str] = Field(default_factory=list)
    is_monorepo: bool = False
    monorepo_type: Optional[str] = None
    has_git: bool = False


class ProjectDetector:
    """Read-only software project detection engine."""

    def __init__(
        self,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_depth: int = 2,
    ) -> None:
        """Initializes the ProjectDetector.

        Args:
            max_file_size_bytes: Maximum allowed bytes to read from a manifest file.
            max_depth: Maximum directory depth for subproject / monorepo scanning.
        """
        self.max_file_size_bytes = max_file_size_bytes
        self.max_depth = max_depth

    # --------------------------------------------------------------------------
    # Main Detection Entry Point
    # --------------------------------------------------------------------------

    def detect(self, project_root: Union[str, Path]) -> ProjectDetectionResult:
        """Analyzes a project directory and returns a structured detection result.

        Args:
            project_root: Path to the root directory of the software project.

        Returns:
            ProjectDetectionResult detailing detected technologies and evidence.
        """
        root_path = Path(project_root)
        warnings: List[str] = []

        # 1. Directory Validation
        is_valid, validation_status, root_resolved = self._validate_directory(root_path, warnings)
        if not is_valid or root_resolved is None:
            return ProjectDetectionResult(
                project_root=str(root_path),
                is_valid_directory=False,
                detection_status=validation_status,
                detection_warnings=warnings,
            )

        evidence: Dict[str, List[str]] = {}
        markers: List[str] = []
        languages: Set[str] = set()
        frameworks: Set[str] = set()
        build_tools: Set[str] = set()
        package_managers: Set[str] = set()
        infrastructure: Set[str] = set()

        # 2. Gather top-level and subproject files safely
        discovered_files, dir_structure = self._collect_relevant_files(root_resolved, warnings)

        # 3. Detect Git Repository
        has_git = self._detect_git(root_resolved, markers, evidence)

        # 4. Detect Monorepo Indicators
        is_monorepo, monorepo_type = self._detect_monorepo(
            root_resolved, discovered_files, markers, evidence
        )

        # 5. Technology Detections
        # Python
        self._detect_python(
            root_resolved, discovered_files, languages, frameworks, build_tools, package_managers, markers, evidence, warnings
        )

        # JavaScript / TypeScript
        self._detect_javascript_typescript(
            root_resolved, discovered_files, languages, frameworks, build_tools, package_managers, markers, evidence, warnings
        )

        # Java
        self._detect_java(
            root_resolved, discovered_files, languages, frameworks, build_tools, package_managers, markers, evidence, warnings
        )

        # C / C++
        self._detect_cpp(
            root_resolved, discovered_files, languages, frameworks, build_tools, package_managers, markers, evidence, warnings
        )

        # Flutter / Dart
        self._detect_dart_flutter(
            root_resolved, discovered_files, languages, frameworks, build_tools, package_managers, markers, evidence, warnings
        )

        # .NET / C# / F#
        self._detect_dotnet(
            root_resolved, discovered_files, languages, frameworks, build_tools, package_managers, markers, evidence, warnings
        )

        # Docker / Infrastructure
        self._detect_docker(
            root_resolved, discovered_files, infrastructure, markers, evidence
        )

        # 6. Determine Primary Language
        sorted_languages = sorted(languages)
        primary_language = self._determine_primary_language(sorted_languages, evidence)

        status = (
            DetectionStatus.SUCCESS
            if (sorted_languages or frameworks or build_tools or infrastructure or has_git)
            else DetectionStatus.NO_TECHNOLOGIES_DETECTED
        )

        return ProjectDetectionResult(
            project_root=str(root_resolved),
            is_valid_directory=True,
            detection_status=status,
            primary_language=primary_language,
            detected_languages=sorted_languages,
            frameworks=sorted(frameworks),
            build_tools=sorted(build_tools),
            package_managers=sorted(package_managers),
            infrastructure=sorted(infrastructure),
            project_markers=sorted(set(markers)),
            evidence=evidence,
            detection_warnings=warnings,
            is_monorepo=is_monorepo,
            monorepo_type=monorepo_type,
            has_git=has_git,
        )

    # --------------------------------------------------------------------------
    # Validation & File Collection Helpers
    # --------------------------------------------------------------------------

    def _validate_directory(
        self,
        path: Path,
        warnings: List[str],
    ) -> Tuple[bool, DetectionStatus, Optional[Path]]:
        """Verifies that the target path exists, is a directory, and is readable."""
        try:
            if not path.exists():
                warnings.append(f"Directory does not exist: {path}")
                return False, DetectionStatus.INVALID_DIRECTORY, None
        except (PermissionError, OSError) as e:
            warnings.append(f"Access error checking existence of {path}: {e}")
            return False, DetectionStatus.ACCESS_DENIED, None

        try:
            resolved = path.resolve()
        except Exception as e:
            warnings.append(f"Could not resolve directory path {path}: {e}")
            return False, DetectionStatus.INVALID_DIRECTORY, None

        if not resolved.is_dir():
            warnings.append(f"Target path is not a directory: {resolved}")
            return False, DetectionStatus.INVALID_DIRECTORY, None

        # Test accessibility / listdir permission
        try:
            _ = list(os.scandir(resolved))
        except PermissionError:
            warnings.append(f"Permission denied reading directory: {resolved}")
            return False, DetectionStatus.ACCESS_DENIED, None
        except Exception as e:
            warnings.append(f"Error accessing directory: {e}")
            return False, DetectionStatus.ACCESS_DENIED, None

        return True, DetectionStatus.SUCCESS, resolved

    def _collect_relevant_files(
        self,
        root: Path,
        warnings: List[str],
    ) -> Tuple[Dict[str, Path], Set[str]]:
        """Safely scans root and immediate child directories up to max_depth.

        Returns:
            Tuple of (dict mapping relative_path_str -> Path object, set of subdirectory names).
        """
        discovered: Dict[str, Path] = {}
        subdirs: Set[str] = set()

        def scan_dir(current_dir: Path, current_depth: int) -> None:
            if current_depth > self.max_depth:
                return

            try:
                entries = list(os.scandir(current_dir))
            except PermissionError:
                rel = current_dir.relative_to(root)
                warnings.append(f"Permission denied accessing subfolder: {rel}")
                return
            except OSError as e:
                warnings.append(f"Error scanning folder {current_dir.name}: {e}")
                return

            for entry in entries:
                name = entry.name
                # Check ignored directory names
                if entry.is_dir(follow_symlinks=False):
                    if name in IGNORED_DIRS:
                        continue

                    # Handle symlinks safely: prevent escaping the project root
                    if entry.is_symlink():
                        try:
                            sym_target = Path(entry.path).resolve()
                            if not (sym_target == root or root in sym_target.parents):
                                warnings.append(
                                    f"Skipping symlink '{name}' pointing outside project root ({sym_target})"
                                )
                                continue
                        except Exception:
                            continue

                    subdirs.add(name)
                    scan_dir(Path(entry.path), current_depth + 1)

                elif entry.is_file(follow_symlinks=False):
                    try:
                        p = Path(entry.path)
                        rel_path = str(p.relative_to(root)).replace("\\", "/")
                        discovered[rel_path] = p
                    except Exception:
                        pass

        scan_dir(root, 0)
        return discovered, subdirs

    def _read_file_safe(self, path: Path, warnings: List[str]) -> Optional[str]:
        """Safely reads a text file with size limit and exception handling."""
        try:
            stat = path.stat()
            if stat.st_size > self.max_file_size_bytes:
                warnings.append(
                    f"File '{path.name}' exceeds {self.max_file_size_bytes} bytes limit. Skipped deep inspection."
                )
                return None
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except PermissionError:
            warnings.append(f"Permission denied reading file: {path.name}")
            return None
        except Exception as e:
            warnings.append(f"Error reading file {path.name}: {e}")
            return None

    def _add_evidence(self, evidence_dict: Dict[str, List[str]], key: str, reason: str) -> None:
        """Appends evidence reason cleanly."""
        if key not in evidence_dict:
            evidence_dict[key] = []
        if reason not in evidence_dict[key]:
            evidence_dict[key].append(reason)

    # --------------------------------------------------------------------------
    # Git & Monorepo Detection
    # --------------------------------------------------------------------------

    def _detect_git(
        self,
        root: Path,
        markers: List[str],
        evidence: Dict[str, List[str]],
    ) -> bool:
        """Checks for .git directory or file."""
        git_path = root / ".git"
        if git_path.exists():
            markers.append(".git")
            self._add_evidence(evidence, "Git", "Found .git directory/file at project root")
            return True
        return False

    def _detect_monorepo(
        self,
        root: Path,
        files: Dict[str, Path],
        markers: List[str],
        evidence: Dict[str, List[str]],
    ) -> Tuple[bool, Optional[str]]:
        """Identifies monorepo configurations."""
        if "pnpm-workspace.yaml" in files:
            markers.append("pnpm-workspace.yaml")
            self._add_evidence(evidence, "Monorepo", "Found pnpm-workspace.yaml")
            return True, "pnpm Workspaces"

        if "lerna.json" in files:
            markers.append("lerna.json")
            self._add_evidence(evidence, "Monorepo", "Found lerna.json")
            return True, "Lerna"

        if "nx.json" in files:
            markers.append("nx.json")
            self._add_evidence(evidence, "Monorepo", "Found nx.json")
            return True, "Nx"

        if "turbo.json" in files:
            markers.append("turbo.json")
            self._add_evidence(evidence, "Monorepo", "Found turbo.json")
            return True, "Turborepo"

        # Check root package.json for workspaces field
        if "package.json" in files:
            content = self._read_file_safe(files["package.json"], [])
            if content:
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "workspaces" in data:
                        markers.append("package.json (workspaces)")
                        self._add_evidence(evidence, "Monorepo", "Root package.json contains 'workspaces' configuration")
                        return True, "npm/yarn/pnpm Workspaces"
                except Exception:
                    pass

        # Check for multiple package.json in subfolders (e.g. packages/ or apps/)
        subproject_manifests = [
            f for f in files.keys()
            if "/" in f and (f.endswith("/package.json") or f.endswith("/pyproject.toml") or f.endswith("/pom.xml"))
        ]
        if len(subproject_manifests) >= 2:
            self._add_evidence(evidence, "Monorepo", f"Found multiple subproject manifests: {', '.join(subproject_manifests[:3])}")
            return True, "Multi-package layout"

        return False, None

    # --------------------------------------------------------------------------
    # Python Detection
    # --------------------------------------------------------------------------

    def _detect_python(
        self,
        root: Path,
        files: Dict[str, Path],
        languages: Set[str],
        frameworks: Set[str],
        build_tools: Set[str],
        package_managers: Set[str],
        markers: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Detects Python language, build tools, package managers, and frameworks."""
        python_markers = [
            "pyproject.toml",
            "requirements.txt",
            "Pipfile",
            "setup.py",
            "setup.cfg",
            "poetry.lock",
            "Pipfile.lock",
        ]

        found_python_markers = [m for m in python_markers if m in files]
        # Also check for .py files in root
        py_files = [f for f in files.keys() if f.endswith(".py") and "/" not in f]

        if not found_python_markers and not py_files:
            return

        languages.add("Python")
        for m in found_python_markers:
            markers.append(m)
            self._add_evidence(evidence, "Python", f"Found Python manifest: {m}")

        # Package managers & build tools
        if "Pipfile" in files or "Pipfile.lock" in files:
            package_managers.add("pipenv")
            self._add_evidence(evidence, "pipenv", "Found Pipfile / Pipfile.lock")

        if "requirements.txt" in files:
            package_managers.add("pip")
            self._add_evidence(evidence, "pip", "Found requirements.txt")

        if "setup.py" in files or "setup.cfg" in files:
            build_tools.add("setuptools")
            self._add_evidence(evidence, "setuptools", "Found setup.py / setup.cfg")

        # Inspect pyproject.toml
        if "pyproject.toml" in files:
            content = self._read_file_safe(files["pyproject.toml"], warnings)
            if content:
                self._inspect_pyproject_toml(
                    content, frameworks, build_tools, package_managers, evidence
                )

        # Inspect requirements.txt for frameworks
        if "requirements.txt" in files:
            content = self._read_file_safe(files["requirements.txt"], warnings)
            if content:
                self._inspect_python_requirements(content, frameworks, evidence)

    def _inspect_pyproject_toml(
        self,
        content: str,
        frameworks: Set[str],
        build_tools: Set[str],
        package_managers: Set[str],
        evidence: Dict[str, List[str]],
    ) -> None:
        """Parses pyproject.toml to extract tools and dependencies."""
        try:
            import tomllib
            data = tomllib.loads(content)
        except Exception:
            # Fallback regex if malformed TOML
            if "poetry" in content.lower():
                build_tools.add("poetry")
                package_managers.add("poetry")
                self._add_evidence(evidence, "poetry", "Found poetry reference in pyproject.toml")
            return

        # Check build system
        build_backend = (
            data.get("build-system", {}).get("build-backend", "").lower()
        )
        if "poetry" in build_backend or "tool" in data and "poetry" in data["tool"]:
            build_tools.add("poetry")
            package_managers.add("poetry")
            self._add_evidence(evidence, "poetry", "Found Poetry configuration in pyproject.toml")
        elif "flit" in build_backend:
            build_tools.add("flit")
            self._add_evidence(evidence, "flit", "Found Flit build system in pyproject.toml")
        elif "hatch" in build_backend:
            build_tools.add("hatch")
            self._add_evidence(evidence, "hatch", "Found Hatch build system in pyproject.toml")
        elif "setuptools" in build_backend:
            build_tools.add("setuptools")
            self._add_evidence(evidence, "setuptools", "Found setuptools build system in pyproject.toml")

        # Check dependencies for frameworks
        all_deps: List[str] = []
        # PEP 621 dependencies
        project_deps = data.get("project", {}).get("dependencies", [])
        if isinstance(project_deps, list):
            all_deps.extend(str(d).lower() for d in project_deps)

        # Poetry dependencies
        poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        if isinstance(poetry_deps, dict):
            all_deps.extend(k.lower() for k in poetry_deps.keys())

        deps_text = " ".join(all_deps)
        self._check_python_framework_keywords(deps_text, frameworks, evidence, source="pyproject.toml")

    def _inspect_python_requirements(
        self,
        content: str,
        frameworks: Set[str],
        evidence: Dict[str, List[str]],
    ) -> None:
        """Inspects requirements.txt lines for framework dependencies."""
        self._check_python_framework_keywords(content.lower(), frameworks, evidence, source="requirements.txt")

    def _check_python_framework_keywords(
        self,
        text: str,
        frameworks: Set[str],
        evidence: Dict[str, List[str]],
        source: str,
    ) -> None:
        """Maps package names to standard Python frameworks."""
        framework_keywords = {
            "django": "Django",
            "flask": "Flask",
            "fastapi": "FastAPI",
            "tornado": "Tornado",
            "pyramid": "Pyramid",
            "streamlit": "Streamlit",
            "celery": "Celery",
        }
        for kw, fw in framework_keywords.items():
            # Match word boundary
            if re.search(rf"\b{re.escape(kw)}\b", text):
                frameworks.add(fw)
                self._add_evidence(evidence, fw, f"Found '{kw}' dependency in {source}")

    # --------------------------------------------------------------------------
    # JavaScript / TypeScript Detection
    # --------------------------------------------------------------------------

    def _detect_javascript_typescript(
        self,
        root: Path,
        files: Dict[str, Path],
        languages: Set[str],
        frameworks: Set[str],
        build_tools: Set[str],
        package_managers: Set[str],
        markers: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Detects JS/TS, package managers, and UI/backend frameworks."""
        has_package_json = "package.json" in files
        has_tsconfig = "tsconfig.json" in files

        if not has_package_json and not has_tsconfig:
            # Check for standalone JS or TS files
            has_js = any(f.endswith(".js") or f.endswith(".mjs") for f in files.keys())
            has_ts = any(f.endswith(".ts") for f in files.keys())
            if has_js:
                languages.add("JavaScript")
            if has_ts:
                languages.add("TypeScript")
            return

        if has_package_json:
            markers.append("package.json")
            languages.add("JavaScript")
            self._add_evidence(evidence, "JavaScript", "Found package.json manifest")

        if has_tsconfig:
            markers.append("tsconfig.json")
            languages.add("TypeScript")
            self._add_evidence(evidence, "TypeScript", "Found tsconfig.json configuration")

        # Package manager lockfile detection
        if "package-lock.json" in files:
            package_managers.add("npm")
            markers.append("package-lock.json")
            self._add_evidence(evidence, "npm", "Found package-lock.json")
        if "yarn.lock" in files:
            package_managers.add("yarn")
            markers.append("yarn.lock")
            self._add_evidence(evidence, "yarn", "Found yarn.lock")
        if "pnpm-lock.yaml" in files:
            package_managers.add("pnpm")
            markers.append("pnpm-lock.yaml")
            self._add_evidence(evidence, "pnpm", "Found pnpm-lock.yaml")
        if "bun.lockb" in files or "bun.lock" in files:
            package_managers.add("bun")
            markers.append("bun.lockb" if "bun.lockb" in files else "bun.lock")
            self._add_evidence(evidence, "bun", "Found Bun lockfile")

        # Deep inspection of package.json for frameworks and build tools
        if has_package_json:
            content = self._read_file_safe(files["package.json"], warnings)
            if content:
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        self._inspect_package_json_deps(
                            data, languages, frameworks, build_tools, evidence
                        )
                except json.JSONDecodeError as jde:
                    warnings.append(f"Malformed package.json (JSON parse error): {jde}")
                except Exception as e:
                    warnings.append(f"Error parsing package.json: {e}")

    def _inspect_package_json_deps(
        self,
        data: Dict[str, Any],
        languages: Set[str],
        frameworks: Set[str],
        build_tools: Set[str],
        evidence: Dict[str, List[str]],
    ) -> None:
        """Inspects package.json dependencies and devDependencies."""
        deps: Dict[str, str] = {}
        if isinstance(data.get("dependencies"), dict):
            deps.update(data["dependencies"])
        if isinstance(data.get("devDependencies"), dict):
            deps.update(data["devDependencies"])

        # Check for TypeScript in dependencies
        if "typescript" in deps:
            languages.add("TypeScript")
            self._add_evidence(evidence, "TypeScript", "Found 'typescript' in package dependencies")

        # Framework detection mappings (key dependency -> Framework name)
        framework_map = {
            "react": "React",
            "react-dom": "React",
            "vue": "Vue",
            "next": "Next.js",
            "@angular/core": "Angular",
            "svelte": "Svelte",
            "express": "Express",
            "@nestjs/core": "NestJS",
            "nuxt": "Nuxt",
            "gatsby": "Gatsby",
            "electron": "Electron",
            "react-native": "React Native",
            "fastify": "Fastify",
        }

        for dep_key, fw_name in framework_map.items():
            if dep_key in deps:
                frameworks.add(fw_name)
                self._add_evidence(evidence, fw_name, f"Found dependency '{dep_key}' in package.json")

        # Build tools detection
        build_tool_map = {
            "webpack": "Webpack",
            "vite": "Vite",
            "rollup": "Rollup",
            "esbuild": "esbuild",
            "turbo": "Turborepo",
            "@babel/core": "Babel",
            "parcel": "Parcel",
        }
        for tool_key, tool_name in build_tool_map.items():
            if tool_key in deps:
                build_tools.add(tool_name)
                self._add_evidence(evidence, tool_name, f"Found build tool '{tool_key}' in package.json")

    # --------------------------------------------------------------------------
    # Java Detection
    # --------------------------------------------------------------------------

    def _detect_java(
        self,
        root: Path,
        files: Dict[str, Path],
        languages: Set[str],
        frameworks: Set[str],
        build_tools: Set[str],
        package_managers: Set[str],
        markers: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Detects Java projects (Maven and Gradle)."""
        has_maven = "pom.xml" in files
        has_gradle = any(f in files for f in ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"))

        if not has_maven and not has_gradle:
            # Check for .java files
            if any(f.endswith(".java") for f in files.keys()):
                languages.add("Java")
            return

        languages.add("Java")

        if has_maven:
            markers.append("pom.xml")
            build_tools.add("Maven")
            package_managers.add("Maven")
            self._add_evidence(evidence, "Java", "Found pom.xml Maven manifest")
            self._add_evidence(evidence, "Maven", "Found pom.xml")

            content = self._read_file_safe(files["pom.xml"], warnings)
            if content:
                content_lower = content.lower()
                if "spring-boot" in content_lower or "springframework.boot" in content_lower:
                    frameworks.add("Spring Boot")
                    self._add_evidence(evidence, "Spring Boot", "Found Spring Boot dependency in pom.xml")
                if "quarkus" in content_lower:
                    frameworks.add("Quarkus")
                    self._add_evidence(evidence, "Quarkus", "Found Quarkus in pom.xml")
                if "micronaut" in content_lower:
                    frameworks.add("Micronaut")
                    self._add_evidence(evidence, "Micronaut", "Found Micronaut in pom.xml")

        if has_gradle:
            gradle_marker = "build.gradle.kts" if "build.gradle.kts" in files else "build.gradle"
            markers.append(gradle_marker)
            build_tools.add("Gradle")
            package_managers.add("Gradle")
            self._add_evidence(evidence, "Java", f"Found Gradle build manifest: {gradle_marker}")
            self._add_evidence(evidence, "Gradle", f"Found {gradle_marker}")

            if "build.gradle.kts" in files:
                languages.add("Kotlin")
                self._add_evidence(evidence, "Kotlin", "Found Kotlin DSL build script (build.gradle.kts)")

            content = self._read_file_safe(files[gradle_marker], warnings)
            if content:
                content_lower = content.lower()
                if "spring-boot" in content_lower or "springframework.boot" in content_lower:
                    frameworks.add("Spring Boot")
                    self._add_evidence(evidence, "Spring Boot", f"Found Spring Boot in {gradle_marker}")
                if "quarkus" in content_lower:
                    frameworks.add("Quarkus")
                    self._add_evidence(evidence, "Quarkus", f"Found Quarkus in {gradle_marker}")
                if "micronaut" in content_lower:
                    frameworks.add("Micronaut")
                    self._add_evidence(evidence, "Micronaut", f"Found Micronaut in {gradle_marker}")

    # --------------------------------------------------------------------------
    # C / C++ Detection
    # --------------------------------------------------------------------------

    def _detect_cpp(
        self,
        root: Path,
        files: Dict[str, Path],
        languages: Set[str],
        frameworks: Set[str],
        build_tools: Set[str],
        package_managers: Set[str],
        markers: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Detects C / C++ projects via CMake, Makefile, Meson."""
        has_cmake = "CMakeLists.txt" in files
        has_makefile = "Makefile" in files or "makefile" in files
        has_meson = "meson.build" in files

        # Source file presence
        has_cpp_src = any(f.endswith((".cpp", ".cxx", ".cc", ".hpp", ".hxx")) for f in files.keys())
        has_c_src = any(f.endswith((".c", ".h")) for f in files.keys())

        if not (has_cmake or has_makefile or has_meson or has_cpp_src or has_c_src):
            return

        # Determine C vs C++
        if has_cpp_src:
            languages.add("C++")
            self._add_evidence(evidence, "C++", "Found C++ source/header files (.cpp, .hpp, .cc)")
        if has_c_src and not has_cpp_src:
            languages.add("C")
            self._add_evidence(evidence, "C", "Found C source/header files (.c, .h)")

        if has_cmake:
            markers.append("CMakeLists.txt")
            build_tools.add("CMake")
            self._add_evidence(evidence, "CMake", "Found CMakeLists.txt")
            # If no source files found yet, infer C/C++ from CMakeLists
            if "C++" not in languages and "C" not in languages:
                languages.add("C/C++")
                self._add_evidence(evidence, "C/C++", "Inferred from CMakeLists.txt build file")

        if has_makefile:
            m_name = "Makefile" if "Makefile" in files else "makefile"
            markers.append(m_name)
            build_tools.add("Make")
            self._add_evidence(evidence, "Make", f"Found {m_name}")
            if "C++" not in languages and "C" not in languages and (has_cpp_src or has_c_src):
                languages.add("C/C++")

        if has_meson:
            markers.append("meson.build")
            build_tools.add("Meson")
            self._add_evidence(evidence, "Meson", "Found meson.build")
            if "C++" not in languages and "C" not in languages:
                languages.add("C/C++")

    # --------------------------------------------------------------------------
    # Flutter / Dart Detection
    # --------------------------------------------------------------------------

    def _detect_dart_flutter(
        self,
        root: Path,
        files: Dict[str, Path],
        languages: Set[str],
        frameworks: Set[str],
        build_tools: Set[str],
        package_managers: Set[str],
        markers: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Detects Dart and Flutter specifically when pubspec.yaml indicates Flutter."""
        if "pubspec.yaml" not in files:
            if any(f.endswith(".dart") for f in files.keys()):
                languages.add("Dart")
            return

        markers.append("pubspec.yaml")
        languages.add("Dart")
        package_managers.add("pub")
        self._add_evidence(evidence, "Dart", "Found pubspec.yaml manifest")

        content = self._read_file_safe(files["pubspec.yaml"], warnings)
        if not content:
            return

        # Check for Flutter in pubspec.yaml
        is_flutter = False
        try:
            import yaml
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                # 1. Check flutter section
                if "flutter" in data:
                    is_flutter = True
                # 2. Check dependencies for flutter sdk
                deps = data.get("dependencies", {})
                if isinstance(deps, dict) and "flutter" in deps:
                    is_flutter = True
        except Exception:
            # Fallback regex search
            if re.search(r"^\s*flutter:\s*$", content, re.MULTILINE) or "sdk: flutter" in content:
                is_flutter = True

        if is_flutter:
            frameworks.add("Flutter")
            build_tools.add("Flutter CLI")
            self._add_evidence(evidence, "Flutter", "Manifest specifies Flutter SDK dependency / configuration")

    # --------------------------------------------------------------------------
    # .NET Detection
    # --------------------------------------------------------------------------

    def _detect_dotnet(
        self,
        root: Path,
        files: Dict[str, Path],
        languages: Set[str],
        frameworks: Set[str],
        build_tools: Set[str],
        package_managers: Set[str],
        markers: List[str],
        evidence: Dict[str, List[str]],
        warnings: List[str],
    ) -> None:
        """Detects .NET, C#, F#, solutions, and project files."""
        csproj_files = [f for f in files.keys() if f.endswith(".csproj")]
        fsproj_files = [f for f in files.keys() if f.endswith(".fsproj")]
        sln_files = [f for f in files.keys() if f.endswith(".sln")]

        if not csproj_files and not fsproj_files and not sln_files:
            return

        build_tools.add("dotnet")
        package_managers.add("NuGet")

        if csproj_files:
            languages.add("C#")
            for cs in csproj_files:
                markers.append(cs)
                self._add_evidence(evidence, "C#", f"Found C# project manifest: {cs}")
                self._add_evidence(evidence, ".NET", f"Found .NET project file: {cs}")

        if fsproj_files:
            languages.add("F#")
            for fs in fsproj_files:
                markers.append(fs)
                self._add_evidence(evidence, "F#", f"Found F# project manifest: {fs}")
                self._add_evidence(evidence, ".NET", f"Found .NET project file: {fs}")

        if sln_files:
            for sln in sln_files:
                markers.append(sln)
                self._add_evidence(evidence, ".NET", f"Found Visual Studio solution file: {sln}")

    # --------------------------------------------------------------------------
    # Docker / Infrastructure Detection
    # --------------------------------------------------------------------------

    def _detect_docker(
        self,
        root: Path,
        files: Dict[str, Path],
        infrastructure: Set[str],
        markers: List[str],
        evidence: Dict[str, List[str]],
    ) -> None:
        """Detects Dockerfile and Docker Compose configurations."""
        dockerfile_candidates = [f for f in files.keys() if f == "Dockerfile" or f.endswith("/Dockerfile") or f.startswith("Dockerfile.")]
        compose_candidates = [
            f for f in files.keys()
            if f in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml")
        ]

        if dockerfile_candidates:
            infrastructure.add("Docker")
            for df in dockerfile_candidates:
                markers.append(df)
                self._add_evidence(evidence, "Docker", f"Found Dockerfile: {df}")

        if compose_candidates:
            infrastructure.add("Docker Compose")
            for comp in compose_candidates:
                markers.append(comp)
                self._add_evidence(evidence, "Docker Compose", f"Found compose configuration: {comp}")

    # --------------------------------------------------------------------------
    # Primary Language Determination
    # --------------------------------------------------------------------------

    def _determine_primary_language(
        self,
        languages: List[str],
        evidence: Dict[str, List[str]],
    ) -> Optional[str]:
        """Infers the dominant primary language, or None if ambiguous/insufficient evidence."""
        if not languages:
            return None

        if len(languages) == 1:
            return languages[0]

        # Prioritize TypeScript if both JS and TS are detected in a single JS/TS project
        if set(languages) == {"JavaScript", "TypeScript"}:
            return "TypeScript"

        # Count evidence entries per language
        lang_scores: Dict[str, int] = {}
        for lang in languages:
            reasons = evidence.get(lang, [])
            # Weight root manifest files higher
            score = 0
            for r in reasons:
                if "root" in r.lower() or "manifest" in r.lower():
                    score += 2
                else:
                    score += 1
            lang_scores[lang] = score

        # Sort by score descending
        sorted_by_score = sorted(lang_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_by_score) >= 2:
            # If top two have identical high scores, do not invent a primary language
            if sorted_by_score[0][1] == sorted_by_score[1][1]:
                return None

        return sorted_by_score[0][0]
