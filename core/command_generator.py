"""PowerShell Command Generator module for AI Terminal Assistant.

Translates intent classification and entity extraction results into controlled,
validated PowerShell command templates, explains operations, assigns risk levels,
detects missing parameters, and generates safe preview representations without execution.
"""

from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

import config
from core.entity_extractor import EntityExtractor, EntityExtractionResult
from core.intent_classifier import IntentClassifier
from core.semantic_engine import SemanticEngine, HybridPrediction

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Supported PowerShell operation types."""
    LIST = "LIST"
    CREATE_DIR = "CREATE_DIR"
    DELETE = "DELETE"
    SEARCH = "SEARCH"
    CHECK_VERSION = "CHECK_VERSION"
    GET_LOCATION = "GET_LOCATION"
    MOVE = "MOVE"
    COPY = "COPY"
    SYSTEM_INFO = "SYSTEM_INFO"
    NETWORK_INFO = "NETWORK_INFO"
    CREATE_FILE = "CREATE_FILE"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    """Risk classification for terminal command execution."""
    LOW = "LOW"        # Read-only operations (List, Search, System Info, Check Version)
    MEDIUM = "MEDIUM"  # Non-destructive modifications (Create Dir, Create File, Copy, Move)
    HIGH = "HIGH"      # Destructive modifications (Delete)


class GenerationStatus(str, Enum):
    """Lifecycle status of command generation."""
    READY_FOR_VALIDATION = "READY_FOR_VALIDATION"  # Complete arguments, ready for Phase 7
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"    # Missing required argument(s)
    UNSUPPORTED_INTENT = "UNSUPPORTED_INTENT"      # Unrecognized or out-of-scope intent
    AMBIGUOUS_REQUEST = "AMBIGUOUS_REQUEST"        # Competing entities or unclear targets
    LOW_CONFIDENCE = "LOW_CONFIDENCE"              # NLP model confidence below threshold


class CommandOperation(BaseModel):
    """Structured representation of a generated PowerShell command."""
    operation_type: OperationType
    intent: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    powershell_preview: Optional[str] = None
    explanation: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    missing_parameters: List[str] = Field(default_factory=list)
    clarification_prompt: Optional[str] = None
    generation_status: GenerationStatus
    source_query: str = ""
    inference_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Backward compatibility properties for Phase 1 GeneratedCommand consumers
    @property
    def command(self) -> Optional[str]:
        return self.powershell_preview

    @property
    def is_valid(self) -> bool:
        return self.generation_status == GenerationStatus.READY_FOR_VALIDATION

    @property
    def clarification_needed(self) -> Optional[str]:
        return self.clarification_prompt

    @property
    def entities(self) -> Dict[str, Any]:
        return self.arguments


# Alias for backward compatibility
GeneratedCommand = CommandOperation


class CommandGenerator:
    """Intelligent PowerShell Command Generator.

    Orchestrates Hybrid NLP intent inference, entity extraction, parameter
    verification, and controlled command template construction.
    """

    # Strict allowlist of software runtimes for CHECK_VERSION
    ALLOWLISTED_VERSION_COMMANDS: Dict[str, str] = {
        "python": "python --version",
        "python3": "python --version",
        "node": "node --version",
        "node.js": "node --version",
        "nodejs": "node --version",
        "git": "git --version",
        "docker": "docker --version",
        "powershell": "$PSVersionTable.PSVersion",
        "pwsh": "$PSVersionTable.PSVersion",
        "npm": "npm --version",
        "java": "java --version",
        "rust": "rustc --version",
        "rustc": "rustc --version",
        "go": "go version",
        "gcc": "gcc --version",
        "dotnet": "dotnet --version",
        "pip": "pip --version",
        "pip3": "pip --version",
        "code": "code --version",
        "vscode": "code --version",
    }

    # Intent to OperationType mapping
    INTENT_OPERATION_MAP: Dict[str, OperationType] = {
        "LIST_FILES": OperationType.LIST,
        "CREATE_DIRECTORY": OperationType.CREATE_DIR,
        "DELETE_FILE": OperationType.DELETE,
        "SEARCH_FILES": OperationType.SEARCH,
        "CHECK_VERSION": OperationType.CHECK_VERSION,
        "CURRENT_DIRECTORY": OperationType.GET_LOCATION,
        "MOVE_FILE": OperationType.MOVE,
        "COPY_FILE": OperationType.COPY,
        "SYSTEM_INFO": OperationType.SYSTEM_INFO,
        "NETWORK_INFO": OperationType.NETWORK_INFO,
        "CREATE_FILE": OperationType.CREATE_FILE,
        "UNKNOWN": OperationType.UNKNOWN,
    }

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        semantic_engine: Optional[SemanticEngine] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ) -> None:
        """Initializes the CommandGenerator.

        Accepts optional pre-initialized NLP components to avoid redundant loading.
        """
        self._classifier = classifier
        self._semantic_engine = semantic_engine
        self._entity_extractor = entity_extractor

    @property
    def classifier(self) -> IntentClassifier:
        """Lazy access to IntentClassifier."""
        if self._classifier is None:
            self._classifier = IntentClassifier()
        return self._classifier

    @property
    def semantic_engine(self) -> SemanticEngine:
        """Lazy access to SemanticEngine."""
        if self._semantic_engine is None:
            self._semantic_engine = SemanticEngine()
        return self._semantic_engine

    @property
    def entity_extractor(self) -> EntityExtractor:
        """Lazy access to EntityExtractor."""
        if self._entity_extractor is None:
            self._entity_extractor = EntityExtractor()
        return self._entity_extractor

    @staticmethod
    def _quote_path(path_str: str) -> str:
        """Safely quotes a file or directory path for PowerShell command previews.

        Escapes internal double quotes with a backtick (`") to prevent injection.
        """
        cleaned = (path_str or "").strip()
        # Remove any surrounding double quotes the user or extractor may have preserved
        if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
            cleaned = cleaned[1:-1]
        escaped = cleaned.replace('`', '``').replace('"', '`"')
        return f'"{escaped}"'

    def generate(
        self,
        query: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
    ) -> CommandOperation:
        """Generates a structured PowerShell command from natural language query or explicit inputs.

        Args:
            query: User's raw natural language command request.
            intent: Optional override for predicted intent.
            entities: Optional override for extracted entities.

        Returns:
            CommandOperation object containing structured operation, preview, and explanation.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return CommandOperation(
                operation_type=OperationType.UNKNOWN,
                intent="UNKNOWN",
                arguments={},
                powershell_preview=None,
                explanation="No command request provided.",
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                missing_parameters=[],
                clarification_prompt="Please enter a terminal instruction.",
                generation_status=GenerationStatus.NEEDS_CLARIFICATION,
                source_query=clean_query,
                inference_metadata={"confidence": 0.0, "source": "INPUT_EMPTY"},
            )

        # 1. Intent Inference (if not explicitly supplied)
        inference_metadata: Dict[str, Any] = {}
        if intent is None:
            hybrid_pred: HybridPrediction = self.semantic_engine.predict_hybrid(
                clean_query,
                classifier=self.classifier,
            )
            intent = hybrid_pred.final_intent
            inference_metadata = {
                "source": hybrid_pred.source,
                "confidence": hybrid_pred.confidence,
                "classifier_intent": hybrid_pred.classifier_intent,
                "classifier_confidence": hybrid_pred.classifier_confidence,
                "semantic_intent": hybrid_pred.semantic_intent,
                "semantic_score": hybrid_pred.semantic_score,
                "status": hybrid_pred.status,
            }

            # Reject low-confidence inference or explicit UNKNOWN
            if hybrid_pred.final_intent == "UNKNOWN" or hybrid_pred.status == "LOW_CONFIDENCE":
                if hybrid_pred.confidence < 0.35:
                    return CommandOperation(
                        operation_type=OperationType.UNKNOWN,
                        intent="UNKNOWN",
                        arguments={},
                        powershell_preview=None,
                        explanation="This request is unsupported or could not be recognized as a valid terminal operation.",
                        risk_level=RiskLevel.LOW,
                        requires_confirmation=False,
                        missing_parameters=[],
                        clarification_prompt="I could not identify a supported terminal operation. Please try rephrasing your request.",
                        generation_status=GenerationStatus.UNSUPPORTED_INTENT,
                        source_query=clean_query,
                        inference_metadata=inference_metadata,
                    )

        # 2. Entity Extraction (if not explicitly supplied)
        if entities is None:
            extraction_result: EntityExtractionResult = self.entity_extractor.extract(
                clean_query,
                intent=intent,
            )
            extracted_args = self._build_arguments_from_extraction(extraction_result)
        else:
            extracted_args = dict(entities)

        # 3. Controlled Template Mapping & Validation
        op_type = self.INTENT_OPERATION_MAP.get(intent, OperationType.UNKNOWN)

        if op_type == OperationType.LIST:
            return self._generate_list_files(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.CREATE_DIR:
            return self._generate_create_directory(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.DELETE:
            return self._generate_delete_file(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.SEARCH:
            return self._generate_search_files(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.CHECK_VERSION:
            return self._generate_check_version(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.GET_LOCATION:
            return self._generate_current_directory(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.MOVE:
            return self._generate_move_file(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.COPY:
            return self._generate_copy_file(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.SYSTEM_INFO:
            return self._generate_system_info(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.NETWORK_INFO:
            return self._generate_network_info(clean_query, intent, extracted_args, inference_metadata)
        elif op_type == OperationType.CREATE_FILE:
            return self._generate_create_file(clean_query, intent, extracted_args, inference_metadata)
        else:
            return CommandOperation(
                operation_type=OperationType.UNKNOWN,
                intent="UNKNOWN",
                arguments=extracted_args,
                powershell_preview=None,
                explanation="No recognized terminal operation mapped to this request.",
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                missing_parameters=[],
                clarification_prompt="Unsupported terminal request.",
                generation_status=GenerationStatus.UNSUPPORTED_INTENT,
                source_query=clean_query,
                inference_metadata=inference_metadata,
            )

    def _build_arguments_from_extraction(
        self,
        extraction: EntityExtractionResult,
    ) -> Dict[str, Any]:
        """Flattens extracted entities into a canonical argument dictionary."""
        args: Dict[str, Any] = {}
        by_type = extraction.entities_by_type

        # Primary paths and names
        if by_type.get("SOURCE_PATH"):
            args["source"] = by_type["SOURCE_PATH"][0]
        elif by_type.get("FILE_NAME"):
            args["source"] = by_type["FILE_NAME"][0]

        if by_type.get("DESTINATION_PATH"):
            args["destination"] = by_type["DESTINATION_PATH"][0]

        if by_type.get("FILE_NAME"):
            args["file_name"] = by_type["FILE_NAME"][0]

        if by_type.get("DIRECTORY_NAME"):
            args["directory_name"] = by_type["DIRECTORY_NAME"][0]

        if by_type.get("VERSION_NAME"):
            args["version_name"] = by_type["VERSION_NAME"][0]

        if by_type.get("KEYWORD"):
            args["keyword"] = by_type["KEYWORD"][0]

        if by_type.get("FILE_EXTENSION"):
            args["file_extension"] = by_type["FILE_EXTENSION"][0]

        if by_type.get("NUMBER"):
            try:
                args["limit"] = int(by_type["NUMBER"][0])
            except ValueError:
                pass

        # Preserve any detected missing parameters from Phase 4
        if extraction.missing_parameters:
            args["_missing_parameters"] = extraction.missing_parameters

        return args

    # --------------------------------------------------------------------------
    # Individual Intent Generators
    # --------------------------------------------------------------------------

    def _generate_list_files(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates Get-ChildItem for LIST_FILES."""
        # Check target directory
        target_dir = args.get("directory_name") or args.get("source") or args.get("destination")
        include_hidden = any(w in query.lower() for w in ("hidden", "-force"))

        flags = []
        if include_hidden:
            flags.append("-Force")

        if target_dir:
            quoted_dir = self._quote_path(target_dir)
            cmd = f"Get-ChildItem -LiteralPath {quoted_dir}"
            if flags:
                cmd += " " + " ".join(flags)
            explanation = f"Lists all files and folders in directory {quoted_dir}."
            if include_hidden:
                explanation += " Includes hidden items."
        else:
            cmd = "Get-ChildItem"
            if flags:
                cmd += " " + " ".join(flags)
            explanation = "Lists all files and folders in the current directory."
            if include_hidden:
                explanation += " Includes hidden items."

        return CommandOperation(
            operation_type=OperationType.LIST,
            intent=intent,
            arguments=args,
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_create_directory(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates New-Item -ItemType Directory for CREATE_DIRECTORY."""
        dir_name = args.get("directory_name") or args.get("destination") or args.get("source")
        if not dir_name:
            return CommandOperation(
                operation_type=OperationType.CREATE_DIR,
                intent=intent,
                arguments=args,
                powershell_preview=None,
                explanation="Creates a new folder or directory.",
                risk_level=RiskLevel.MEDIUM,
                requires_confirmation=False,
                missing_parameters=["directory_name"],
                clarification_prompt="Please specify the name or path of the directory you want to create.",
                generation_status=GenerationStatus.NEEDS_CLARIFICATION,
                source_query=query,
                inference_metadata=metadata,
            )

        quoted_path = self._quote_path(dir_name)
        cmd = f"New-Item -ItemType Directory -Path {quoted_path}"
        explanation = f"Creates a new directory at {quoted_path}."

        return CommandOperation(
            operation_type=OperationType.CREATE_DIR,
            intent=intent,
            arguments={"directory_name": dir_name},
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=False,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_delete_file(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates Remove-Item -LiteralPath for DELETE_FILE.

        High risk: Requires explicit confirmation; never includes -Recurse or -Force.
        """
        target = args.get("file_name") or args.get("source") or args.get("directory_name")
        if not target:
            return CommandOperation(
                operation_type=OperationType.DELETE,
                intent=intent,
                arguments=args,
                powershell_preview=None,
                explanation="Permanently removes a file.",
                risk_level=RiskLevel.HIGH,
                requires_confirmation=True,
                missing_parameters=["file_name"],
                clarification_prompt="Please specify the file name or path you want to delete.",
                generation_status=GenerationStatus.NEEDS_CLARIFICATION,
                source_query=query,
                inference_metadata=metadata,
            )

        quoted_path = self._quote_path(target)
        cmd = f"Remove-Item -LiteralPath {quoted_path}"
        explanation = (
            f"Permanently deletes file {quoted_path}. "
            "Caution: This is a destructive operation that will permanently remove the file."
        )

        return CommandOperation(
            operation_type=OperationType.DELETE,
            intent=intent,
            arguments={"file_name": target},
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_search_files(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates Get-ChildItem search for SEARCH_FILES."""
        search_dir = args.get("directory_name") or "."
        keyword = args.get("keyword")
        ext = args.get("file_extension")

        # Determine search filter pattern
        if ext and not keyword:
            pattern = f"*{ext}" if not ext.startswith("*") else ext
        elif keyword and not ext:
            pattern = f"*{keyword}*" if not ("*" in keyword or "?" in keyword) else keyword
        elif keyword and ext:
            clean_ext = ext if ext.startswith(".") else f".{ext}"
            pattern = f"*{keyword}*{clean_ext}"
        else:
            pattern = "*.*"

        # Check recursion
        recurse = any(w in query.lower() for w in ("recursively", "recurse", "subfolders", "all folders", "deep"))

        quoted_dir = self._quote_path(search_dir)
        quoted_pattern = self._quote_path(pattern)

        cmd = f"Get-ChildItem -Path {quoted_dir} -Filter {quoted_pattern} -File"
        if recurse:
            cmd += " -Recurse"

        dir_desc = "current folder" if search_dir == "." else f"directory {quoted_dir}"
        rec_desc = " (recursively)" if recurse else ""
        explanation = f"Searches for files matching pattern {quoted_pattern} in {dir_desc}{rec_desc}."

        return CommandOperation(
            operation_type=OperationType.SEARCH,
            intent=intent,
            arguments={
                "directory": search_dir,
                "pattern": pattern,
                "recurse": recurse,
                "keyword": keyword,
                "extension": ext,
            },
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_check_version(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates version query command for allowlisted runtimes/tools."""
        tool = args.get("version_name")

        if not tool:
            # Fallback search inside query text if extractor missed it
            for candidate in self.ALLOWLISTED_VERSION_COMMANDS:
                if candidate in query.lower():
                    tool = candidate
                    break

        if not tool:
            return CommandOperation(
                operation_type=OperationType.CHECK_VERSION,
                intent=intent,
                arguments=args,
                powershell_preview=None,
                explanation="Checks installed software runtime version.",
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                missing_parameters=["version_name"],
                clarification_prompt="Which software version would you like to check (e.g., Python, Node.js, Git, Docker)?",
                generation_status=GenerationStatus.NEEDS_CLARIFICATION,
                source_query=query,
                inference_metadata=metadata,
            )

        normalized_tool = tool.lower().strip()
        cmd = self.ALLOWLISTED_VERSION_COMMANDS.get(normalized_tool)

        if not cmd:
            return CommandOperation(
                operation_type=OperationType.CHECK_VERSION,
                intent=intent,
                arguments={"version_name": tool},
                powershell_preview=None,
                explanation=f"Version check for '{tool}' is not in the approved software allowlist.",
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                missing_parameters=[],
                clarification_prompt=(
                    f"'{tool}' is not in the allowlist of supported tools for version checks. "
                    "Supported tools: Python, Node.js, Git, Docker, PowerShell, npm, Java, Rust, Go, GCC, dotnet."
                ),
                generation_status=GenerationStatus.UNSUPPORTED_INTENT,
                source_query=query,
                inference_metadata=metadata,
            )

        explanation = f"Checks the installed version of {tool}."

        return CommandOperation(
            operation_type=OperationType.CHECK_VERSION,
            intent=intent,
            arguments={"version_name": tool, "command": cmd},
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_current_directory(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates Get-Location for CURRENT_DIRECTORY."""
        return CommandOperation(
            operation_type=OperationType.GET_LOCATION,
            intent=intent,
            arguments={},
            powershell_preview="Get-Location",
            explanation="Displays the current working directory path.",
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_move_file(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates Move-Item -LiteralPath for MOVE_FILE."""
        source = args.get("source") or args.get("file_name")
        dest = args.get("destination") or args.get("destination_path")

        missing = []
        if not source:
            missing.append("source")
        if not dest:
            missing.append("destination")

        if missing:
            missing_desc = " and ".join(missing)
            return CommandOperation(
                operation_type=OperationType.MOVE,
                intent=intent,
                arguments=args,
                powershell_preview=None,
                explanation="Moves a file or folder from source to destination.",
                risk_level=RiskLevel.MEDIUM,
                requires_confirmation=True,
                missing_parameters=missing,
                clarification_prompt=f"Please specify the {missing_desc} for the move operation.",
                generation_status=GenerationStatus.NEEDS_CLARIFICATION,
                source_query=query,
                inference_metadata=metadata,
            )

        quoted_src = self._quote_path(source)
        quoted_dest = self._quote_path(dest)
        cmd = f"Move-Item -LiteralPath {quoted_src} -Destination {quoted_dest}"
        explanation = f"Moves {quoted_src} to {quoted_dest}."

        return CommandOperation(
            operation_type=OperationType.MOVE,
            intent=intent,
            arguments={"source": source, "destination": dest},
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=True,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_copy_file(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates Copy-Item -LiteralPath for COPY_FILE."""
        source = args.get("source") or args.get("file_name")
        dest = args.get("destination") or args.get("destination_path")

        missing = []
        if not source:
            missing.append("source")
        if not dest:
            missing.append("destination")

        if missing:
            missing_desc = " and ".join(missing)
            return CommandOperation(
                operation_type=OperationType.COPY,
                intent=intent,
                arguments=args,
                powershell_preview=None,
                explanation="Copies a file or directory from source to destination.",
                risk_level=RiskLevel.MEDIUM,
                requires_confirmation=True,
                missing_parameters=missing,
                clarification_prompt=f"Please specify the {missing_desc} for the copy operation.",
                generation_status=GenerationStatus.NEEDS_CLARIFICATION,
                source_query=query,
                inference_metadata=metadata,
            )

        quoted_src = self._quote_path(source)
        quoted_dest = self._quote_path(dest)
        cmd = f"Copy-Item -LiteralPath {quoted_src} -Destination {quoted_dest}"
        explanation = f"Copies {quoted_src} to {quoted_dest}."

        return CommandOperation(
            operation_type=OperationType.COPY,
            intent=intent,
            arguments={"source": source, "destination": dest},
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=True,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_system_info(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates safe read-only system information commands for SYSTEM_INFO."""
        q_lower = query.lower()

        if any(w in q_lower for w in ("hostname", "computer name", "machine name")):
            cmd = "hostname"
            explanation = "Displays the computer hostname."
        elif re.search(r"\b(?:os|operating\s+system)\b", q_lower):
            cmd = "Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture"
            explanation = "Displays operating system version, architecture, and edition."
        else:
            cmd = "Get-ComputerInfo"
            explanation = "Displays detailed hardware, memory, operating system, and system configuration."

        return CommandOperation(
            operation_type=OperationType.SYSTEM_INFO,
            intent=intent,
            arguments=args,
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_network_info(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates safe read-only network queries for NETWORK_INFO."""
        q_lower = query.lower()

        if any(w in q_lower for w in ("adapter", "interface", "ethernet", "wifi", "hardware")):
            cmd = "Get-NetAdapter"
            explanation = "Displays network interface cards and adapter hardware status."
        elif any(w in q_lower for w in ("connection", "port", "socket", "listening", "tcp")):
            cmd = "Get-NetTCPConnection"
            explanation = "Displays current network connections, endpoints, and listening TCP ports."
        else:
            cmd = "ipconfig"
            explanation = "Displays IP configuration, subnet mask, and default gateways."

        return CommandOperation(
            operation_type=OperationType.NETWORK_INFO,
            intent=intent,
            arguments=args,
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )

    def _generate_create_file(
        self,
        query: str,
        intent: str,
        args: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CommandOperation:
        """Generates New-Item -ItemType File for CREATE_FILE."""
        file_name = args.get("file_name") or args.get("source")
        if not file_name:
            return CommandOperation(
                operation_type=OperationType.CREATE_FILE,
                intent=intent,
                arguments=args,
                powershell_preview=None,
                explanation="Creates a new empty file.",
                risk_level=RiskLevel.MEDIUM,
                requires_confirmation=False,
                missing_parameters=["file_name"],
                clarification_prompt="Please specify the name of the file you want to create.",
                generation_status=GenerationStatus.NEEDS_CLARIFICATION,
                source_query=query,
                inference_metadata=metadata,
            )

        quoted_path = self._quote_path(file_name)
        cmd = f"New-Item -ItemType File -Path {quoted_path}"
        explanation = f"Creates a new blank file at {quoted_path}."

        return CommandOperation(
            operation_type=OperationType.CREATE_FILE,
            intent=intent,
            arguments={"file_name": file_name},
            powershell_preview=cmd,
            explanation=explanation,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=False,
            missing_parameters=[],
            clarification_prompt=None,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query=query,
            inference_metadata=metadata,
        )
