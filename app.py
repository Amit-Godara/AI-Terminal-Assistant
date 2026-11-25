"""Main application entry point for AI Terminal Assistant.

Orchestrates the complete NLP-to-Execution pipeline:
Natural Language Input
→ Text Preprocessing
→ Intent Classification
→ Semantic Similarity / Hybrid Inference
→ Entity Extraction
→ Command Generation
→ Independent Safety Validation
→ User Confirmation (when required)
→ Controlled Execution
→ Output Display
→ Command History
"""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from core.preprocessor import TextPreprocessor
from core.intent_classifier import IntentClassifier
from core.entity_extractor import EntityExtractor, EntityExtractionResult
from core.semantic_engine import SemanticEngine, HybridPrediction
from core.command_generator import (
    CommandGenerator,
    CommandOperation,
    OperationType,
    RiskLevel,
    GenerationStatus,
)
from core.command_validator import (
    CommandValidator,
    ValidationResult,
    ValidationStatus,
)
from core.executor import (
    CommandExecutor,
    ExecutionResult,
    ExecutionStatus,
)
from utils.terminal_ui import TerminalUI
from utils.history import HistoryManager

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AITerminalAssistant")


class AITerminalAssistant:
    """End-to-End Orchestrator for the AI Terminal Assistant."""

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        safe_mode: bool = True,
        console: Optional[Any] = None,
        history_file_path: Optional[Path] = None,
    ) -> None:
        """Initializes all components once and stores them for reuse."""
        self.workspace_dir = (workspace_dir or config.WORKSPACE_DIR).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.safe_mode = safe_mode

        # Initialize UI & History
        self.ui = TerminalUI(console=console)
        self.history_file_path = history_file_path or config.COMMAND_HISTORY_PATH
        self.history_manager = HistoryManager(self.history_file_path)

        # Initialize NLP Pipeline once
        self.preprocessor = TextPreprocessor()
        self.classifier = IntentClassifier()
        self.semantic_engine = SemanticEngine()
        self.entity_extractor = EntityExtractor()

        # Initialize Command Generator with shared NLP instances
        self.generator = CommandGenerator(
            classifier=self.classifier,
            semantic_engine=self.semantic_engine,
            entity_extractor=self.entity_extractor,
        )

        # Initialize Independent Validator with strict workspace enforcement
        self.validator = CommandValidator(
            workspace_dir=self.workspace_dir,
            safe_mode=self.safe_mode,
        )

        # Initialize Controlled Executor with interactive callbacks
        self.executor = CommandExecutor(
            timeout_seconds=config.COMMAND_TIMEOUT_SECONDS,
            history_file_path=self.history_file_path,
            confirmation_callback=self._handle_confirmation,
            deletion_token_callback=self._handle_deletion_token,
        )

    # --------------------------------------------------------------------------
    # Interactive Callback Handlers
    # --------------------------------------------------------------------------

    def _handle_confirmation(self, prompt: str, risk_level: RiskLevel) -> bool:
        """Invoked by executor when confirmation is required."""
        return self.ui.prompt_confirmation(
            operation_type="State-Modifying Operation",
            risk_level=risk_level,
            details=prompt,
        )

    def _handle_deletion_token(self, target_name: str) -> str:
        """Invoked by executor when double confirmation for deletion is required."""
        return self.ui.prompt_deletion_token(target_name)

    # --------------------------------------------------------------------------
    # Core Pipeline Processing
    # --------------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        approved: Optional[bool] = None,
        deletion_token: Optional[str] = None,
    ) -> Tuple[Optional[CommandOperation], Optional[ValidationResult], Optional[ExecutionResult]]:
        """Executes the complete natural language to execution pipeline.

        Args:
            query: User's natural language request.
            approved: Optional confirmation override (for testing or automation).
            deletion_token: Optional confirmation token override (for testing).

        Returns:
            Tuple of (CommandOperation, ValidationResult, ExecutionResult).
        """
        clean_query = (query or "").strip()
        if not clean_query:
            self.ui.show_warning("Empty command instruction received.")
            return None, None, None

        self.ui.show_query_header(clean_query)

        try:
            # 1. Text Preprocessing
            _ = self.preprocessor.tokenize(clean_query)
            _ = self.preprocessor.normalize_text(clean_query)

            # 2. Hybrid Intent Classification (Classifier + Semantic Engine)
            hybrid_pred: HybridPrediction = self.semantic_engine.predict_hybrid(
                clean_query,
                classifier=self.classifier,
            )
            intent = hybrid_pred.final_intent
            confidence = hybrid_pred.confidence
            source = hybrid_pred.source

            inference_metadata = {
                "source": hybrid_pred.source,
                "confidence": hybrid_pred.confidence,
                "classifier_intent": hybrid_pred.classifier_intent,
                "classifier_confidence": hybrid_pred.classifier_confidence,
                "semantic_intent": hybrid_pred.semantic_intent,
                "semantic_score": hybrid_pred.semantic_score,
                "status": hybrid_pred.status,
            }

            self.ui.show_intent(intent, confidence, source)
            self.ui.show_inference_metadata(inference_metadata)

            # 3. Entity Extraction from raw query
            extraction_result: EntityExtractionResult = self.entity_extractor.extract(
                clean_query,
                intent=intent,
            )
            extracted_entities = self.generator._build_arguments_from_extraction(extraction_result)
            self.ui.show_entities(extracted_entities)

            # 4. Command Generation
            operation: CommandOperation = self.generator.generate(
                query=clean_query,
                intent=intent,
                entities=extracted_entities,
            )

            # Display informational PowerShell preview and explanation
            self.ui.show_command_preview(
                operation.powershell_preview,
                operation.explanation,
                operation.risk_level,
            )

            # 5. Check Generation Status
            if operation.generation_status == GenerationStatus.NEEDS_CLARIFICATION:
                prompt_text = (
                    operation.clarification_prompt
                    or "Missing required parameters. Please provide more details."
                )
                self.ui.show_clarification_prompt(prompt_text, operation.missing_parameters)
                return operation, None, None

            if (
                operation.generation_status == GenerationStatus.UNSUPPORTED_INTENT
                or operation.operation_type == OperationType.UNKNOWN
            ):
                self.ui.show_error(
                    operation.explanation
                    or "This request is unsupported or could not be recognized as a valid terminal operation.",
                    error_type="UnsupportedIntent",
                )
                return operation, None, None

            if operation.generation_status == GenerationStatus.LOW_CONFIDENCE:
                self.ui.show_warning(
                    "NLP inference confidence is too low to safely execute this operation. "
                    "Please rephrase your request with specific file names or command verbs."
                )
                return operation, None, None

            if operation.generation_status == GenerationStatus.AMBIGUOUS_REQUEST:
                self.ui.show_warning(
                    operation.explanation
                    or "Your request is ambiguous. Please specify the target file or directory explicitly."
                )
                return operation, None, None

            # 6. Independent Safety Validation
            val_result: ValidationResult = self.validator.validate(operation)
            self.ui.show_validation_result(val_result)

            if not val_result.is_valid:
                # Validation blocked execution
                return operation, val_result, None

            # 7. Controlled Execution
            exec_result: ExecutionResult = self.executor.execute(
                operation=operation,
                validation_result=val_result,
                approved=approved,
                deletion_token=deletion_token,
            )
            self.ui.show_execution_result(exec_result)

            return operation, val_result, exec_result

        except Exception as exc:
            logger.exception("Unexpected error processing query: %s", exc)
            self.ui.show_error(f"Unexpected processing error: {exc}", error_type=type(exc).__name__)
            return None, None, None

    # --------------------------------------------------------------------------
    # Interactive REPL
    # --------------------------------------------------------------------------

    def run(self) -> None:
        """Starts the interactive terminal command loop."""
        self.ui.show_banner()
        self.ui.show_welcome(
            self.workspace_dir,
            model_info={
                "Intent Classifier": "TF-IDF + Logistic Regression (12 intents)",
                "Semantic Engine": "SentenceTransformer (all-MiniLM-L6-v2) + Sparse Fallback",
                "Entity Extractor": "spaCy EntityRuler + Deterministic Regex",
            },
        )

        while True:
            try:
                user_input = self.ui.prompt_user()
                if not user_input:
                    continue

                cmd_lower = user_input.strip().lower()

                # 1. Exit Commands
                if cmd_lower in ("$exit", "exit", "quit", "$quit", ":q"):
                    self.ui.show_exit()
                    break

                # 2. Help Commands
                if cmd_lower in ("$help", "help", "?", "$?"):
                    self.ui.show_help(self.workspace_dir)
                    continue

                # 3. Clear Screen Commands
                if cmd_lower in ("$clear", "clear", "cls", "$cls"):
                    self.ui.clear_screen()
                    continue

                # 4. History Commands
                if cmd_lower in ("$history", "history"):
                    history_records = self.history_manager.get_history()
                    self.ui.show_history(history_records)
                    continue

                # 5. Natural Language Request ($ask <query> or raw input)
                if user_input.startswith("$ask"):
                    query = user_input[4:].strip().strip('"').strip("'").strip()
                else:
                    query = user_input.strip()

                if not query:
                    self.ui.show_warning("Please enter a question or instruction after $ask.")
                    continue

                self.process_query(query)

            except (KeyboardInterrupt, EOFError):
                self.ui.show_exit()
                break
            except Exception as e:
                logger.exception("Loop exception: %s", e)
                self.ui.show_error(f"An unexpected error occurred: {e}")


# Backward compatibility alias
TerminalAssistantApp = AITerminalAssistant


def main() -> None:
    """CLI entry point for launching the assistant."""
    parser = argparse.ArgumentParser(description="AI Terminal Assistant - Windows PowerShell Assistant")
    parser.add_argument("--query", "-q", type=str, help="Single query to process and exit")
    parser.add_argument("--workspace", "-w", type=str, help="Custom workspace directory")
    args = parser.parse_args()

    custom_ws = Path(args.workspace).resolve() if args.workspace else None
    assistant = AITerminalAssistant(workspace_dir=custom_ws)

    if args.query:
        assistant.process_query(args.query)
    else:
        assistant.run()


if __name__ == "__main__":
    main()
