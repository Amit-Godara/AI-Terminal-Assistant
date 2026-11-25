# Live Demonstration Guide: AI Terminal Assistant

A reliable, step-by-step demonstration sequence for project defense and presentation.

---

## 1. Setup & Launch Instructions

Open **Windows PowerShell** and run:

```powershell
# Navigate to project repository
cd "D:\AI Terminal Assistance"

# Launch the interactive assistant
python app.py
```

You will see the **AI Terminal Assistant** header and the green **Ready for Instructions** panel displaying the configured sandbox workspace:
```text
D:\AI Terminal Assistance\workspace
```

---

## 2. Step-by-Step Live Demonstration Script

---

### Demo 1 — Read-Only Operation (LOW Risk)

**What it demonstrates:**
- Natural language query interpretation.
- Independent safety validation.
- Automatic execution for LOW-risk operations (no confirmation prompt required).
- Distinction between informational PowerShell preview and actual Python standard library handler.

**Command to Enter:**
```text
$ask "show current directory"
```
*(Alternatively, you can omit `$ask` and type: `show current directory`)*

**Expected Terminal Output:**
- **Predicted Intent**: `CURRENT_DIRECTORY` (Confidence: ~86%)
- **Extracted Entities**: *(none)*
- **Preview Card**: `Get-Location` (marked *Informational preview. Never executed directly as arbitrary shell script*)
- **Independent Safety Validation**: `PASSED (VALID)`
- **Execution Outcome**: `SUCCESS` (Duration: < 1 ms)
- **Standard Output**: Displays `D:\AI Terminal Assistance\workspace`

**Speaker Note:**
> "Notice that the assistant identifies the intent as CURRENT_DIRECTORY, displays the informational PowerShell cmdlet `Get-Location`, validates that the path is within our workspace, and executes immediately because LOW-risk read-only commands do not require user interruption."

---

### Demo 2 — File Listing (LOW Risk)

**What it demonstrates:**
- Directory listing within the sandbox.
- Path resolution against the workspace boundary.

**Command to Enter:**
```text
$ask "list all files"
```

**Expected Terminal Output:**
- **Predicted Intent**: `LIST_FILES`
- **Preview Card**: `Get-ChildItem`
- **Validation**: `PASSED (VALID)`
- **Execution Outcome**: `SUCCESS`
- **Standard Output**: Displays the items currently in the workspace.

---

### Demo 3 — State-Modifying Operation & User Approval (MEDIUM Risk)

**What it demonstrates:**
- Parameter extraction (`FILE_NAME`: `presentation_notes.txt`).
- `MEDIUM` risk classification for state-modifying actions.
- Mandatory interactive `[y/N]` confirmation gating (defaults to NO).
- Atomic exclusive file creation (`open(..., 'x')`).

**Command to Enter:**
```text
$ask "create a file named presentation_notes.txt"
```

**Expected Terminal Output:**
- **Predicted Intent**: `CREATE_FILE`
- **Extracted Entities**: `file_name: presentation_notes.txt`
- **Preview Card**: `New-Item -ItemType File -Path "presentation_notes.txt"`
- **Validation**: `PASSED (VALID)`
- **Authorization Prompt**:
  ```text
  Execute this operation? [y/N]
  Confirmation [y/N]:
  ```

**Action:**
Type `y` and press **Enter**.

**Expected Result:**
- **Execution Outcome**: `SUCCESS`
- **Standard Output**: `Created file: D:\AI Terminal Assistance\workspace\presentation_notes.txt`

**Speaker Note:**
> "Notice the yellow panel. Because creating a file modifies the filesystem, the assistant pauses and requests authorization. Notice also that the prompt defaults to NO; only an explicit 'y' authorizes execution."

---

### Demo 4 — Semantic Understanding & Paraphrase Handling

**What it demonstrates:**
- Sentence Transformer dense vector embedding (`all-MiniLM-L6-v2`).
- Rescuing conversational paraphrases where a keyword classifier might fail.

**Command to Enter:**
```text
$ask "what directory am I in right now"
```

**Expected Terminal Output:**
- **Predicted Intent**: `CURRENT_DIRECTORY` (Confidence: ~85%)
- **Inference Mode**: `SEMANTIC_ENGINE`
- **Explanation**: Displays current working directory path.
- **Outcome**: `SUCCESS`

**Speaker Note:**
> "Here, we did not use standard keywords like 'current directory' or 'pwd'. The supervised classifier was uncertain, but our Sentence Transformer semantic engine computed high cosine similarity against reference embeddings and correctly mapped the conversational phrasing."

---

### Demo 5 — Incomplete Request & Clarification Gating

**What it demonstrates:**
- Missing parameter detection.
- Prevention of partial or hallucinated execution.

**Command to Enter:**
```text
$ask "move file"
```

**Expected Terminal Output:**
- **Predicted Intent**: `MOVE_FILE`
- **Clarification Needed Panel**:
  ```text
  Missing Required Arguments: source, destination
  Please specify both the source file and destination path.
  ```
- **Validation & Execution**: Neither validator nor executor is invoked.

**Speaker Note:**
> "Notice that when a required parameter is missing, the assistant does not guess or invent a filename. It halts immediately and prompts the user for clarification, preventing unintended command execution."

---

### Demo 6 — Security Defense: Traversal Attack Rejection

**What it demonstrates:**
- Independent security validation.
- Workspace sandboxing and path traversal blocking (`..\..\`).

**Command to Enter:**
```text
$ask "list files in ../../Windows"
```

**Expected Terminal Output:**
- **Preview Card**: `Get-ChildItem -LiteralPath "..\..\Windows"`
- **Independent Safety Validation**:
  ```text
  BLOCKED (UNSAFE_PATH)
  • Path '../../Windows' escapes the permitted workspace boundary (D:\AI Terminal Assistance\workspace).
  ```
- **Execution**: Completely blocked. The executor is never called.

**Speaker Note:**
> "Here, a query attempts to escape the sandbox using directory traversal to target the Windows directory. Even though the generator formulated a preview, the independent validator intercepted the request and blocked it because it escapes the workspace boundary."

---

### Demo 7 — Destructive Operation Protection: File Deletion

**What it demonstrates:**
- `HIGH` risk classification for destructive actions.
- Two-stage confirmation requirement: `[y/N]` approval AND explicit confirmation token `"DELETE"`.
- Token mismatch cancellation vs. valid token deletion.

#### Step 7A: Token Mismatch (Safe Cancellation)
**Command to Enter:**
```text
$ask "delete file presentation_notes.txt"
```

**Prompts & Actions:**
1. The assistant displays the destructive operation warning.
2. At `Confirmation [y/N]:`, type `y` and press **Enter**.
3. At the double-confirmation prompt:
   ```text
   Type DELETE to permanently delete this file:
   Token:
   ```
   Type `cancel` (or `yes` or press Enter without typing DELETE).

**Expected Result:**
- **Outcome**: `CANCELLED BY USER`
- **Error / Notice**: `Deletion cancelled: exact confirmation token 'DELETE' was not provided.`
- The file `presentation_notes.txt` remains intact and safe in the workspace.

#### Step 7B: Valid Deletion (Permanently Removed)
**Command to Enter:**
```text
$ask "delete file presentation_notes.txt"
```

**Prompts & Actions:**
1. At `Confirmation [y/N]:`, type `y` and press **Enter**.
2. At `Token:`, type `DELETE` (must be uppercase) and press **Enter**.

**Expected Result:**
- **Execution Outcome**: `SUCCESS`
- **Standard Output**: `Removed file: D:\AI Terminal Assistance\workspace\presentation_notes.txt`
- The temporary file is permanently deleted via `pathlib.Path.unlink()`.

**Speaker Note:**
> "For destructive actions like deletion, accidental approval is prevented by requiring the user to explicitly type the word 'DELETE'. In step A, typing anything else cancelled the operation and preserved the file. In step B, typing the exact token safely deleted only that single file."

---

### Demo 8 — Command Audit History

**What it demonstrates:**
- Persistent JSON audit logging (`data/command_history.json`).
- Resilient history table rendering with timestamps, risk levels, and durations.

**Command to Enter:**
```text
$history
```

**Expected Terminal Output:**
A formatted Rich table listing all operations executed during the session:
```text
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃  # ┃ Timestamp           ┃ Intent            ┃ Operation    ┃ Risk ┃ Status   ┃ Duration ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│  1 │ 2026-11-26 00:08:50 │ CURRENT_DIRECTORY │ GET_LOCATION │ LOW  │ SUCCESS  │    0.0ms │
│  2 │ 2026-11-26 00:08:51 │ LIST_FILES        │ LIST         │ LOW  │ SUCCESS  │    0.1ms │
│  3 │ 2026-11-26 00:08:52 │ CREATE_FILE       │ CREATE_FILE  │ MED  │ SUCCESS  │    0.2ms │
│  4 │ 2026-11-26 00:08:54 │ DELETE_FILE       │ DELETE       │ HIGH │ CANCEL   │    0.0ms │
│  5 │ 2026-11-26 00:08:55 │ DELETE_FILE       │ DELETE       │ HIGH │ SUCCESS  │    0.1ms │
└────┴─────────────────────┴───────────────────┴──────────────┴──────┴──────────┴──────────┘
```

---

### Demo 9 — Clean Exit

**Command to Enter:**
```text
$exit
```

**Expected Terminal Output:**
```text
Exiting AI Terminal Assistant. Goodbye!
```
The application terminates cleanly with exit code `0` and zero traceback.

---

## 3. Single-Query Mode (For Scripting / Non-Interactive Use)

You can also demonstrate running single commands directly from PowerShell without entering the REPL:

```powershell
python app.py --query "check python version"
python app.py --query "show current directory"
```
