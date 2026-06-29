# 🏗️ Architecture Overview: Commander-Worker Task Queue

## 🎯 The Objective
To transform the system into a dynamic, Task-Centric operating system where the **Supervisor is the sole planner (Commander)** and the Subagents are strictly execution-only (**Dumb Workers**). The Supervisor investigates failures, queries external registries, writes highly specific instructions, and dynamically spawns new tasks to handle breaking changes. 

---

## 🏛️ 1. The Core Data Model: Dynamic Tasks
We move away from looping over static `VulnerabilityGroup` records. The engine now processes active `RemediationTask` objects in a dynamic queue.
*   **`task_id`:** Unique identifier (e.g., `task-1`).
*   **`parent_group_id`:** Maps the task back to the vulnerability for QA scanner math.
*   **`strategy`:** `VERSION_BUMP` or `CODE_WORKAROUND`.
*   **`instruction`:** The exact, explicit command written by the Supervisor (e.g., *"Update express-jwt to 6.0.0 using an override in frontend/package.json"*).
*   **`status`:** `PENDING`, `OPTIMISTICALLY_FIXED`, `QA_PASSED`, `NEEDS_RETRY`, `UNFIXABLE`.
*   **`ancestry_depth`:** Tracks how many times a task has spawned a child (capped at `3` to prevent infinite LLM task explosions).

---

## 🧠 2. The 5 Core Supervisor Playbooks (Edge Case Handling)
Because the Supervisor is now an active Planner with access to the NPM registry, it executes specific "Senior Engineer" playbooks when the QA Critic reports a failure:

1. **The "Stubborn CVE" (Scanner still flags after bump)**
   * *Trigger:* QA Critic returns `SECURITY_FLAG`.
   * *Playbook:* Supervisor queries the NPM registry. It finds the absolute latest version. It rewrites the task instruction to forcefully apply an `"overrides"` block at the repository root to nuke all transitive ghosts. If already on the latest version, it pivots the strategy to `CODE_WORKAROUND`.
2. **The "Dependency Conflict" (`ERESOLVE` / Peer Conflicts)**
   * *Trigger:* QA Critic returns `PEER_CONFLICT`.
   * *Playbook:* Supervisor queries the NPM registry for a **backported patch** (e.g., if bumping from 1.x to 2.x caused a conflict, it checks if a safe `1.0.1` patch exists). If no backport exists, it pivots the strategy to `CODE_WORKAROUND` to avoid the bump entirely.
3. **The "Breaking Change" (Tests fail after bump)**
   * *Trigger:* QA Critic returns `BREAKING_CHANGE`.
   * *Playbook:* Supervisor marks the version bump task as `SUCCESS` and locks the new version in the Constraints Ledger. It then **spawns a new Child Task** (`Depth: 1`) assigned to the Workaround Subagent, instructing it to refactor the broken API calls in the codebase to comply with the new package version.
4. **The "Abandoned Package" (No patch exists)**
   * *Trigger:* Triage requests a bump, or the Subagent fails to find the version.
   * *Playbook:* Supervisor queries the NPM registry and sees the package hasn't been published in 5 years. It immediately pivots the strategy to `CODE_WORKAROUND` and instructs the codebase agent to write sanitization logic around the abandoned library.
5. **The `EBADENGINE` Trap (Environment Mismatch)**
   * *Trigger:* QA Critic or Subagent inline-check returns `EBADENGINE`.
   * *Playbook:* Supervisor recognizes the patched version requires a newer Node.js runtime than the Docker Sandbox possesses. It queries the NPM registry for an older, compatible patched version. If none exists, it pivots to `CODE_WORKAROUND`.

---

## ⚙️ 3. The Nodes & Responsibilities

### The Supervisor Node (The Commander)
*   **Internal Architecture (Two-Phase):**
    *   **Phase 1 (The Planner):** A bounded ReAct loop with access to the **`view_npm_package_versions`** tool. It looks up what versions exist, cross-references the QA feedback, applies the Playbooks above, and formulates a new plan in a scratchpad.
    *   **Phase 2 (The Router):** A zero-shot LLM reads the scratchpad and outputs the strict `SupervisorDecision` Pydantic model (spawning tasks, updating instructions, and routing the graph).

### The Update Subagent (The Dumb Worker)
*   **Role:** Strictly executes manifest modifications based on the Supervisor's exact instruction. 
*   **Tools:** `modify_npm_dependency` and a fast inline lockfile check (`npm install --package-lock-only`). 
*   **No Planning Allowed:** It does not have access to the NPM registry. It cannot guess fallback versions. It blindly follows the Commander's orders.

### The Workaround Subagent (The Stubbed Worker)
*   **Role:** Reserved for `CODE_WORKAROUND` tasks or spawned `BREAKING_CHANGE` refactoring tasks. *(Note: For this initial refactor, this node will act as a routing endpoint/stub to verify the task pipeline works).*

### The QA Critic (Map-Reduce Evaluator)
*   **Role:** Runs the heavy infrastructure (`npm install`, `run_security_scan`, `run_unit_tests`). 
*   **Feedback:** Provides exact failure logs back to the Supervisor, triggering the Supervisor to enter its Planning phase to select the appropriate Playbook for the next round.

---

## 🔄 4. The Execution Lifecycle

1. **Initialization:** The graph translates the initial CVEs into `Depth: 0` tasks in the `task_queue`.
2. **Supervisor Planning:** The Supervisor reads the queue. For pending tasks, it queries the registry (if needed) and writes the initial strict instructions.
3. **Dispatch:** The Supervisor routes the tasks to the "Dumb" Update Subagent.
4. **Blind Execution:** The Update Subagent applies the exact version requested. If successful, marks `OPTIMISTICALLY_FIXED`.
5. **QA Evaluation:** The QA Critic executes the Map-Reduce flow and returns strict Pydantic verdicts.
6. **Commander Replanning (The Magic):**
    * The Supervisor wakes up and evaluates the QA verdicts.
    * It applies the 5 Core Playbooks (e.g., spawning child tasks, finding backports via the registry, or pivoting strategies).
    * It overwrites the task instructions and routes them back to the workers.
7. **Teardown:** When the Queue is empty or all tasks hit max depth/retries, a final pristine Git Diff is generated.