# TODO


Sample repo 1: 
https://github.com/juice-shop/juice-shop.git
*set npm config to make package lock before running npm install

agent-remediation-engine/
├── config/
│   ├── prompts/                # System templates (Jinja2) for triage & remedy brains
│   ├── schemas/                # JSON/YAML tool definitions for LLM function calling
│   └── rules.yaml              # Global constraints (e.g., max tokens, forbidden files)
├── data/
│   ├── clones/                 # Git-ignored ephemeral workspace for target repos
│   ├── trajectories/           # Markdown traces of states, LLM calls, tools, and outcomes
│   └── vector_store/           # Persistent embeddings for codebase RAG
├── evals/
│   ├── benchmarks/             # Curated datasets of known bugs/vulnerabilities
│   └── golden_fixes/           # Ground-truth solutions to measure agent accuracy
├── src/
│   ├── agents/                 # Orchestration logic (LangGraph, CrewAI, or State Machines)
│   │   ├── triage_agent.py     # Logic for error parsing and root-cause localization
│   │   └── remedy_agent.py     # Logic for code generation and iterative fixing
│   ├── tools/                  # Function-calling modules for the LLM
│   │   ├── git_tools.py        # Clone, branch, commit, and PR creation
│   │   ├── search_tools.py     # Semantic search, Grep, and AST-based navigation
│   │   ├── code_map.py         # Generates structural repo maps (classes/functions)
│   │   ├── edit_tools.py       # Atomic file writing and line-specific editing
│   │   └── test_tools.py       # Proxy for executing tests via the runtime
│   ├── runtime/                # The isolation and execution layer
│   │   ├── sandbox_mgr.py      # Manages Docker/MicroVM lifecycle
│   │   └── executor.py         # Securely relays shell commands to the container
│   ├── utils/                  # Token counting, logging, and cost tracking
│   └── main.py                 # CLI/API Entry point
├── tests/                      # Unit tests for the engine's internal logic
├── .env.example                # Template for API keys and configurations
├── .gitignore
├── Dockerfile                  # Image for the Remediation Engine (The Agent)
├── sandbox.Dockerfile          # Image for the target environment (The Workspace)
└── README.md
