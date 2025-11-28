# Search Multi-Agent Architecture

## Overview

This is a hierarchical three-agent architecture designed for information search and question answering tasks. The architecture follows a hierarchical structure with distinct layers: a control layer (Verifier Agent) that orchestrates the execution, and an execution layer (Search Agent and Answer Agent) that performs the actual work.

## Hierarchical Architecture

The architecture is organized into two distinct layers:

### Layer 1: Control Layer (Verifier Agent)
- **Role**: Acts as the control layer, evaluating information sufficiency and routing requests to the appropriate execution layer agent
- **Layer Position**: Top layer - always executes first
- **Input**: Original question + all historical search queries + all retrieved information + context
- **Output**: `<verify>yes</verify>` or `<verify>no</verify>`
- **Routing Logic**:
  - `yes`: Information is sufficient → Route to Answer Agent (Layer 2)
  - `no`: More information needed → Route to Search Agent (Layer 2)

### Layer 2: Execution Layer

The execution layer consists of two specialized agents that perform the actual work:

#### 2a. Search Agent
- **Role**: Generate search queries to gather external information
- **Layer Position**: Execution layer - triggered by control layer
- **Trigger**: Only executes when Verifier Agent (Layer 1) outputs `no`
- **Output**: `<search>query</search>` format
- **Behavior**: Only generates search queries, does not attempt to answer questions directly

#### 2b. Answer Agent
- **Role**: Generate high-quality answers when information is sufficient
- **Layer Position**: Execution layer - triggered by control layer
- **Trigger**: Only executes when Verifier Agent (Layer 1) outputs `yes`
- **Output**: `<answer>final answer</answer>` format
- **Behavior**: Synthesizes all available information into a comprehensive answer

## Execution Flow

The hierarchical execution follows a top-down approach:

```
Layer 1 (Control):
  Verifier Agent → Evaluates if historical information is sufficient
    │
    ├─ If "no" → Layer 2 (Execution): Search Agent → Generates search query → Return search query
    └─ If "yes" → Layer 2 (Execution): Answer Agent → Generates answer → Return answer
```


