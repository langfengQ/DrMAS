# Search Multi-Agent Architecture

## Overview

This is a three-agent architecture designed for information search and question answering tasks, using Verifier Agent as a router.

## Architecture

### 1. Verifier Agent (Router)
- **Role**: Act as a router to determine if current/historical information is sufficient to answer the question
- **Execution**: Always runs first
- **Input**: Original question + all historical search queries + all retrieved information + context
- **Output**: `<verify>yes</verify>` or `<verify>no</verify>`
- **Routing Logic**:
  - `yes`: Information is sufficient → Route to Answer Agent
  - `no`: More information needed → Route to Search Agent

### 2. Search Agent
- **Role**: Generate search queries to gather external information
- **Trigger**: Only executes when Verifier Agent outputs `no`
- **Output**: `<search>query</search>` format
- **Behavior**: Only generates search queries, does not attempt to answer questions directly

### 3. Answer Agent
- **Role**: Generate high-quality answers when information is sufficient
- **Trigger**: Only executes when Verifier Agent outputs `yes`
- **Output**: `<answer>final answer</answer>` format
- **Behavior**: Synthesizes all available information into a comprehensive answer

## Execution Flow

```
1. Verifier Agent (Router) → Evaluates if historical information is sufficient
   ├─ If "no" → 2a. Search Agent → Generates search query → Return search query
   └─ If "yes" → 2b. Answer Agent → Generates answer → Return answer
```

## Configuration

To use this architecture, configure your agent_ids as:

```yaml
agent_ids:
  - "Search Agent"
  - "Verifier Agent"
  - "Answer Agent"
```

All three agents are required for the orchestra to function properly.

## Output Behavior

- **When Verifier outputs "no"**: The `text_actions` will contain the Search Agent's search query
- **When Verifier outputs "yes"**: The `text_actions` will contain the Answer Agent's final answer

## Example

### Scenario 1: Insufficient Information (First Call)
```
Question: "What is the capital of France?"
History: (empty)

Verifier: <verify>no</verify> (needs more info)
→ Routes to Search Agent
Search Agent: <search>capital of France</search>
Final Output: <search>capital of France</search>
```

### Scenario 2: Sufficient Information (After Search)
```
Question: "What is the capital of France?"
History: 
  - Search: "capital of France"
  - Information: "Paris is the capital and largest city of France..."

Verifier: <verify>yes</verify> (information sufficient)
→ Routes to Answer Agent
Answer Agent: <answer>Paris</answer>
Final Output: <answer>Paris</answer>
```

## Key Features

1. **Router-Based Architecture**: Verifier Agent acts as an intelligent router, making decisions based on historical information
2. **Separation of Concerns**: Each agent has a single, well-defined responsibility
3. **Conditional Execution**: Only one of Search/Answer Agent runs per call (never both)
4. **Efficient Resource Usage**: Avoids unnecessary LLM calls by routing correctly
5. **Quality Control**: Verifier ensures sufficient information before attempting to answer
6. **History-Aware**: Verifier considers all historical searches and retrieved information

