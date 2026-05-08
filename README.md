# llm-ledger

Zero-config LLM call interceptor. Track token usage, cost, and latency — locally, no cloud required.

```bash
pip install llm-ledger
```

```python
from llm_ledger import watchdog

@watchdog(tag="summarise")
def ask(prompt):
    return client.chat.completions.create(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )

ask("Summarise this document...")
```

```bash
$ watchdog summary
  llm-ledger summary  ·  Last 24h
────────────────────────────────────────────────────────────────────────
  Calls                  47
  Input tokens         82.3k
  Output tokens        46.1k
  Total tokens        128.4k
  Total cost          $0.4100
  Avg latency           834ms
  Success rate         97.9%
────────────────────────────────────────────────────────────────────────
  Models:
    gpt-4o                                   31 calls
    qwen2.5:0.5b                             12 calls
    gpt-4o-mini                               4 calls
```

---

## Why llm-ledger?

You're building an LLM feature. You want to know: how many tokens did each call use, how long did it take, what did it cost, did it fail silently? Right now you add print statements. Or you install LangSmith and spend a day configuring it.

`llm-ledger` gives you a decorator and a CLI. No cloud account. No config files. No framework lock-in.

|                        | llm-ledger | LangSmith | Helicone |
|------------------------|:------------:|:---------:|:--------:|
| pip install + done     | ✓            | ✗         | ✗        |
| No cloud account       | ✓            | ✗         | ✗        |
| No traffic proxying    | ✓            | ✓         | ✗        |
| Works with Ollama      | ✓            | partial   | ✗        |
| Budget enforcement     | ✓            | ✗         | ✗        |

---

## Installation

```bash
pip install llm-ledger          # core + CLI
```

**Requires:** Python 3.10+, `click>=8.0`, `openai>=1.0` (optional, for OpenAI/Ollama)

**Works with Ollama out of the box** — free, local, no API key:

```bash
ollama pull qwen2.5:0.5b
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

---

## Usage

### `@watchdog` decorator

Wrap any function that returns a raw API response object:

```python
from llm_ledger import watchdog
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

@watchdog(tag="summarise", user_id="alice")
def ask(prompt: str):
    return client.chat.completions.create(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )

response = ask("What is the capital of France?")
print(response.choices[0].message.content)
# Every call is now logged to ~/.llm-ledger/logs.db
```

Parameters:

| Parameter    | Type            | Description                                      |
|--------------|-----------------|--------------------------------------------------|
| `tag`        | `str` \| `None` | Label for grouping calls in the CLI              |
| `user_id`    | `str` \| `None` | Per-user identifier, used by Budget enforcement  |
| `session_id` | `str` \| `None` | Optional session grouping label                  |
| `db_path`    | `Path` \| `None`| Override default DB path (useful in tests)       |

---

### `watch()` context manager

Track one or more API calls in a block and read live metrics afterward:

```python
from llm_ledger import watch

with watch(tag="batch-summarise", user_id="alice") as w:
    for doc in documents:
        resp = client.chat.completions.create(
            model="qwen2.5:0.5b",
            messages=[{"role": "user", "content": doc}]
        )
        w.add_response(resp)

print(f"Used {w.tokens_used:,} tokens")
print(f"Cost: ${w.cost_usd:.6f}")
print(f"Latency: {w.latency_ms:.0f}ms")
print(f"Calls: {w.call_count}")
```

---

### `Budget()` enforcement

Stop a runaway loop before it empties your wallet:

```python
from llm_ledger import Budget, BudgetExceeded

try:
    with Budget(max_usd=0.10, user_id="alice") as b:
        for chunk in large_document_chunks:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": chunk}]
            )
            b.add_response(resp)   # raises BudgetExceeded if over $0.10
except BudgetExceeded as e:
    print(f"Stopped: {e.user_id} spent ${e.spent:.4f} (limit ${e.limit:.4f})")
```

**Period options** — control what prior spend counts against the budget:

```python
Budget(max_usd=0.10, user_id="alice", period="today")    # default: since midnight UTC
Budget(max_usd=1.00, user_id="alice", period="all")      # all historical spend
Budget(max_usd=0.05, user_id="alice", period="session")  # this block only
```

**FastAPI example:**

```python
@app.post("/generate")
async def generate(prompt: str, user_id: str):
    with Budget(max_usd=0.05, user_id=user_id) as b:
        resp = client.chat.completions.create(...)
        b.add_response(resp)
    return {"result": resp.choices[0].message.content}
```

---

### CLI

All commands read from `~/.llm-ledger/logs.db` by default. Override with `--db /path/to/logs.db`.

#### `watchdog summary`

```bash
watchdog summary                   # last 24h
watchdog summary --last 7d         # last 7 days
watchdog summary --last all        # all time
```

#### `watchdog tail`

```bash
watchdog tail                      # last 20 calls
watchdog tail -n 50                # last 50 calls
watchdog tail --tag summarise      # filter by tag
watchdog tail --user alice         # filter by user
watchdog tail --model gpt-4o       # filter by model
```

#### `watchdog top`

```bash
watchdog top                              # top tags by cost
watchdog top --by tokens                  # rank by token usage
watchdog top --by calls                   # rank by call count
watchdog top --group user                 # group by user_id
watchdog top --group model                # group by model
watchdog top --limit 5 --last 7d          # top 5 in last 7 days
```

---

## Storage

All data is stored locally in `~/.llm-ledger/logs.db` (SQLite). No data ever leaves your machine.

Override the path with the `--db` flag on any CLI command, or pass `db_path=` to any decorator/context manager.

Each log entry contains:

| Field           | Description                                  |
|-----------------|----------------------------------------------|
| `id`            | UUID                                         |
| `timestamp`     | UTC datetime                                 |
| `provider`      | `openai` / `anthropic` / `unknown`           |
| `model`         | Model name from API response                 |
| `input_tokens`  | From `response.usage.prompt_tokens`          |
| `output_tokens` | From `response.usage.completion_tokens`      |
| `cost_usd`      | Calculated from embedded pricing table       |
| `latency_ms`    | Wall-clock milliseconds                      |
| `success`       | `True` / `False`                             |
| `tag`           | Optional label                               |
| `user_id`       | Optional user identifier                     |
| `session_id`    | Optional session identifier                  |
| `error_msg`     | Exception message on failure                 |

---

## Supported models & pricing

Pricing is embedded in `pricing/prices.json` and used for cost calculation. Ollama models are always free.

| Provider   | Models                                          |
|------------|-------------------------------------------------|
| **Ollama** | qwen2.5, llama3.x, phi3.5/4, mistral, gemma2, deepseek-r1, … (free) |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o3-mini, … |
| **Anthropic** | claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, … |

Unknown models default to $0.00 cost and are still tracked for tokens and latency.

---

## What llm-ledger does NOT do

- **No prompt storage** — the content of your messages is never logged
- **No cloud sync** — all data stays in your local SQLite file
- **No network calls** — zero outbound connections, ever
- **No token counting library** — tokens come from `response.usage` only (what the API reports)
- **No framework required** — works standalone alongside LangChain, LlamaIndex, or raw API calls

---

## Roadmap

- [x] `@watchdog` decorator
- [x] `watch()` context manager
- [x] `Budget()` enforcement
- [x] CLI: `summary`, `tail`, `top`
- [ ] `llm_ledger.intercept()` — zero-code-change global monkey-patch (Day 7)
- [ ] Anthropic provider (Day 8)
- [ ] `watchdog export --format csv/json` (Day 9)
- [ ] Streaming response support (Day 10)

---

## Development

```bash
git clone https://github.com/yourname/llm-ledger
cd llm-ledger
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (no Ollama needed)
pytest tests/unit/ -v

# Run integration tests (requires Ollama running)
pytest tests/integration/ -v -m integration

# Lint
ruff check src/
```

---

## License

MIT