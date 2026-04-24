---
name: strict-code-review
description: Performs strict code review for quality, correctness, performance, security, privacy, and maintainability. Use when the user asks for a code review, "кодревью", review of changes/PR, or requests to audit code for bugs, leaks, inefficiencies, or security issues.
---

# Strict Code Review

## Scope
Apply this skill when reviewing:
- a PR / branch / commit series
- a diff / patch
- a file or module for refactor readiness
- a suspected bug/perf/memory leak/security issue

## Review output format (required)
Return feedback grouped as:
- **🔴 Blockers**: must fix before merge
- **🟡 Important**: high value improvements (should fix soon)
- **🟢 Suggestions**: optional improvements
- **✅ Positives**: what’s good and should be kept

For each item include:
- **Location**: file + symbol or line range
- **Impact**: user impact / risk / perf / security
- **Recommendation**: concrete change
- **Confidence**: high/medium/low (only when not obvious)

## Workflow
1. **Understand the change**
   - Summarize what changed and why (from diff / PR description).
   - Identify entrypoints, inputs, outputs, side effects.
2. **Risk scan**
   - Identify the highest-risk areas first (auth, payments, data writes, parsing, concurrency, migrations, infra).
3. **Deep review pass**
   - Apply the checklist below systematically.
4. **Verification**
   - If possible, run tests/lints/build locally.
   - If runtime behavior matters, propose minimal reproduction steps.

## Strict checklist

### Correctness & robustness
- Validate assumptions about inputs (nullability, empty states, encoding, locales, timezones).
- Ensure boundary conditions and error paths are handled (timeouts, retries, partial failures).
- Check idempotency and replay safety for handlers/jobs.
- Confirm deterministic behavior where required (sorting, pagination, stable outputs).
- Avoid undefined behavior and reliance on incidental ordering.

### Security & privacy
- No secrets in code/logs/config (tokens, keys, cookies, session ids).
- No unsafe string interpolation into commands/SQL/URLs/HTML.
- Proper escaping/encoding for output contexts (HTML/JS/URL/SQL).
- SSRF, path traversal, open redirects, deserialization hazards.
- Authorization checks at the right layer (not only UI).
- PII handling: minimize logging, redact sensitive fields, respect data retention.

### Resource leaks & lifecycle (memory/file/network)
- Close files/sockets/streams; use context managers / finally blocks.
- Avoid accumulating unbounded data in memory (lists, caches, logs).
- Ensure background tasks are cancellable and don’t leak threads/processes.
- Check connection pooling, timeouts, and max sizes.

### Performance & efficiency
- Algorithmic complexity: avoid accidental \(O(n^2)\) in hot paths.
- Avoid repeated expensive work (N+1 calls, repeated parsing/serialization).
- Prefer batching, caching with invalidation strategy, streaming when appropriate.
- Consider large payloads: pagination, limits, backpressure.
- Watch for excessive allocations/copies, string concatenations, regex backtracking.

### Concurrency & consistency
- Data races: shared mutable state, global caches, singleton clients.
- Correct locking/atomicity for multi-step writes.
- Transaction boundaries and isolation (if DB).
- Retry behavior: exponential backoff, jitter, exactly-once vs at-least-once semantics.

### API/UX behavior (when applicable)
- Backward compatibility for public APIs (payload shape, status codes, headers).
- Clear error messages; don’t leak internals to users.
- Stable contracts and versioning; deprecations documented.

### Observability & operability
- Logs: actionable, structured, no noise, no sensitive data.
- Metrics/tracing: key counters, latency, error rates for critical paths.
- Feature flags / gradual rollout plans for risky changes.
- Runbooks or docs for non-trivial ops workflows.

### Code quality & maintainability
- Readability: naming, cohesion, minimal cleverness, clear invariants.
- No duplicated logic when it increases bug surface; extract shared code wisely.
- Keep functions/modules small and testable.
- Prefer explicit types/contracts at module boundaries.

### Testing & validation
- Tests cover: happy path + edge cases + error paths.
- If refactor: ensure behavior parity via golden tests/snapshots where suitable.
- For bugfix: add regression test that fails before and passes after.
- Verify new dependencies/licenses and lockfiles updates.

## Standard “red flags” (auto-blockers unless justified)
- Silent exception swallowing / broad catch without handling.
- Missing timeouts on network calls.
- Writing secrets/PII to logs.
- Unbounded retries or loops.
- Removing validation or auth checks.
- Disabling lint/tests without replacement.

## Tips for reviewing diffs quickly
- Start with public interfaces and call sites.
- Trace data flow: input → validation → transformation → side effects → output.
- Evaluate failure modes first for critical paths.
