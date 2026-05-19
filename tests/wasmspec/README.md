# Wasm Spec Harness

This directory contains the integration harness for running `.wast` scripts from
the WebAssembly spec tests against `watgo`.

Detailed per-command/debug tracing is disabled by default. To enable it for a
run, set `WATGO_WASMSPEC_DEBUG=1`.

## Files

- [scripts/](./scripts)
  The actual `.wast` script corpus.
- [wasmspec_test.go](./wasmspec_test.go)
  Discovers scripts under `scripts/`, runs each selected script as a Go subtest,
  and reports per-command failures through a backend-independent core.
- [wasmspec_node_test.go](./wasmspec_node_test.go)
  Runs the corpus through Node.js.
- [wasmspec_wasmvm_test.go](./wasmspec_wasmvm_test.go)
  Runs the corpus through `wasmvm`.
- [wasmspec_harness.go](./wasmspec_harness.go)
  Script parsing, command models, shared comparison helpers, and the Node
  command runner.
- [node_wasm_runner.js](./node_wasm_runner.js)
  A small JSON-over-stdio bridge to Node's `WebAssembly` API.

## High-Level Flow

For each `.wast` file:

1. [wasmspec_test.go](./wasmspec_test.go) reads the script and parses it into
   top-level commands such as `module`, `invoke`, `assert_return`,
   `assert_trap`, and so on.
2. [wasmspec_test.go](./wasmspec_test.go) passes those commands to the selected
   backend.
3. Text modules are compiled by `watgo`; binary modules are decoded directly.
4. Modules are instantiated and invoked through either
   [node_wasm_runner.js](./node_wasm_runner.js) or `wasmvm`.
5. The shared harness compares results and trap text against the script's expected
   assertions.

The important point is that `.wast` execution is stateful: later commands can
refer to modules instantiated earlier in the same script, use `(register ...)`,
and observe mutated memory/table/global state.

## Backends

The Node backend is the broad compatibility backend and runs the full corpus.
The `wasmvm` backend discovers the corpus too, then filters out a deny list for
unsupported or partially implemented features.

For local triage, `-wasmspec.wasmvm-scripts` can override the `wasmvm` script
list with comma-separated `scripts/...` paths:

```bash
go test ./tests/wasmspec -run TestWasmSpecScriptsWasmvm -count=1 -args -wasmspec.wasmvm-scripts=scripts/f32.wast
```

### Node Runtime

[node_wasm_runner.js](./node_wasm_runner.js) keeps one Node process alive for
the duration of a single `.wast` file. The Go harness sends line-delimited JSON
requests like:

- `instantiate`
- `validate`
- `invoke`
- `get`

and receives JSON responses back.

This bridge is intentionally narrow: it only supports the value kinds and
operations needed by the current spec tests.

Some wasm results are awkward to observe exactly through the JS embedding API:

- `f32`/`f64` results can lose exact NaN payload information when converted to
  JS numbers.
- `v128` results are not directly exposed as raw SIMD bytes in a useful form.
- `anyref` sometimes needs in-wasm classification with `ref.test`.

To handle this, the Go harness renders small helper modules as WAT, compiles
them with `watgo`, and sends the resulting wasm bytes to
[node_wasm_runner.js](./node_wasm_runner.js). These wrappers import the target
function, call it inside wasm, and convert the result into a JS-friendly exact
form such as integer bits or raw bytes before it crosses back into JS.
