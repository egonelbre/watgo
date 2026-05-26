# watgo Notes

`watgo` is a Go toolkit for parsing, printing, validating, and encoding
WebAssembly. It also includes `wasmvm`, a small interpreter runtime intended
for testing and experimentation rather than high-performance execution.

## Public API

The public entry points are in [watgo.go](../watgo.go):

- `ParseWAT`: WAT -> `wasmir.Module`
- `ParseAndValidateWAT`: WAT -> validated `wasmir.Module`
- `DecodeWASM`: binary wasm -> `wasmir.Module`
- `ValidateModule`: semantic validation over `wasmir.Module`
- `EncodeWASM`: `wasmir.Module` -> binary wasm
- `PrintWAT`: `wasmir.Module` -> WAT
- `CompileWATToWASM`: parse + lower + validate + encode

Runtime support lives in package `wasmvm`. Its main public entry points are:

- `NewRuntime`: create an empty runtime
- `Runtime.Instantiate`: instantiate a validated `wasmir.Module`
- `ModuleInstance.ExportedFunc`: look up a callable exported function
- `NewHostFunc`: satisfy a WebAssembly function import with a Go callback

## Internal Structure

- `wasmir`: semantic IR and public IR types
- `internal/textformat`: WAT parsing and lowering
- `internal/binaryformat`: wasm binary decoding/encoding
- `internal/printer`: WAT printing from `wasmir`
- `internal/validate`: semantic validation
- `wasmvm`: public runtime API
- `internal/vm`: private interpreter engine used by `wasmvm`
- `internal/instrdef`: shared instruction catalog used by text, binary, and
  validation code

The main pipeline is:

1. WAT -> `textformat` -> `wasmir`
2. wasm binary -> `binaryformat` -> `wasmir`
3. `wasmir` -> `validate`
4. `wasmir` -> `binaryformat` encoder
5. `wasmir` -> `printer` -> WAT
6. validated `wasmir` -> `wasmvm` -> interpreted execution

`wasmir` is the canonical semantic representation. Text-specific source details
such as folded syntax and literal spelling are intentionally not preserved
there. Binary name-section metadata is preserved where `wasmir` has explicit
name fields.

## Testing

- Unit tests cover parser, encoder/decoder, validator, and CLI layers.
- `wasmvm` unit tests cover the interpreter API and instruction behavior.
- `tests/wasmspec` runs `.wast` scripts against `watgo`.
- The wasmspec harness uses Node as the broad compatibility execution backend
  and also runs selected coverage through `wasmvm`.
- Detailed wasmspec tracing is off by default and can be enabled with
  `WATGO_WASMSPEC_DEBUG=1`.
