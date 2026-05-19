package tests

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/eliben/watgo"
	"github.com/eliben/watgo/internal/binaryformat"
	"github.com/eliben/watgo/internal/validate"
	"github.com/eliben/watgo/wasmir"
	"github.com/eliben/watgo/wasmvm"
)

// Allow list for scripts
var wasmSpecWasmvmScripts = []string{
	"scripts/i32.wast",
	"scripts/nop.wast",
}

type wasmSpecWasmvmRunner struct {
	rt                 *wasmvm.Runtime
	current            *wasmvm.ModuleInstance
	currentMeta        *moduleMetadata
	currentRuntimeName string
	instances          map[string]*wasmvm.ModuleInstance
	moduleMeta         map[string]*moduleMetadata
	moduleAlias        map[string]string
}

// wasmSpecWasmvmBackend returns the initial wasmvm-backed wasmspec runner.
func wasmSpecWasmvmBackend(t *testing.T) wasmSpecBackend {
	t.Helper()

	return wasmSpecBackend{
		name:    "wasmvm",
		scripts: wasmSpecWasmvmScripts,
		run:     runWasmSpecWasmvmScript,
	}
}

// newWasmSpecWasmvmRunner creates an empty wasmvm script runner.
func newWasmSpecWasmvmRunner() *wasmSpecWasmvmRunner {
	return &wasmSpecWasmvmRunner{
		rt:          wasmvm.NewRuntime(),
		instances:   map[string]*wasmvm.ModuleInstance{},
		moduleMeta:  map[string]*moduleMetadata{},
		moduleAlias: map[string]string{},
	}
}

// runWasmSpecWasmvmScript executes parsed wasmspec commands through wasmvm.
func runWasmSpecWasmvmScript(t *testing.T, scriptPath string, commands []scriptCommand, opts runOptions) []commandResult {
	t.Helper()

	runner := newWasmSpecWasmvmRunner()
	return runner.run(commands, opts)
}

// run executes commands in script order and returns one result per command.
func (r *wasmSpecWasmvmRunner) run(commands []scriptCommand, opts runOptions) []commandResult {
	results := make([]commandResult, 0, len(commands))
	for i, cmd := range commands {
		if opts.progress != nil {
			opts.progress(i, len(commands), cmd)
		}
		start := time.Now()
		res := commandResult{
			index: i,
			kind:  cmd.kind,
			loc:   cmd.loc,
		}
		r.runCommand(&res, cmd, opts)
		if opts.progressDone != nil {
			opts.progressDone(i, len(commands), cmd, res, time.Since(start))
		}
		results = append(results, res)
	}
	return results
}

// runCommand dispatches one parsed wasmspec command to wasmvm behavior.
func (r *wasmSpecWasmvmRunner) runCommand(res *commandResult, cmd scriptCommand, opts runOptions) {
	switch cmd.kind {
	case commandModule:
		r.runModule(res, cmd)
	case commandInvoke:
		r.runInvoke(res, cmd)
	case commandRegister:
		r.runRegister(res, cmd)
	case commandAssertReturn:
		r.runAssertReturn(res, cmd)
	case commandAssertTrap:
		r.runAssertTrap(res, cmd)
	case commandAssertExhaustion:
		r.runAssertExhaustion(res, cmd)
	case commandAssertUnlinkable:
		r.runAssertUnlinkable(res, cmd, opts)
	case commandAssertInvalid:
		r.runAssertInvalid(res, cmd, opts)
	case commandAssertMalformed:
		r.runAssertMalformed(res, cmd, opts)
	default:
		res.status = false
		res.detail = fmt.Sprintf("unsupported wasmvm command kind %q", cmd.kind)
	}
}

// runModule compiles and instantiates a top-level module command.
func (r *wasmSpecWasmvmRunner) runModule(res *commandResult, cmd scriptCommand) {
	m, meta, err := compileWasmSpecCommandForWasmvm(cmd)
	if err != nil {
		res.status = false
		res.detail = fmt.Sprintf("module compile failed: %v", err)
		return
	}

	inst, err := r.rt.Instantiate(m, nil)
	if err != nil {
		res.status = false
		res.detail = fmt.Sprintf("module instantiate failed: %v", err)
		return
	}

	runtimeName := runtimeModuleName(cmd.moduleName)
	r.current = inst
	r.currentMeta = meta
	r.currentRuntimeName = runtimeName
	r.instances[runtimeName] = inst
	r.moduleMeta[runtimeName] = meta
	if cmd.moduleName != "" {
		r.moduleAlias[cmd.moduleName] = runtimeName
	}
	res.status = true
}

// runRegister aliases an already-instantiated wasmvm module for later named
// actions.
func (r *wasmSpecWasmvmRunner) runRegister(res *commandResult, cmd scriptCommand) {
	if cmd.registerName == "" {
		res.status = false
		res.detail = "register command missing name"
		return
	}
	sourceName := cmd.registerFrom
	if sourceName == "" {
		sourceName = r.currentRuntimeName
	}
	inst, meta, err := r.lookupInstance(sourceName)
	if err != nil {
		res.status = false
		res.detail = fmt.Sprintf("register source failed: %v", err)
		return
	}
	r.instances[cmd.registerName] = inst
	r.moduleMeta[cmd.registerName] = meta
	r.moduleAlias[cmd.registerName] = cmd.registerName
	if cmd.registerFrom != "" {
		r.moduleAlias[cmd.registerFrom] = cmd.registerName
	}
	if sourceName == r.currentRuntimeName {
		r.current = inst
		r.currentMeta = meta
		r.currentRuntimeName = cmd.registerName
	}
	res.status = true
}

// runInvoke invokes an exported function and requires the call to succeed.
func (r *wasmSpecWasmvmRunner) runInvoke(res *commandResult, cmd scriptCommand) {
	if _, err := r.invoke(cmd.action); err != nil {
		res.status = false
		res.detail = fmt.Sprintf("invoke failed: %v", err)
		return
	}
	res.status = true
}

// runAssertReturn invokes an action and compares its results with expected
// values.
func (r *wasmSpecWasmvmRunner) runAssertReturn(res *commandResult, cmd scriptCommand) {
	if cmd.getAction != nil {
		res.status = false
		res.detail = "wasmvm backend does not support get assertions yet"
		return
	}
	results, err := r.invoke(cmd.action)
	if err != nil {
		res.status = false
		res.detail = fmt.Sprintf("action failed: %v", err)
		return
	}

	if len(results) != len(cmd.expectValues) {
		res.status = false
		res.detail = fmt.Sprintf("result arity mismatch: got %d want %d", len(results), len(cmd.expectValues))
		return
	}
	for i := range results {
		want := cmd.expectValues[i]
		if !runtimeValueMatchesExpected(results[i], want) {
			res.status = false
			res.detail = fmt.Sprintf("result[%d] mismatch: got %s want %s", i, formatGotValueLikeExpected(results[i], want), formatExpectedValue(want))
			return
		}
	}
	res.status = true
}

// runAssertTrap requires an invocation or start-time instantiation to trap.
func (r *wasmSpecWasmvmRunner) runAssertTrap(res *commandResult, cmd scriptCommand) {
	var err error
	if cmd.moduleExpr != nil {
		err = r.instantiateTrapModule(cmd)
	} else {
		_, err = r.invoke(cmd.action)
	}
	if err == nil {
		res.status = false
		res.detail = "expected trap, got success"
		return
	}
	if cmd.expectText != "" && !matchesExpectedFailureText(err.Error(), cmd.expectText) {
		res.status = false
		res.detail = fmt.Sprintf("trap text mismatch: got %q want substring %q", err.Error(), cmd.expectText)
		return
	}
	res.status = true
}

// runAssertExhaustion requires an invocation to fail with an exhaustion-style
// error.
func (r *wasmSpecWasmvmRunner) runAssertExhaustion(res *commandResult, cmd scriptCommand) {
	_, err := r.invoke(cmd.action)
	if err == nil {
		res.status = false
		res.detail = "expected exhaustion, got success"
		return
	}
	if cmd.expectText != "" && !matchesExpectedFailureText(err.Error(), cmd.expectText) {
		res.status = false
		res.detail = fmt.Sprintf("exhaustion text mismatch: got %q want substring %q", err.Error(), cmd.expectText)
		return
	}
	res.status = true
}

// runAssertUnlinkable requires module compilation or instantiation to fail.
func (r *wasmSpecWasmvmRunner) runAssertUnlinkable(res *commandResult, cmd scriptCommand, opts runOptions) {
	m, _, err := compileWasmSpecCommandForWasmvm(cmd)
	if err == nil {
		_, err = r.rt.Instantiate(m, nil)
	}
	if err == nil {
		res.status = false
		res.detail = "expected unlinkable module error, got success"
		return
	}
	if opts.strictErrorText && cmd.expectText != "" && !strings.Contains(err.Error(), cmd.expectText) {
		res.status = false
		res.detail = fmt.Sprintf("unlinkable error text mismatch: got %q want substring %q", err.Error(), cmd.expectText)
		return
	}
	res.status = true
}

// runAssertInvalid requires module validation to fail.
func (r *wasmSpecWasmvmRunner) runAssertInvalid(res *commandResult, cmd scriptCommand, opts runOptions) {
	err := compileWasmSpecInvalidModule(cmd)
	if err == nil {
		res.status = false
		res.detail = "expected invalid module error, got success"
		return
	}
	if opts.strictErrorText && cmd.expectText != "" && !strings.Contains(err.Error(), cmd.expectText) {
		res.status = false
		res.detail = fmt.Sprintf("invalid error text mismatch: got %q want substring %q", err.Error(), cmd.expectText)
		return
	}
	res.status = true
}

// runAssertMalformed requires module text or binary decoding to fail.
func (r *wasmSpecWasmvmRunner) runAssertMalformed(res *commandResult, cmd scriptCommand, opts runOptions) {
	err := compileWasmSpecMalformedModule(cmd)
	if err == nil {
		res.status = false
		res.detail = "expected malformed module error, got success"
		return
	}
	if opts.strictErrorText && cmd.expectText != "" && !malformedErrorMatches(err.Error(), cmd.expectText) {
		res.status = false
		res.detail = fmt.Sprintf("malformed error text mismatch: got %q want substring %q", err.Error(), cmd.expectText)
		return
	}
	res.status = true
}

// instantiateTrapModule compiles cmd's module expression and returns the
// instantiation error expected by an assert_trap command.
func (r *wasmSpecWasmvmRunner) instantiateTrapModule(cmd scriptCommand) error {
	m, _, err := compileWasmSpecCommandForWasmvm(cmd)
	if err != nil {
		return err
	}
	_, err = r.rt.Instantiate(m, nil)
	return err
}

// invoke calls one exported function through wasmvm and converts results to
// the wasmspec harness value representation.
func (r *wasmSpecWasmvmRunner) invoke(action *invokeAction) ([]runtimeValue, error) {
	if action == nil {
		return nil, fmt.Errorf("nil invoke action")
	}
	inst, meta, err := r.lookupInstance(action.moduleName)
	if err != nil {
		return nil, err
	}
	sig, ok := meta.funcExports[action.funcName]
	if !ok {
		return nil, fmt.Errorf("exported function %q not found", action.funcName)
	}
	if len(sig.Params) != len(action.args) {
		return nil, fmt.Errorf("invoke arg arity mismatch for %q: got %d want %d", action.funcName, len(action.args), len(sig.Params))
	}
	f, ok := inst.ExportedFunc(action.funcName)
	if !ok {
		return nil, fmt.Errorf("exported function %q not found in wasmvm instance", action.funcName)
	}
	args := make([]wasmvm.Value, len(action.args))
	for i, arg := range action.args {
		args[i], err = scriptValueToWasmvmValue(arg, sig.Params[i])
		if err != nil {
			return nil, fmt.Errorf("arg[%d]: %w", i, err)
		}
	}
	results, err := f.Call(args...)
	if err != nil {
		return nil, err
	}
	return wasmvmValuesToRuntimeValues(results)
}

// lookupInstance resolves a script module name to a wasmvm instance and its
// decoded export metadata.
func (r *wasmSpecWasmvmRunner) lookupInstance(scriptModuleName string) (*wasmvm.ModuleInstance, *moduleMetadata, error) {
	if scriptModuleName == "" {
		if r.current == nil || r.currentMeta == nil {
			return nil, nil, fmt.Errorf("no current module")
		}
		return r.current, r.currentMeta, nil
	}
	runtimeName := scriptModuleName
	if aliased, ok := r.moduleAlias[scriptModuleName]; ok {
		runtimeName = aliased
	}
	inst, ok := r.instances[runtimeName]
	if !ok {
		return nil, nil, fmt.Errorf("module %q not found", scriptModuleName)
	}
	meta, ok := r.moduleMeta[runtimeName]
	if !ok || meta == nil {
		return nil, nil, fmt.Errorf("module metadata for %q not found", runtimeName)
	}
	return inst, meta, nil
}

// compileWasmSpecCommandForWasmvm compiles a module-bearing command into a
// validated module and decoded export metadata.
func compileWasmSpecCommandForWasmvm(cmd scriptCommand) (*wasmir.Module, *moduleMetadata, error) {
	wasmBytes, ok, err := compileWasmSpecCommandForPrintRoundTrip(cmd)
	if err != nil {
		return nil, nil, err
	}
	if !ok {
		return nil, nil, fmt.Errorf("command %s does not contain a wasm module", cmd.kind)
	}
	m, err := watgo.DecodeWASM(wasmBytes)
	if err != nil {
		return nil, nil, err
	}
	if err := watgo.ValidateModule(m); err != nil {
		return nil, nil, err
	}
	meta, err := decodeModuleMetadata(wasmBytes)
	if err != nil {
		return nil, nil, err
	}
	return m, meta, nil
}

// compileWasmSpecInvalidModule compiles cmd as an invalid-module assertion and
// returns the validation error, if any.
func compileWasmSpecInvalidModule(cmd scriptCommand) error {
	if isModuleBinaryExpr(cmd.moduleExpr) {
		wasmBytes, err := parseBinaryModuleBytes(cmd.moduleExpr)
		if err != nil {
			return err
		}
		m, err := binaryformat.DecodeModule(wasmBytes)
		if err != nil {
			return err
		}
		return validate.ValidateModule(m, nil)
	}
	_, _, err := compileWasmSpecModuleTextFromCommand(cmd)
	return err
}

// compileWasmSpecMalformedModule compiles cmd as a malformed-module assertion
// and returns the text or binary decoding error, if any.
func compileWasmSpecMalformedModule(cmd scriptCommand) error {
	if isModuleBinaryExpr(cmd.moduleExpr) {
		wasmBytes, err := parseBinaryModuleBytes(cmd.moduleExpr)
		if err != nil {
			return err
		}
		_, err = binaryformat.DecodeModule(wasmBytes)
		return err
	}
	_, err := watgo.CompileWATToWASM([]byte(cmd.quotedWAT))
	return err
}

// compileWasmSpecModuleTextFromCommand compiles the text module carried by cmd.
func compileWasmSpecModuleTextFromCommand(cmd scriptCommand) ([]byte, bool, error) {
	if cmd.quotedWAT != "" {
		return compileWasmSpecModuleText(cmd.quotedWAT)
	}
	src, err := sexprToWAT(cmd.moduleExpr)
	if err != nil {
		return nil, false, fmt.Errorf("module text generation failed: %w", err)
	}
	return compileWasmSpecModuleText(src)
}

// scriptValueToWasmvmValue converts a parsed spec-script constant argument to
// a wasmvm runtime value of targetType.
func scriptValueToWasmvmValue(arg scriptValue, targetType wasmir.ValueType) (wasmvm.Value, error) {
	switch arg.kind {
	case valueI32Const:
		return wasmvm.I32(int32(uint32(arg.bits))), nil
	case valueI64Const:
		return wasmvm.I64(int64(arg.bits)), nil
	case valueF32Const:
		return wasmvm.F32(math.Float32frombits(uint32(arg.bits))), nil
	case valueF64Const:
		return wasmvm.F64(math.Float64frombits(arg.bits)), nil
	case valueF32NaNCanonical, valueF32NaNArithmetic:
		return wasmvm.F32(math.Float32frombits(0x7fc00000)), nil
	case valueF64NaNCanonical, valueF64NaNArithmetic:
		return wasmvm.F64(math.Float64frombits(0x7ff8000000000000)), nil
	case valueV128Const:
		return wasmvm.Value{Type: wasmir.ValueTypeV128, V128: arg.v128}, nil
	case valueRefNull:
		return wasmvm.Value{Type: targetType}, nil
	default:
		return wasmvm.Value{}, fmt.Errorf("unsupported invoke arg kind %q", arg.kind)
	}
}

// wasmvmValuesToRuntimeValues converts wasmvm results to the wasmspec harness
// value representation.
func wasmvmValuesToRuntimeValues(values []wasmvm.Value) ([]runtimeValue, error) {
	out := make([]runtimeValue, len(values))
	for i, value := range values {
		converted, err := wasmvmValueToRuntimeValue(value)
		if err != nil {
			return nil, fmt.Errorf("result[%d]: %w", i, err)
		}
		out[i] = converted
	}
	return out, nil
}

// wasmvmValueToRuntimeValue converts one wasmvm result to the wasmspec harness
// value representation.
func wasmvmValueToRuntimeValue(value wasmvm.Value) (runtimeValue, error) {
	switch value.Type.Kind {
	case wasmir.ValueKindI32:
		return runtimeValue{scalar: uint64(uint32(value.I32))}, nil
	case wasmir.ValueKindI64:
		return runtimeValue{scalar: uint64(value.I64)}, nil
	case wasmir.ValueKindF32:
		return runtimeValue{scalar: uint64(math.Float32bits(value.F32))}, nil
	case wasmir.ValueKindF64:
		return runtimeValue{scalar: math.Float64bits(value.F64)}, nil
	case wasmir.ValueKindV128:
		return runtimeValue{v128: value.V128, isV128: true}, nil
	case wasmir.ValueKindRef:
		if value.Ref.Kind == 0 {
			return runtimeValue{}, nil
		}
		return runtimeValue{scalar: 1}, nil
	default:
		return runtimeValue{}, fmt.Errorf("unsupported wasmvm value type %s", value.Type)
	}
}
