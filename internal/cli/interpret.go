package cli

import (
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/eliben/watgo"
	"github.com/eliben/watgo/wasmir"
	"github.com/eliben/watgo/wasmvm"
)

// runInterpretOptions executes `watgo interpret` after CLI arguments have been
// parsed.
func runInterpretOptions(opts interpretOptions, stdin io.Reader, stdout, stderr io.Writer) int {
	src, err := readInput(opts.inputPath, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "watgo interpret: %v\n", err)
		return 1
	}
	m, err := parseValidatedModule(src)
	if err != nil {
		fmt.Fprintf(stderr, "watgo interpret: %v\n", err)
		return 1
	}

	vmModule, imports, err := interpretImports(m, opts.hostPrint, stdout)
	if err != nil {
		fmt.Fprintf(stderr, "watgo interpret: %v\n", err)
		return 1
	}
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(vmModule, imports)
	if err != nil {
		fmt.Fprintf(stderr, "watgo interpret: %v\n", err)
		return 1
	}

	if opts.invokeName == "" {
		return 0
	}
	fn, ok := inst.ExportedFunc(opts.invokeName)
	if !ok {
		fmt.Fprintf(stderr, "watgo interpret: exported function %q not found\n", opts.invokeName)
		return 1
	}
	params, _, err := fn.Type()
	if err != nil {
		fmt.Fprintf(stderr, "watgo interpret: %v\n", err)
		return 1
	}
	if len(opts.invokeArgs) != len(params) {
		fmt.Fprintf(stderr, "watgo interpret: exported function %q expects %d arguments, got %d\n",
			opts.invokeName, len(params), len(opts.invokeArgs))
		return 2
	}
	values := make([]wasmvm.Value, len(params))
	for i, param := range params {
		values[i], err = parseRuntimeValue(opts.invokeArgs[i], param)
		if err != nil {
			fmt.Fprintf(stderr, "watgo interpret: argument %d: %v\n", i, err)
			return 2
		}
	}

	results, err := fn.Call(values...)
	if err != nil {
		fmt.Fprintf(stderr, "watgo interpret: %v\n", err)
		return 1
	}
	if len(results) == 0 {
		return 0
	}
	fmt.Fprintf(stdout, "%s =>", formatInvocationLabel(opts.invokeName, opts.invokeArgs))
	formatted := make([]string, len(results))
	for i, result := range results {
		formatted[i] = formatRuntimeValue(result)
	}
	fmt.Fprintf(stdout, " %s", strings.Join(formatted, ", "))
	fmt.Fprintln(stdout)
	return 0
}

// parseValidatedModule decodes or parses src, then validates the resulting
// module before it is passed to wasmvm.
func parseValidatedModule(src []byte) (*wasmir.Module, error) {
	if isBinaryWasm(src) {
		m, err := watgo.DecodeWASM(src)
		if err != nil {
			return nil, err
		}
		if err := watgo.ValidateModule(m); err != nil {
			return nil, err
		}
		return m, nil
	}
	return watgo.ParseAndValidateWAT(src)
}

// interpretImports constructs the import object map used for interpretation.
//
// When hostPrint is enabled, fixed host.print_i32, host.print_i64,
// host.print_f32, and host.print_f64 imports are supplied.
func interpretImports(m *wasmir.Module, hostPrint bool, stdout io.Writer) (*wasmir.Module, wasmvm.Imports, error) {
	if !hostPrint {
		return m, nil, nil
	}

	return m, wasmvm.Imports{
		"host": {
			"print_i32": interpretHostPrint(wasmir.ValueTypeI32, stdout),
			"print_i64": interpretHostPrint(wasmir.ValueTypeI64, stdout),
			"print_f32": interpretHostPrint(wasmir.ValueTypeF32, stdout),
			"print_f64": interpretHostPrint(wasmir.ValueTypeF64, stdout),
		},
	}, nil
}

// interpretHostPrint returns one fixed numeric host printing callback.
func interpretHostPrint(typ wasmir.ValueType, stdout io.Writer) *wasmvm.HostFunc {
	return wasmvm.NewHostFunc([]wasmir.ValueType{typ}, nil, func(_ *wasmvm.Context, args []wasmvm.Value) ([]wasmvm.Value, error) {
		fmt.Fprintln(stdout, formatPrintValue(args[0]))
		return nil, nil
	})
}

// parseRuntimeValue converts one CLI argument into a wasmvm runtime value of
// the expected WebAssembly type.
func parseRuntimeValue(text string, typ wasmir.ValueType) (wasmvm.Value, error) {
	if _, value, ok := strings.Cut(text, ":"); ok {
		text = value
	}
	switch typ.Kind {
	case wasmir.ValueKindI32:
		v, err := parseIntegerBits(text, 32)
		if err != nil {
			return wasmvm.Value{}, fmt.Errorf("parse i32 %q: %w", text, err)
		}
		return wasmvm.I32(int32(uint32(v))), nil
	case wasmir.ValueKindI64:
		v, err := parseIntegerBits(text, 64)
		if err != nil {
			return wasmvm.Value{}, fmt.Errorf("parse i64 %q: %w", text, err)
		}
		return wasmvm.I64(int64(v)), nil
	case wasmir.ValueKindF32:
		v, err := strconv.ParseFloat(text, 32)
		if err != nil {
			return wasmvm.Value{}, fmt.Errorf("parse f32 %q: %w", text, err)
		}
		return wasmvm.F32(float32(v)), nil
	case wasmir.ValueKindF64:
		v, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return wasmvm.Value{}, fmt.Errorf("parse f64 %q: %w", text, err)
		}
		return wasmvm.F64(v), nil
	default:
		return wasmvm.Value{}, fmt.Errorf("unsupported argument type %s", typ)
	}
}

// parseIntegerBits parses signed or unsigned integer text into the requested
// bit width while preserving the resulting bit pattern.
func parseIntegerBits(text string, bitSize int) (uint64, error) {
	if v, err := strconv.ParseInt(text, 0, bitSize); err == nil {
		return uint64(v), nil
	}
	return strconv.ParseUint(text, 0, bitSize)
}

// formatInvocationLabel formats the left side of an invocation result line.
func formatInvocationLabel(name string, args []string) string {
	if len(args) == 0 {
		return name + "()"
	}
	return name + "(" + strings.Join(args, ", ") + ")"
}

// formatRuntimeValue formats one wasmvm value using a WABT-style
// kind:value representation.
func formatRuntimeValue(v wasmvm.Value) string {
	kind := valueKindName(v.Type)
	switch v.Type.Kind {
	case wasmir.ValueKindI32:
		return kind + ":" + strconv.FormatUint(uint64(uint32(v.I32)), 10)
	case wasmir.ValueKindI64:
		return kind + ":" + strconv.FormatUint(uint64(v.I64), 10)
	case wasmir.ValueKindF32:
		return kind + ":" + strconv.FormatFloat(float64(v.F32), 'f', 6, 32)
	case wasmir.ValueKindF64:
		return kind + ":" + strconv.FormatFloat(v.F64, 'f', 6, 64)
	case wasmir.ValueKindV128:
		return kind + ":" + fmt.Sprintf("%x", v.V128)
	case wasmir.ValueKindRef:
		if v.Ref.Kind == wasmvm.RefKindNull {
			return kind + ":0"
		}
		return kind + ":1"
	default:
		return kind + ":?"
	}
}

// formatPrintValue formats the bare value printed by host print callbacks.
func formatPrintValue(v wasmvm.Value) string {
	switch v.Type.Kind {
	case wasmir.ValueKindI32:
		return strconv.FormatInt(int64(v.I32), 10)
	case wasmir.ValueKindI64:
		return strconv.FormatInt(v.I64, 10)
	case wasmir.ValueKindF32:
		return strconv.FormatFloat(float64(v.F32), 'g', -1, 32)
	case wasmir.ValueKindF64:
		return strconv.FormatFloat(v.F64, 'g', -1, 64)
	default:
		return formatRuntimeValue(v)
	}
}

// valueKindName returns the short text label used for a WebAssembly value type
// in interpreter output.
func valueKindName(vt wasmir.ValueType) string {
	switch vt.Kind {
	case wasmir.ValueKindI32:
		return "i32"
	case wasmir.ValueKindI64:
		return "i64"
	case wasmir.ValueKindF32:
		return "f32"
	case wasmir.ValueKindF64:
		return "f64"
	case wasmir.ValueKindV128:
		return "v128"
	case wasmir.ValueKindRef:
		switch vt.HeapType.Kind {
		case wasmir.HeapKindFunc, wasmir.HeapKindNoFunc:
			return "funcref"
		case wasmir.HeapKindExtern, wasmir.HeapKindNoExtern:
			return "externref"
		default:
			return "ref"
		}
	default:
		return vt.String()
	}
}
