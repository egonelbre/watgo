package wasmvm_test

import (
	"fmt"

	"github.com/eliben/watgo"
	"github.com/eliben/watgo/wasmir"
	"github.com/eliben/watgo/wasmvm"
)

// ExampleNewRuntime shows how to instantiate a validated module and call one
// of its exported functions.
func ExampleNewRuntime() {
	m, err := watgo.ParseAndValidateWAT([]byte(`
(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )
)`))
	if err != nil {
		panic(err)
	}

	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(m, nil)
	if err != nil {
		panic(err)
	}

	add, ok := inst.ExportedFunc("add")
	if !ok {
		panic("missing add export")
	}
	results, err := add.Call(wasmvm.I32(20), wasmvm.I32(22))
	if err != nil {
		panic(err)
	}

	fmt.Println(results[0].I32)
	// Output:
	// 42
}

// ExampleNewHostFunc shows how to satisfy a WebAssembly function import with a
// Go callback.
func ExampleNewHostFunc() {
	m, err := watgo.ParseAndValidateWAT([]byte(`
(module
  (import "env" "inc" (func $inc (param i32) (result i32)))
  (func (export "call_inc") (param i32) (result i32)
    local.get 0
    call $inc
  )
)`))
	if err != nil {
		panic(err)
	}

	imports := wasmvm.Imports{
		"env": {
			"inc": wasmvm.NewHostFunc(
				[]wasmir.ValueType{wasmir.ValueTypeI32},
				[]wasmir.ValueType{wasmir.ValueTypeI32},
				func(_ *wasmvm.Context, args []wasmvm.Value) ([]wasmvm.Value, error) {
					return []wasmvm.Value{wasmvm.I32(args[0].I32 + 1)}, nil
				},
			),
		},
	}

	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(m, imports)
	if err != nil {
		panic(err)
	}

	callInc, ok := inst.ExportedFunc("call_inc")
	if !ok {
		panic("missing call_inc export")
	}
	results, err := callInc.Call(wasmvm.I32(41))
	if err != nil {
		panic(err)
	}

	fmt.Println(results[0].I32)
	// Output:
	// 42
}
