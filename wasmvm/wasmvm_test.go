package wasmvm_test

import (
	"math"
	"testing"

	"github.com/eliben/watgo"
	"github.com/eliben/watgo/wasmir"
	"github.com/eliben/watgo/wasmvm"
)

func parseWAT(t *testing.T, src string) *wasmir.Module {
	t.Helper()

	m, err := watgo.ParseAndValidateWAT([]byte(src))
	if err != nil {
		t.Fatalf("ParseAndValidateWAT failed: %v", err)
	}
	return m
}

func callExport(t *testing.T, inst *wasmvm.ModuleInstance, name string, args ...wasmvm.Value) []wasmvm.Value {
	t.Helper()

	f, ok := inst.ExportedFunc(name)
	if !ok {
		t.Fatalf("missing %s export", name)
	}
	results, err := f.Call(args...)
	if err != nil {
		t.Fatalf("Call %s failed: %v", name, err)
	}
	return results
}

func TestExportedAdd(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "add") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.add))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "add", wasmvm.I32(3), wasmvm.I32(4))
	if len(results) != 1 || results[0] != wasmvm.I32(7) {
		t.Fatalf("got results %#v, want i32 7", results)
	}
}

// TestNop checks that nop executes without changing the operand stack or
// control flow.
func TestNop(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "run") (result i32)
				nop
				i32.const 40
				nop
				i32.const 2
				i32.add
				nop))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "run")
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("run got results %#v, want i32 42", results)
	}
}

func TestHostFunctionImport(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(import "env" "inc" (func $inc (param i32) (result i32)))
			(func (export "call_inc") (param i32) (result i32)
				local.get 0
				call $inc))
	`), wasmvm.Imports{
		"env": {
			"inc": wasmvm.NewHostFunc(
				[]wasmir.ValueType{wasmir.ValueTypeI32},
				[]wasmir.ValueType{wasmir.ValueTypeI32},
				func(_ *wasmvm.Context, args []wasmvm.Value) ([]wasmvm.Value, error) {
					return []wasmvm.Value{wasmvm.I32(args[0].I32 + 1)}, nil
				},
			),
		},
	})
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "call_inc", wasmvm.I32(41))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("got results %#v, want i32 42", results)
	}
}

// TestReturnCall checks that return_call invokes a module-defined function and
// immediately returns its results from the current function.
func TestReturnCall(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func $add (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.add)
			(func (export "tail_add") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				return_call $add
				i32.const 99))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "tail_add", wasmvm.I32(20), wasmvm.I32(22))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("tail_add got results %#v, want i32 42", results)
	}
}

// TestReturnCallHostFunction checks that return_call can tail-call an imported
// host function through the same resolver path as call.
func TestReturnCallHostFunction(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(import "env" "double" (func $double (param i32) (result i32)))
			(func (export "tail_double") (param i32) (result i32)
				local.get 0
				return_call $double
				i32.const 99))
	`), wasmvm.Imports{
		"env": {
			"double": wasmvm.NewHostFunc(
				[]wasmir.ValueType{wasmir.ValueTypeI32},
				[]wasmir.ValueType{wasmir.ValueTypeI32},
				func(_ *wasmvm.Context, args []wasmvm.Value) ([]wasmvm.Value, error) {
					return []wasmvm.Value{wasmvm.I32(args[0].I32 * 2)}, nil
				},
			),
		},
	})
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "tail_double", wasmvm.I32(21))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("tail_double got results %#v, want i32 42", results)
	}
}

// TestReferenceInstructions checks the minimal reference instruction set:
// ref.null, ref.func, and ref.is_null.
func TestReferenceInstructions(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func $target)
			(elem declare func $target)
			(func (export "null_is_null") (result i32)
				ref.null func
				ref.is_null)
			(func (export "func_is_null") (result i32)
				ref.func $target
				ref.is_null)
			(func (export "return_null") (result funcref)
				ref.null func)
			(func (export "return_func") (result funcref)
				ref.func $target)
			(func (export "local_func_is_null") (result i32)
				(local funcref)
				ref.func $target
				local.set 0
				local.get 0
				ref.is_null))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "null_is_null")
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("null_is_null got results %#v, want i32 1", results)
	}
	results = callExport(t, inst, "func_is_null")
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("func_is_null got results %#v, want i32 0", results)
	}
	results = callExport(t, inst, "local_func_is_null")
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("local_func_is_null got results %#v, want i32 0", results)
	}

	results = callExport(t, inst, "return_null")
	if len(results) != 1 || !results[0].Type.IsRef() {
		t.Fatalf("return_null got results %#v, want one reference", results)
	}
	results = callExport(t, inst, "return_func")
	if len(results) != 1 || !results[0].Type.IsRef() || results[0].Ref.FuncIndex != 0 {
		t.Fatalf("return_func got results %#v, want function reference 0", results)
	}
}

// TestReferenceTableAndGlobalOps checks reference values moving through
// globals and tables.
func TestReferenceTableAndGlobalOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(table $t_func 1 funcref)
			(table $t_extern 1 externref)
			(elem funcref (ref.func 2) (ref.null func))
			(elem externref (ref.null extern))
			(global $g (mut funcref) (ref.null func))

			(func (export "ref_null_func") (result funcref)
				ref.null func)
			(func (export "ref_null_extern") (result externref)
				ref.null extern)
			(func $ref_is_null_func (export "ref_is_null_func") (result i32)
				global.get $g
				ref.is_null)
			(func $ref_func (export "ref_func") (result funcref)
				ref.func $ref_is_null_func)
			(func $table_set (export "table_set")
				i32.const 0
				call $ref_func
				table.set $t_func)
			(func (export "table_get") (result i32)
				call $table_set
				i32.const 0
				table.get $t_func
				ref.is_null)
			(func $global_set (export "global_set") (result funcref)
				ref.func $ref_is_null_func
				global.set $g
				global.get $g))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "ref_is_null_func")
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("ref_is_null_func got results %#v, want i32 1", results)
	}
	results = callExport(t, inst, "ref_func")
	if len(results) != 1 || !results[0].Type.IsRef() || results[0].Ref.FuncIndex != 2 {
		t.Fatalf("ref_func got results %#v, want function reference 2", results)
	}
	callExport(t, inst, "table_set")
	results = callExport(t, inst, "table_get")
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("table_get got results %#v, want i32 0", results)
	}
	results = callExport(t, inst, "global_set")
	if len(results) != 1 || !results[0].Type.IsRef() || results[0].Ref.FuncIndex != 2 {
		t.Fatalf("global_set got results %#v, want function reference 2", results)
	}
}

// TestTableBasics checks module-defined table instantiation, active element
// initialization, table.size, table.get, and table.set.
func TestTableBasics(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(table 3 funcref)
			(func $a)
			(func $b)
			(elem (i32.const 1) func $a)
			(elem declare func $b)
			(func (export "size") (result i32)
				table.size)
			(func (export "is_null") (param i32) (result i32)
				local.get 0
				table.get
				ref.is_null)
			(func (export "set_b") (param i32)
				local.get 0
				ref.func $b
				table.set))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "size")
	if len(results) != 1 || results[0] != wasmvm.I32(3) {
		t.Fatalf("size got results %#v, want i32 3", results)
	}
	results = callExport(t, inst, "is_null", wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("is_null(0) got results %#v, want i32 1", results)
	}
	results = callExport(t, inst, "is_null", wasmvm.I32(1))
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("is_null(1) got results %#v, want i32 0", results)
	}
	callExport(t, inst, "set_b", wasmvm.I32(2))
	results = callExport(t, inst, "is_null", wasmvm.I32(2))
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("is_null(2) after set got results %#v, want i32 0", results)
	}
}

// TestSharedTableImportExport checks the public table import/export API and
// cross-module function references in a shared table.
func TestSharedTableImportExport(t *testing.T) {
	rt := wasmvm.NewRuntime()

	owner, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $out-i32 (func (result i32)))
			(table (export "shared-table") 3 funcref)
			(func (export "call-1") (result i32)
				i32.const 1
				call_indirect (type $out-i32)))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate owner failed: %v", err)
	}
	sharedTable, ok := owner.ExportedTable("shared-table")
	if !ok {
		t.Fatal("missing shared-table export")
	}
	if got, want := sharedTable.Size(), uint64(3); got != want {
		t.Fatalf("shared table size got %d, want %d", got, want)
	}

	writer, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $out-i32 (func (result i32)))
			(import "owner" "shared-table" (table 3 funcref))
			(elem (i32.const 1) $answer)
			(func $answer (type $out-i32)
				i32.const 42))
	`), wasmvm.Imports{
		"owner": {
			"shared-table": sharedTable,
		},
	})
	if err != nil {
		t.Fatalf("Instantiate writer failed: %v", err)
	}
	_ = writer

	results := callExport(t, owner, "call-1")
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("call-1 got results %#v, want i32 42", results)
	}
}

// TestTableGrowFillCopy checks table.grow failure/success behavior and the
// bulk table.fill/table.copy operations over reference values.
func TestTableGrowFillCopy(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(table 2 5 funcref)
			(func $a)
			(func $b)
			(elem declare func $a $b)
			(func (export "size") (result i32)
				table.size)
			(func (export "is_null") (param i32) (result i32)
				local.get 0
				table.get
				ref.is_null)
			(func (export "grow") (param i32) (result i32)
				ref.func $a
				local.get 0
				table.grow)
			(func (export "fill_b") (param i32 i32)
				local.get 0
				ref.func $b
				local.get 1
				table.fill)
			(func (export "fill_null") (param i32 i32)
				local.get 0
				ref.null func
				local.get 1
				table.fill)
			(func (export "copy") (param i32 i32 i32)
				local.get 0
				local.get 1
				local.get 2
				table.copy))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "size")
	if len(results) != 1 || results[0] != wasmvm.I32(2) {
		t.Fatalf("size got results %#v, want i32 2", results)
	}
	results = callExport(t, inst, "grow", wasmvm.I32(2))
	if len(results) != 1 || results[0] != wasmvm.I32(2) {
		t.Fatalf("grow got results %#v, want old size i32 2", results)
	}
	results = callExport(t, inst, "size")
	if len(results) != 1 || results[0] != wasmvm.I32(4) {
		t.Fatalf("size after grow got results %#v, want i32 4", results)
	}
	results = callExport(t, inst, "grow", wasmvm.I32(2))
	if len(results) != 1 || results[0] != wasmvm.I32(-1) {
		t.Fatalf("over-max grow got results %#v, want i32 -1", results)
	}

	callExport(t, inst, "fill_null", wasmvm.I32(2), wasmvm.I32(2))
	results = callExport(t, inst, "is_null", wasmvm.I32(2))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("is_null(2) after fill_null got results %#v, want i32 1", results)
	}
	callExport(t, inst, "fill_b", wasmvm.I32(0), wasmvm.I32(2))
	callExport(t, inst, "copy", wasmvm.I32(2), wasmvm.I32(0), wasmvm.I32(2))
	results = callExport(t, inst, "is_null", wasmvm.I32(3))
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("is_null(3) after copy got results %#v, want i32 0", results)
	}
}

// TestPassiveElementSegments checks that table.init can copy from a passive
// element segment and elem.drop makes that segment unavailable afterward.
func TestPassiveElementSegments(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(table 2 funcref)
			(func $a)
			(elem $e funcref (ref.func $a))
			(func (export "init") (param i32 i32 i32)
				local.get 0
				local.get 1
				local.get 2
				table.init $e)
			(func (export "drop")
				elem.drop $e)
			(func (export "is_null") (param i32) (result i32)
				local.get 0
				table.get
				ref.is_null))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "is_null", wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("is_null(0) before init got results %#v, want i32 1", results)
	}
	callExport(t, inst, "init", wasmvm.I32(0), wasmvm.I32(0), wasmvm.I32(1))
	results = callExport(t, inst, "is_null", wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("is_null(0) after init got results %#v, want i32 0", results)
	}
	callExport(t, inst, "drop")
	initFunc, ok := inst.ExportedFunc("init")
	if !ok {
		t.Fatal("missing init export")
	}
	_, err = initFunc.Call(wasmvm.I32(1), wasmvm.I32(0), wasmvm.I32(1))
	if err == nil {
		t.Fatal("Call init after elem.drop succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 3 table.init: element segment out of bounds"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestIndirectCalls checks call_indirect and return_call_indirect through a
// funcref table populated by an active element segment.
func TestIndirectCalls(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $binary (func (param i32 i32) (result i32)))
			(table 3 funcref)
			(func $add (type $binary)
				local.get 0
				local.get 1
				i32.add)
			(func $sub (type $binary)
				local.get 0
				local.get 1
				i32.sub)
			(elem (i32.const 0) func $add $sub)
			(func (export "call") (param i32 i32 i32) (result i32)
				local.get 0
				local.get 1
				local.get 2
				call_indirect (type $binary))
			(func (export "tail") (param i32 i32 i32) (result i32)
				local.get 0
				local.get 1
				local.get 2
				return_call_indirect (type $binary)))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "call", wasmvm.I32(20), wasmvm.I32(22), wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("call add got results %#v, want i32 42", results)
	}
	results = callExport(t, inst, "call", wasmvm.I32(50), wasmvm.I32(8), wasmvm.I32(1))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("call sub got results %#v, want i32 42", results)
	}
	results = callExport(t, inst, "tail", wasmvm.I32(45), wasmvm.I32(3), wasmvm.I32(1))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("tail got results %#v, want i32 42", results)
	}
}

// TestIndirectCallTraps checks the runtime traps specific to indirect calls:
// null table elements and function references whose type does not match the
// call_indirect type immediate.
func TestIndirectCallTraps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $binary (func (param i32 i32) (result i32)))
			(type $unary (func (param i32) (result i32)))
			(table 3 funcref)
			(func $add (type $binary)
				local.get 0
				local.get 1
				i32.add)
			(func $inc (type $unary)
				local.get 0
				i32.const 1
				i32.add)
			(elem (i32.const 0) func $add $inc)
			(func (export "call") (param i32 i32 i32) (result i32)
				local.get 0
				local.get 1
				local.get 2
				call_indirect (type $binary)))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	call, ok := inst.ExportedFunc("call")
	if !ok {
		t.Fatal("missing call export")
	}

	_, err = call.Call(wasmvm.I32(1), wasmvm.I32(2), wasmvm.I32(1))
	if err == nil {
		t.Fatal("Call with mismatched indirect target succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 3 call_indirect: indirect call type mismatch"; got != want {
		t.Fatalf("type mismatch error = %q, want %q", got, want)
	}

	_, err = call.Call(wasmvm.I32(1), wasmvm.I32(2), wasmvm.I32(2))
	if err == nil {
		t.Fatal("Call through null table slot succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 3 call_indirect: indirect call to null reference"; got != want {
		t.Fatalf("null reference error = %q, want %q", got, want)
	}
}

// TestCallRef checks call_ref and return_call_ref through function references
// produced by ref.func.
func TestCallRef(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $binary (func (param i32 i32) (result i32)))
			(func $add (type $binary)
				local.get 0
				local.get 1
				i32.add)
			(func $sub (type $binary)
				local.get 0
				local.get 1
				i32.sub)
			(elem declare func $add $sub)
			(func (export "call") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				ref.func $add
				call_ref $binary)
			(func (export "tail") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				ref.func $sub
				return_call_ref $binary))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "call", wasmvm.I32(20), wasmvm.I32(22))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("call got results %#v, want i32 42", results)
	}
	results = callExport(t, inst, "tail", wasmvm.I32(50), wasmvm.I32(8))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("tail got results %#v, want i32 42", results)
	}
}

// TestCallRefTraps checks call_ref traps for null references and runtime
// function type mismatches.
func TestCallRefTraps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $binary (func (param i32 i32) (result i32)))
			(func (export "call_null") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				ref.null $binary
				call_ref $binary))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	callNull, ok := inst.ExportedFunc("call_null")
	if !ok {
		t.Fatal("missing call_null export")
	}
	_, err = callNull.Call(wasmvm.I32(1), wasmvm.I32(2))
	if err == nil {
		t.Fatal("Call through null reference succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 3 call_ref: call_ref to null reference"; got != want {
		t.Fatalf("null reference error = %q, want %q", got, want)
	}

	err = callInvalidCallRefRuntimeModule(t)
	if got, want := err.Error(), "pc 3 call_ref: indirect call type mismatch"; got != want {
		t.Fatalf("type mismatch error = %q, want %q", got, want)
	}
}

// TestRefAsNonNull checks that ref.as_non_null passes through a non-null
// function reference and can feed call_ref.
func TestRefAsNonNull(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $binary (func (param i32 i32) (result i32)))
			(func $add (type $binary)
				local.get 0
				local.get 1
				i32.add)
			(elem declare func $add)
			(func (export "call_checked") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				ref.func $add
				ref.as_non_null
				call_ref $binary)
			(func (export "is_null_after_check") (result i32)
				ref.func $add
				ref.as_non_null
				ref.is_null))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "call_checked", wasmvm.I32(20), wasmvm.I32(22))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("call_checked got results %#v, want i32 42", results)
	}
	results = callExport(t, inst, "is_null_after_check")
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("is_null_after_check got results %#v, want i32 0", results)
	}
}

// TestRefAsNonNullTrap checks that ref.as_non_null traps when the reference
// operand is null.
func TestRefAsNonNullTrap(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $binary (func (param i32 i32) (result i32)))
			(func (export "check_null")
				ref.null $binary
				ref.as_non_null
				drop))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	checkNull, ok := inst.ExportedFunc("check_null")
	if !ok {
		t.Fatal("missing check_null export")
	}
	_, err = checkNull.Call()
	if err == nil {
		t.Fatal("ref.as_non_null on null reference succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 1 ref.as_non_null: ref.as_non_null to null reference"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestExternRefRoundTrip checks that externref values keep their opaque
// identity through locals, tables, and function results.
func TestExternRefRoundTrip(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(table 2 externref)
			(func (export "store") (param externref)
				i32.const 1
				local.get 0
				table.set)
			(func (export "load") (result externref)
				i32.const 1
				table.get)
			(func (export "is_null") (param externref) (result i32)
				local.get 0
				ref.is_null))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	ref := wasmvm.ExternRef(17)
	results := callExport(t, inst, "is_null", ref)
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("is_null got results %#v, want i32 0", results)
	}

	callExport(t, inst, "store", ref)
	results = callExport(t, inst, "load")
	if len(results) != 1 || results[0] != ref {
		t.Fatalf("load got results %#v, want externref %#v", results, ref)
	}
}

// TestBrOnNull checks both br_on_null paths: a null reference branches and a
// non-null reference falls through as a refined function reference.
func TestBrOnNull(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $result (func (result i32)))
			(func $forty_two (type $result)
				i32.const 42)
			(elem declare func $forty_two)
			(func (export "null_branch") (result i32)
				block $null
					ref.null $result
					br_on_null $null
					drop
					i32.const 99
					return
				end
				i32.const 42)
			(func (export "nonnull_fallthrough") (result i32)
				block $null
					ref.func $forty_two
					br_on_null $null
					call_ref $result
					return
				end
				i32.const 99))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "null_branch")
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("null_branch got results %#v, want i32 42", results)
	}
	results = callExport(t, inst, "nonnull_fallthrough")
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("nonnull_fallthrough got results %#v, want i32 42", results)
	}
}

// TestBrOnNonNull checks both br_on_non_null paths: a non-null reference
// branches as a label value and a null reference is consumed on fallthrough.
func TestBrOnNonNull(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $result (func (result i32)))
			(func $seven (type $result)
				i32.const 7)
			(func $forty_two (type $result)
				i32.const 42)
			(elem declare func $seven $forty_two)
			(func (export "nonnull_branch") (result i32)
				block $target (result (ref $result))
					ref.func $forty_two
					br_on_non_null $target
					ref.func $seven
				end
				call_ref $result)
			(func (export "null_fallthrough") (result i32)
				block $target (result (ref $result))
					ref.null $result
					br_on_non_null $target
					ref.func $seven
				end
				call_ref $result))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "nonnull_branch")
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("nonnull_branch got results %#v, want i32 42", results)
	}
	results = callExport(t, inst, "null_fallthrough")
	if len(results) != 1 || results[0] != wasmvm.I32(7) {
		t.Fatalf("null_fallthrough got results %#v, want i32 7", results)
	}
}

func TestI32Arithmetic(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "calc") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.mul
				i32.const 7
				i32.sub))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "calc", wasmvm.I32(6), wasmvm.I32(5))
	if len(results) != 1 || results[0] != wasmvm.I32(23) {
		t.Fatalf("got results %#v, want i32 23", results)
	}
}

func TestLocalSetAndTee(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "locals") (param i32) (result i32)
				(local i32)
				local.get 0
				local.set 1
				local.get 1
				i32.const 3
				i32.add
				local.tee 1
				local.get 1
				i32.add))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "locals", wasmvm.I32(4))
	if len(results) != 1 || results[0] != wasmvm.I32(14) {
		t.Fatalf("got results %#v, want i32 14", results)
	}
}

func TestSelect(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $result (func (result i32)))
			(func $forty_two (type $result)
				i32.const 42)
			(elem declare func $forty_two)
			(func (export "pick_i32") (param i32) (result i32)
				i32.const 10
				i32.const 20
				local.get 0
				select)
			(func (export "pick_f64") (param i32) (result f64)
				f64.const 1.5
				f64.const 2.5
				local.get 0
				select)
			(func (export "pick_typed_i64") (param i32) (result i64)
				i64.const 30
				i64.const 40
				local.get 0
				select (result i64))
			(func (export "pick_typed_ref") (param i32) (result i32)
				ref.func $forty_two
				ref.null $result
				local.get 0
				select (result (ref null $result))
				ref.as_non_null
				call_ref $result)
			(func (export "pick_null_ref_is_null") (param i32) (result i32)
				ref.func $forty_two
				ref.null $result
				local.get 0
				select (result (ref null $result))
				ref.is_null))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "pick_i32", wasmvm.I32(1))
	if len(results) != 1 || results[0] != wasmvm.I32(10) {
		t.Fatalf("pick_i32(1) got results %#v, want i32 10", results)
	}
	results = callExport(t, inst, "pick_i32", wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I32(20) {
		t.Fatalf("pick_i32(0) got results %#v, want i32 20", results)
	}

	results = callExport(t, inst, "pick_f64", wasmvm.I32(-1))
	if len(results) != 1 || results[0] != wasmvm.F64(1.5) {
		t.Fatalf("pick_f64 got results %#v, want f64 1.5", results)
	}

	results = callExport(t, inst, "pick_typed_i64", wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I64(40) {
		t.Fatalf("pick_typed_i64 got results %#v, want i64 40", results)
	}
	results = callExport(t, inst, "pick_typed_ref", wasmvm.I32(1))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("pick_typed_ref got results %#v, want i32 42", results)
	}
	results = callExport(t, inst, "pick_null_ref_is_null", wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("pick_null_ref_is_null got results %#v, want i32 1", results)
	}
}

// TestRefEqAndConversions checks small reference operations that do not require
// allocating GC objects in the current wasmvm slice.
func TestRefEqAndConversions(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "null_eq") (result i32)
				ref.null eq
				ref.null eq
				ref.eq)
			(func (export "null_any_to_extern_is_null") (result i32)
				ref.null any
				extern.convert_any
				ref.is_null)
			(func (export "null_extern_to_any_is_null") (result i32)
				ref.null extern
				any.convert_extern
				ref.is_null))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "null_eq")
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("null_eq got results %#v, want i32 1", results)
	}
	results = callExport(t, inst, "null_any_to_extern_is_null")
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("null_any_to_extern_is_null got results %#v, want i32 1", results)
	}
	results = callExport(t, inst, "null_extern_to_any_is_null")
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("null_extern_to_any_is_null got results %#v, want i32 1", results)
	}
}

func TestI32Predicates(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "eqz") (param i32) (result i32)
				local.get 0
				i32.eqz)
			(func (export "cmp") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.lt_s
				local.get 0
				local.get 1
				i32.ne
				i32.add)
			(func (export "eq") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.eq)
			(func (export "le") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.le_s)
			(func (export "gt") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.gt_s)
			(func (export "ge") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.ge_s))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "eqz", wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("eqz(0) got results %#v, want i32 1", results)
	}
	results = callExport(t, inst, "eqz", wasmvm.I32(9))
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("eqz(9) got results %#v, want i32 0", results)
	}

	results = callExport(t, inst, "cmp", wasmvm.I32(-2), wasmvm.I32(5))
	if len(results) != 1 || results[0] != wasmvm.I32(2) {
		t.Fatalf("cmp got results %#v, want i32 2", results)
	}

	for _, tt := range []struct {
		name string
		lhs  int32
		rhs  int32
		want int32
	}{
		{name: "eq", lhs: 8, rhs: 8, want: 1},
		{name: "le", lhs: -3, rhs: -2, want: 1},
		{name: "gt", lhs: 10, rhs: 4, want: 1},
		{name: "ge", lhs: 5, rhs: 5, want: 1},
	} {
		results = callExport(t, inst, tt.name, wasmvm.I32(tt.lhs), wasmvm.I32(tt.rhs))
		if len(results) != 1 || results[0] != wasmvm.I32(tt.want) {
			t.Fatalf("%s got results %#v, want i32 %d", tt.name, results, tt.want)
		}
	}
}

func TestI32ExtendedIntegerOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "div_s") (param i32 i32) (result i32) local.get 0 local.get 1 i32.div_s)
			(func (export "div_u") (param i32 i32) (result i32) local.get 0 local.get 1 i32.div_u)
			(func (export "rem_s") (param i32 i32) (result i32) local.get 0 local.get 1 i32.rem_s)
			(func (export "rem_u") (param i32 i32) (result i32) local.get 0 local.get 1 i32.rem_u)
			(func (export "and") (param i32 i32) (result i32) local.get 0 local.get 1 i32.and)
			(func (export "or") (param i32 i32) (result i32) local.get 0 local.get 1 i32.or)
			(func (export "xor") (param i32 i32) (result i32) local.get 0 local.get 1 i32.xor)
			(func (export "shl") (param i32 i32) (result i32) local.get 0 local.get 1 i32.shl)
			(func (export "shr_s") (param i32 i32) (result i32) local.get 0 local.get 1 i32.shr_s)
			(func (export "shr_u") (param i32 i32) (result i32) local.get 0 local.get 1 i32.shr_u)
			(func (export "rotl") (param i32 i32) (result i32) local.get 0 local.get 1 i32.rotl)
			(func (export "rotr") (param i32 i32) (result i32) local.get 0 local.get 1 i32.rotr)
			(func (export "lt_u") (param i32 i32) (result i32) local.get 0 local.get 1 i32.lt_u)
			(func (export "le_u") (param i32 i32) (result i32) local.get 0 local.get 1 i32.le_u)
			(func (export "gt_u") (param i32 i32) (result i32) local.get 0 local.get 1 i32.gt_u)
			(func (export "ge_u") (param i32 i32) (result i32) local.get 0 local.get 1 i32.ge_u))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		lhs  int32
		rhs  int32
		want int32
	}{
		{name: "div_s", lhs: -7, rhs: 2, want: -3},
		{name: "div_u", lhs: -1, rhs: 2, want: 2147483647},
		{name: "rem_s", lhs: -7, rhs: 2, want: -1},
		{name: "rem_u", lhs: -1, rhs: 10, want: 5},
		{name: "and", lhs: 0x0f0f, rhs: 0x00ff, want: 0x000f},
		{name: "or", lhs: 0x0f0f, rhs: 0x00ff, want: 0x0fff},
		{name: "xor", lhs: 0x0f0f, rhs: 0x00ff, want: 0x0ff0},
		{name: "shl", lhs: 1, rhs: 33, want: 2},
		{name: "shr_s", lhs: -4, rhs: 1, want: -2},
		{name: "shr_u", lhs: -4, rhs: 1, want: 2147483646},
		{name: "rotl", lhs: 1, rhs: 33, want: 2},
		{name: "rotr", lhs: 2, rhs: 33, want: 1},
		{name: "lt_u", lhs: -1, rhs: 1, want: 0},
		{name: "le_u", lhs: -1, rhs: -1, want: 1},
		{name: "gt_u", lhs: -1, rhs: 1, want: 1},
		{name: "ge_u", lhs: 0, rhs: -1, want: 0},
	} {
		results := callExport(t, inst, tt.name, wasmvm.I32(tt.lhs), wasmvm.I32(tt.rhs))
		if len(results) != 1 || results[0] != wasmvm.I32(tt.want) {
			t.Fatalf("%s got results %#v, want i32 %d", tt.name, results, tt.want)
		}
	}
}

// TestIntegerUnaryAndSignExtension checks core integer unary operators and
// sign-extension operators for both i32 and i64.
func TestIntegerUnaryAndSignExtension(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i32_counts") (param i32) (result i32 i32 i32)
				local.get 0
				i32.clz
				local.get 0
				i32.ctz
				local.get 0
				i32.popcnt)
			(func (export "i64_counts") (param i64) (result i64 i64 i64)
				local.get 0
				i64.clz
				local.get 0
				i64.ctz
				local.get 0
				i64.popcnt)
			(func (export "i32_ext8") (param i32) (result i32)
				local.get 0
				i32.extend8_s)
			(func (export "i32_ext16") (param i32) (result i32)
				local.get 0
				i32.extend16_s)
			(func (export "i64_ext8") (param i64) (result i64)
				local.get 0
				i64.extend8_s)
			(func (export "i64_ext16") (param i64) (result i64)
				local.get 0
				i64.extend16_s)
			(func (export "i64_ext32") (param i64) (result i64)
				local.get 0
				i64.extend32_s))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "i32_counts", wasmvm.I32(0x00f00000))
	if len(results) != 3 || results[0] != wasmvm.I32(8) || results[1] != wasmvm.I32(20) || results[2] != wasmvm.I32(4) {
		t.Fatalf("i32_counts got results %#v, want [8 20 4]", results)
	}
	results = callExport(t, inst, "i32_counts", wasmvm.I32(0))
	if len(results) != 3 || results[0] != wasmvm.I32(32) || results[1] != wasmvm.I32(32) || results[2] != wasmvm.I32(0) {
		t.Fatalf("i32_counts(0) got results %#v, want [32 32 0]", results)
	}

	results = callExport(t, inst, "i64_counts", wasmvm.I64(0x00f0000000000000))
	if len(results) != 3 || results[0] != wasmvm.I64(8) || results[1] != wasmvm.I64(52) || results[2] != wasmvm.I64(4) {
		t.Fatalf("i64_counts got results %#v, want [8 52 4]", results)
	}
	results = callExport(t, inst, "i64_counts", wasmvm.I64(0))
	if len(results) != 3 || results[0] != wasmvm.I64(64) || results[1] != wasmvm.I64(64) || results[2] != wasmvm.I64(0) {
		t.Fatalf("i64_counts(0) got results %#v, want [64 64 0]", results)
	}

	for _, tt := range []struct {
		name string
		arg  wasmvm.Value
		want wasmvm.Value
	}{
		{name: "i32_ext8", arg: wasmvm.I32(0x80), want: wasmvm.I32(-128)},
		{name: "i32_ext16", arg: wasmvm.I32(0x8001), want: wasmvm.I32(-32767)},
		{name: "i64_ext8", arg: wasmvm.I64(0xff), want: wasmvm.I64(-1)},
		{name: "i64_ext16", arg: wasmvm.I64(0x8001), want: wasmvm.I64(-32767)},
		{name: "i64_ext32", arg: wasmvm.I64(0x80000001), want: wasmvm.I64(-2147483647)},
	} {
		results := callExport(t, inst, tt.name, tt.arg)
		if len(results) != 1 || results[0] != tt.want {
			t.Fatalf("%s got results %#v, want %v", tt.name, results, tt.want)
		}
	}
}

func TestI64ArithmeticAndPredicates(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "calc") (param i64 i64) (result i64)
				local.get 0
				local.get 1
				i64.mul
				i64.const 9
				i64.sub)
			(func (export "eqz") (param i64) (result i32)
				local.get 0
				i64.eqz)
			(func (export "cmp") (param i64 i64) (result i32)
				local.get 0
				local.get 1
				i64.ge_s))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "calc", wasmvm.I64(8), wasmvm.I64(7))
	if len(results) != 1 || results[0] != wasmvm.I64(47) {
		t.Fatalf("calc got results %#v, want i64 47", results)
	}

	results = callExport(t, inst, "eqz", wasmvm.I64(0))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("eqz got results %#v, want i32 1", results)
	}

	results = callExport(t, inst, "cmp", wasmvm.I64(-2), wasmvm.I64(5))
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("cmp got results %#v, want i32 0", results)
	}
}

func TestI64ExtendedIntegerOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "div_s") (param i64 i64) (result i64) local.get 0 local.get 1 i64.div_s)
			(func (export "div_u") (param i64 i64) (result i64) local.get 0 local.get 1 i64.div_u)
			(func (export "rem_s") (param i64 i64) (result i64) local.get 0 local.get 1 i64.rem_s)
			(func (export "rem_u") (param i64 i64) (result i64) local.get 0 local.get 1 i64.rem_u)
			(func (export "and") (param i64 i64) (result i64) local.get 0 local.get 1 i64.and)
			(func (export "or") (param i64 i64) (result i64) local.get 0 local.get 1 i64.or)
			(func (export "xor") (param i64 i64) (result i64) local.get 0 local.get 1 i64.xor)
			(func (export "shl") (param i64 i64) (result i64) local.get 0 local.get 1 i64.shl)
			(func (export "shr_s") (param i64 i64) (result i64) local.get 0 local.get 1 i64.shr_s)
			(func (export "shr_u") (param i64 i64) (result i64) local.get 0 local.get 1 i64.shr_u)
			(func (export "rotl") (param i64 i64) (result i64) local.get 0 local.get 1 i64.rotl)
			(func (export "rotr") (param i64 i64) (result i64) local.get 0 local.get 1 i64.rotr)
			(func (export "lt_u") (param i64 i64) (result i32) local.get 0 local.get 1 i64.lt_u)
			(func (export "le_u") (param i64 i64) (result i32) local.get 0 local.get 1 i64.le_u)
			(func (export "gt_u") (param i64 i64) (result i32) local.get 0 local.get 1 i64.gt_u)
			(func (export "ge_u") (param i64 i64) (result i32) local.get 0 local.get 1 i64.ge_u))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		lhs  int64
		rhs  int64
		want int64
	}{
		{name: "div_s", lhs: -9, rhs: 2, want: -4},
		{name: "div_u", lhs: -1, rhs: 3, want: 6148914691236517205},
		{name: "rem_s", lhs: -9, rhs: 2, want: -1},
		{name: "rem_u", lhs: -1, rhs: 10, want: 5},
		{name: "and", lhs: 0x0f0f, rhs: 0x00ff, want: 0x000f},
		{name: "or", lhs: 0x0f0f, rhs: 0x00ff, want: 0x0fff},
		{name: "xor", lhs: 0x0f0f, rhs: 0x00ff, want: 0x0ff0},
		{name: "shl", lhs: 1, rhs: 65, want: 2},
		{name: "shr_s", lhs: -8, rhs: 1, want: -4},
		{name: "shr_u", lhs: -8, rhs: 1, want: 9223372036854775804},
		{name: "rotl", lhs: 1, rhs: 65, want: 2},
		{name: "rotr", lhs: 2, rhs: 65, want: 1},
	} {
		results := callExport(t, inst, tt.name, wasmvm.I64(tt.lhs), wasmvm.I64(tt.rhs))
		if len(results) != 1 || results[0] != wasmvm.I64(tt.want) {
			t.Fatalf("%s got results %#v, want i64 %d", tt.name, results, tt.want)
		}
	}

	for _, tt := range []struct {
		name string
		lhs  int64
		rhs  int64
		want int32
	}{
		{name: "lt_u", lhs: -1, rhs: 1, want: 0},
		{name: "le_u", lhs: -1, rhs: -1, want: 1},
		{name: "gt_u", lhs: -1, rhs: 1, want: 1},
		{name: "ge_u", lhs: 0, rhs: -1, want: 0},
	} {
		results := callExport(t, inst, tt.name, wasmvm.I64(tt.lhs), wasmvm.I64(tt.rhs))
		if len(results) != 1 || results[0] != wasmvm.I32(tt.want) {
			t.Fatalf("%s got results %#v, want i32 %d", tt.name, results, tt.want)
		}
	}
}

func TestIntegerTrapErrors(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i32_div_zero") (result i32)
				i32.const 1
				i32.const 0
				i32.div_s)
			(func (export "i32_div_overflow") (result i32)
				i32.const -2147483648
				i32.const -1
				i32.div_s)
			(func (export "i64_div_zero") (result i64)
				i64.const 1
				i64.const 0
				i64.div_s)
			(func (export "i64_div_overflow") (result i64)
				i64.const -9223372036854775808
				i64.const -1
				i64.div_s))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		want string
	}{
		{name: "i32_div_zero", want: "pc 2 i32.div_s: integer divide by zero"},
		{name: "i32_div_overflow", want: "pc 2 i32.div_s: integer overflow"},
		{name: "i64_div_zero", want: "pc 2 i64.div_s: integer divide by zero"},
		{name: "i64_div_overflow", want: "pc 2 i64.div_s: integer overflow"},
	} {
		f, ok := inst.ExportedFunc(tt.name)
		if !ok {
			t.Fatalf("missing %s export", tt.name)
		}
		_, err := f.Call()
		if err == nil {
			t.Fatalf("Call %s succeeded unexpectedly", tt.name)
		}
		if got := err.Error(); got != tt.want {
			t.Fatalf("%s error = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestF32ArithmeticAndPredicates(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "calc") (param f32) (result f32)
				local.get 0
				f32.const 2.5
				f32.mul
				f32.const 1.0
				f32.add)
			(func (export "cmp") (param f32 f32) (result i32)
				local.get 0
				local.get 1
				f32.lt))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "calc", wasmvm.F32(4))
	if len(results) != 1 || results[0] != wasmvm.F32(11) {
		t.Fatalf("calc got results %#v, want f32 11", results)
	}

	results = callExport(t, inst, "cmp", wasmvm.F32(-1.5), wasmvm.F32(2.25))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("cmp got results %#v, want i32 1", results)
	}
}

// TestF32UnaryOps checks non-converting f32 unary instructions, including
// nearest's ties-to-even behavior.
func TestF32UnaryOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "abs") (param f32) (result f32) local.get 0 f32.abs)
			(func (export "neg") (param f32) (result f32) local.get 0 f32.neg)
			(func (export "sqrt") (param f32) (result f32) local.get 0 f32.sqrt)
			(func (export "ceil") (param f32) (result f32) local.get 0 f32.ceil)
			(func (export "floor") (param f32) (result f32) local.get 0 f32.floor)
			(func (export "trunc") (param f32) (result f32) local.get 0 f32.trunc)
			(func (export "nearest") (param f32) (result f32) local.get 0 f32.nearest))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		arg  float32
		want float32
	}{
		{name: "abs", arg: -2.25, want: 2.25},
		{name: "neg", arg: 2.25, want: -2.25},
		{name: "sqrt", arg: 9, want: 3},
		{name: "ceil", arg: 2.25, want: 3},
		{name: "floor", arg: 2.75, want: 2},
		{name: "trunc", arg: -2.75, want: -2},
		{name: "nearest", arg: 2.5, want: 2},
		{name: "nearest", arg: 3.5, want: 4},
	} {
		results := callExport(t, inst, tt.name, wasmvm.F32(tt.arg))
		if len(results) != 1 || results[0] != wasmvm.F32(tt.want) {
			t.Fatalf("%s(%v) got results %#v, want f32 %v", tt.name, tt.arg, results, tt.want)
		}
	}
}

func TestF64ArithmeticAndPredicates(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "calc") (param f64) (result f64)
				local.get 0
				f64.const 8.0
				f64.add
				f64.const 2.0
				f64.div)
			(func (export "cmp") (param f64 f64) (result i32)
				local.get 0
				local.get 1
				f64.ge))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "calc", wasmvm.F64(6))
	if len(results) != 1 || results[0] != wasmvm.F64(7) {
		t.Fatalf("calc got results %#v, want f64 7", results)
	}

	results = callExport(t, inst, "cmp", wasmvm.F64(3.5), wasmvm.F64(3.5))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("cmp got results %#v, want i32 1", results)
	}
}

// TestF64UnaryOps checks non-converting f64 unary instructions, including
// nearest's ties-to-even behavior.
func TestF64UnaryOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "abs") (param f64) (result f64) local.get 0 f64.abs)
			(func (export "neg") (param f64) (result f64) local.get 0 f64.neg)
			(func (export "sqrt") (param f64) (result f64) local.get 0 f64.sqrt)
			(func (export "ceil") (param f64) (result f64) local.get 0 f64.ceil)
			(func (export "floor") (param f64) (result f64) local.get 0 f64.floor)
			(func (export "trunc") (param f64) (result f64) local.get 0 f64.trunc)
			(func (export "nearest") (param f64) (result f64) local.get 0 f64.nearest))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		arg  float64
		want float64
	}{
		{name: "abs", arg: -2.25, want: 2.25},
		{name: "neg", arg: 2.25, want: -2.25},
		{name: "sqrt", arg: 9, want: 3},
		{name: "ceil", arg: 2.25, want: 3},
		{name: "floor", arg: 2.75, want: 2},
		{name: "trunc", arg: -2.75, want: -2},
		{name: "nearest", arg: 2.5, want: 2},
		{name: "nearest", arg: 3.5, want: 4},
	} {
		results := callExport(t, inst, tt.name, wasmvm.F64(tt.arg))
		if len(results) != 1 || results[0] != wasmvm.F64(tt.want) {
			t.Fatalf("%s(%v) got results %#v, want f64 %v", tt.name, tt.arg, results, tt.want)
		}
	}
}

// TestFloatBinaryExtraOps checks min, max, and copysign for f32 and f64.
func TestFloatBinaryExtraOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "f32_min") (param f32 f32) (result f32) local.get 0 local.get 1 f32.min)
			(func (export "f32_max") (param f32 f32) (result f32) local.get 0 local.get 1 f32.max)
			(func (export "f32_copysign") (param f32 f32) (result f32) local.get 0 local.get 1 f32.copysign)
			(func (export "f32_sub_inf") (result f32) f32.const inf f32.const inf f32.sub)
			(func (export "f64_min") (param f64 f64) (result f64) local.get 0 local.get 1 f64.min)
			(func (export "f64_max") (param f64 f64) (result f64) local.get 0 local.get 1 f64.max)
			(func (export "f64_copysign") (param f64 f64) (result f64) local.get 0 local.get 1 f64.copysign)
			(func (export "f64_div_inf") (result f64) f64.const inf f64.const inf f64.div))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		lhs  wasmvm.Value
		rhs  wasmvm.Value
		want wasmvm.Value
	}{
		{name: "f32_min", lhs: wasmvm.F32(3.5), rhs: wasmvm.F32(-1.25), want: wasmvm.F32(-1.25)},
		{name: "f32_max", lhs: wasmvm.F32(3.5), rhs: wasmvm.F32(-1.25), want: wasmvm.F32(3.5)},
		{name: "f32_copysign", lhs: wasmvm.F32(3.5), rhs: wasmvm.F32(float32(math.Copysign(0, -1))), want: wasmvm.F32(-3.5)},
		{name: "f64_min", lhs: wasmvm.F64(3.5), rhs: wasmvm.F64(-1.25), want: wasmvm.F64(-1.25)},
		{name: "f64_max", lhs: wasmvm.F64(3.5), rhs: wasmvm.F64(-1.25), want: wasmvm.F64(3.5)},
		{name: "f64_copysign", lhs: wasmvm.F64(3.5), rhs: wasmvm.F64(math.Copysign(0, -1)), want: wasmvm.F64(-3.5)},
	} {
		results := callExport(t, inst, tt.name, tt.lhs, tt.rhs)
		if len(results) != 1 || results[0] != tt.want {
			t.Fatalf("%s got results %#v, want %v", tt.name, results, tt.want)
		}
	}

	const (
		canonicalF32NaNBits = 0x7fc00000
		canonicalF64NaNBits = 0x7ff8000000000000
	)
	for _, tt := range []struct {
		name string
		lhs  wasmvm.Value
		rhs  wasmvm.Value
		want uint64
	}{
		{name: "f32_min", lhs: wasmvm.F32(float32(math.Inf(-1))), rhs: wasmvm.F32(math.Float32frombits(canonicalF32NaNBits)), want: canonicalF32NaNBits},
		{name: "f32_max", lhs: wasmvm.F32(float32(math.Inf(1))), rhs: wasmvm.F32(math.Float32frombits(canonicalF32NaNBits)), want: canonicalF32NaNBits},
		{name: "f64_min", lhs: wasmvm.F64(math.Inf(-1)), rhs: wasmvm.F64(math.Float64frombits(canonicalF64NaNBits)), want: canonicalF64NaNBits},
		{name: "f64_max", lhs: wasmvm.F64(math.Inf(1)), rhs: wasmvm.F64(math.Float64frombits(canonicalF64NaNBits)), want: canonicalF64NaNBits},
	} {
		results := callExport(t, inst, tt.name, tt.lhs, tt.rhs)
		if len(results) != 1 {
			t.Fatalf("%s got results %#v, want one result", tt.name, results)
		}
		var got uint64
		if results[0].Type == wasmir.ValueTypeF32 {
			got = uint64(math.Float32bits(results[0].F32))
		} else {
			got = math.Float64bits(results[0].F64)
		}
		if got != tt.want {
			t.Fatalf("%s NaN bits = %#x, want %#x", tt.name, got, tt.want)
		}
	}

	results := callExport(t, inst, "f32_sub_inf")
	if len(results) != 1 || math.Float32bits(results[0].F32) != canonicalF32NaNBits {
		t.Fatalf("f32_sub_inf got results %#v, want canonical NaN", results)
	}
	results = callExport(t, inst, "f64_div_inf")
	if len(results) != 1 || math.Float64bits(results[0].F64) != canonicalF64NaNBits {
		t.Fatalf("f64_div_inf got results %#v, want canonical NaN", results)
	}
}

// TestNumericConversionsAndReinterpret checks non-trapping numeric conversions
// and bit-preserving reinterpret instructions.
func TestNumericConversionsAndReinterpret(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i32_wrap") (param i64) (result i32) local.get 0 i32.wrap_i64)
			(func (export "i64_ext_s") (param i32) (result i64) local.get 0 i64.extend_i32_s)
			(func (export "i64_ext_u") (param i32) (result i64) local.get 0 i64.extend_i32_u)
			(func (export "f32_i32_s") (param i32) (result f32) local.get 0 f32.convert_i32_s)
			(func (export "f32_i32_u") (param i32) (result f32) local.get 0 f32.convert_i32_u)
			(func (export "f32_i64_s") (param i64) (result f32) local.get 0 f32.convert_i64_s)
			(func (export "f32_i64_u") (param i64) (result f32) local.get 0 f32.convert_i64_u)
			(func (export "f32_demote") (param f64) (result f32) local.get 0 f32.demote_f64)
			(func (export "f64_i32_s") (param i32) (result f64) local.get 0 f64.convert_i32_s)
			(func (export "f64_i32_u") (param i32) (result f64) local.get 0 f64.convert_i32_u)
			(func (export "f64_i64_s") (param i64) (result f64) local.get 0 f64.convert_i64_s)
			(func (export "f64_i64_u") (param i64) (result f64) local.get 0 f64.convert_i64_u)
			(func (export "f64_promote") (param f32) (result f64) local.get 0 f64.promote_f32)
			(func (export "i32_re_f32") (param f32) (result i32) local.get 0 i32.reinterpret_f32)
			(func (export "i64_re_f64") (param f64) (result i64) local.get 0 i64.reinterpret_f64)
			(func (export "f32_re_i32") (param i32) (result f32) local.get 0 f32.reinterpret_i32)
			(func (export "f64_re_i64") (param i64) (result f64) local.get 0 f64.reinterpret_i64))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		arg  wasmvm.Value
		want wasmvm.Value
	}{
		{name: "i32_wrap", arg: wasmvm.I64(0x100000002), want: wasmvm.I32(2)},
		{name: "i64_ext_s", arg: wasmvm.I32(-1), want: wasmvm.I64(-1)},
		{name: "i64_ext_u", arg: wasmvm.I32(-1), want: wasmvm.I64(4294967295)},
		{name: "f32_i32_s", arg: wasmvm.I32(-7), want: wasmvm.F32(-7)},
		{name: "f32_i32_u", arg: wasmvm.I32(-1), want: wasmvm.F32(float32(^uint32(0)))},
		{name: "f32_i64_s", arg: wasmvm.I64(-9), want: wasmvm.F32(-9)},
		{name: "f32_i64_u", arg: wasmvm.I64(9), want: wasmvm.F32(9)},
		{name: "f32_demote", arg: wasmvm.F64(12.5), want: wasmvm.F32(12.5)},
		{name: "f64_i32_s", arg: wasmvm.I32(-11), want: wasmvm.F64(-11)},
		{name: "f64_i32_u", arg: wasmvm.I32(-1), want: wasmvm.F64(float64(^uint32(0)))},
		{name: "f64_i64_s", arg: wasmvm.I64(-13), want: wasmvm.F64(-13)},
		{name: "f64_i64_u", arg: wasmvm.I64(13), want: wasmvm.F64(13)},
		{name: "f64_promote", arg: wasmvm.F32(6.25), want: wasmvm.F64(6.25)},
		{name: "i32_re_f32", arg: wasmvm.F32(1), want: wasmvm.I32(0x3f800000)},
		{name: "i64_re_f64", arg: wasmvm.F64(1), want: wasmvm.I64(0x3ff0000000000000)},
		{name: "f32_re_i32", arg: wasmvm.I32(0x3f800000), want: wasmvm.F32(1)},
		{name: "f64_re_i64", arg: wasmvm.I64(0x3ff0000000000000), want: wasmvm.F64(1)},
	} {
		results := callExport(t, inst, tt.name, tt.arg)
		if len(results) != 1 || results[0] != tt.want {
			t.Fatalf("%s got results %#v, want %v", tt.name, results, tt.want)
		}
	}
}

// TestFloatToIntegerTruncation checks trapping and saturating float-to-integer
// conversions for representative f32 and f64 inputs.
func TestFloatToIntegerTruncation(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i32_f32_s") (param f32) (result i32) local.get 0 i32.trunc_f32_s)
			(func (export "i32_f32_u") (param f32) (result i32) local.get 0 i32.trunc_f32_u)
			(func (export "i32_f64_s") (param f64) (result i32) local.get 0 i32.trunc_f64_s)
			(func (export "i32_f64_u") (param f64) (result i32) local.get 0 i32.trunc_f64_u)
			(func (export "i64_f32_s") (param f32) (result i64) local.get 0 i64.trunc_f32_s)
			(func (export "i64_f32_u") (param f32) (result i64) local.get 0 i64.trunc_f32_u)
			(func (export "i64_f64_s") (param f64) (result i64) local.get 0 i64.trunc_f64_s)
			(func (export "i64_f64_u") (param f64) (result i64) local.get 0 i64.trunc_f64_u)
			(func (export "i32_sat_f32_s") (param f32) (result i32) local.get 0 i32.trunc_sat_f32_s)
			(func (export "i32_sat_f32_u") (param f32) (result i32) local.get 0 i32.trunc_sat_f32_u)
			(func (export "i32_sat_f64_s") (param f64) (result i32) local.get 0 i32.trunc_sat_f64_s)
			(func (export "i32_sat_f64_u") (param f64) (result i32) local.get 0 i32.trunc_sat_f64_u)
			(func (export "i64_sat_f32_s") (param f32) (result i64) local.get 0 i64.trunc_sat_f32_s)
			(func (export "i64_sat_f32_u") (param f32) (result i64) local.get 0 i64.trunc_sat_f32_u)
			(func (export "i64_sat_f64_s") (param f64) (result i64) local.get 0 i64.trunc_sat_f64_s)
			(func (export "i64_sat_f64_u") (param f64) (result i64) local.get 0 i64.trunc_sat_f64_u))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		arg  wasmvm.Value
		want wasmvm.Value
	}{
		{name: "i32_f32_s", arg: wasmvm.F32(-2.9), want: wasmvm.I32(-2)},
		{name: "i32_f32_u", arg: wasmvm.F32(3.9), want: wasmvm.I32(3)},
		{name: "i32_f64_s", arg: wasmvm.F64(-2.9), want: wasmvm.I32(-2)},
		{name: "i32_f64_u", arg: wasmvm.F64(3.9), want: wasmvm.I32(3)},
		{name: "i64_f32_s", arg: wasmvm.F32(-2.9), want: wasmvm.I64(-2)},
		{name: "i64_f32_u", arg: wasmvm.F32(3.9), want: wasmvm.I64(3)},
		{name: "i64_f64_s", arg: wasmvm.F64(-2.9), want: wasmvm.I64(-2)},
		{name: "i64_f64_u", arg: wasmvm.F64(3.9), want: wasmvm.I64(3)},
		{name: "i32_sat_f32_s", arg: wasmvm.F32(float32(math.Inf(1))), want: wasmvm.I32(1<<31 - 1)},
		{name: "i32_sat_f32_u", arg: wasmvm.F32(-1), want: wasmvm.I32(0)},
		{name: "i32_sat_f64_s", arg: wasmvm.F64(math.Inf(-1)), want: wasmvm.I32(-1 << 31)},
		{name: "i32_sat_f64_u", arg: wasmvm.F64(math.Inf(1)), want: wasmvm.I32(-1)},
		{name: "i64_sat_f32_s", arg: wasmvm.F32(float32(math.Inf(1))), want: wasmvm.I64(1<<63 - 1)},
		{name: "i64_sat_f32_u", arg: wasmvm.F32(float32(math.Inf(1))), want: wasmvm.I64(-1)},
		{name: "i64_sat_f64_s", arg: wasmvm.F64(math.Inf(-1)), want: wasmvm.I64(-1 << 63)},
		{name: "i64_sat_f64_u", arg: wasmvm.F64(math.NaN()), want: wasmvm.I64(0)},
	} {
		results := callExport(t, inst, tt.name, tt.arg)
		if len(results) != 1 || results[0] != tt.want {
			t.Fatalf("%s got results %#v, want %v", tt.name, results, tt.want)
		}
	}

	for _, tt := range []struct {
		name string
		arg  wasmvm.Value
		want string
	}{
		{name: "i32_f64_s", arg: wasmvm.F64(2147483648), want: "pc 1 i32.trunc_f64_s: integer overflow"},
		{name: "i32_f64_u", arg: wasmvm.F64(-1), want: "pc 1 i32.trunc_f64_u: integer overflow"},
		{name: "i64_f64_s", arg: wasmvm.F64(math.Inf(1)), want: "pc 1 i64.trunc_f64_s: integer overflow"},
		{name: "i32_f64_s", arg: wasmvm.F64(math.NaN()), want: "pc 1 i32.trunc_f64_s: invalid conversion to integer"},
	} {
		f, ok := inst.ExportedFunc(tt.name)
		if !ok {
			t.Fatalf("missing %s export", tt.name)
		}
		_, err := f.Call(tt.arg)
		if err == nil {
			t.Fatalf("Call %s succeeded unexpectedly", tt.name)
		}
		if got := err.Error(); got != tt.want {
			t.Fatalf("%s error = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestDropAndReturn(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "early") (param i32) (result i32)
				local.get 0
				i32.eqz
				if
					i32.const 42
					return
				end
				i32.const 100
				drop
				local.get 0))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "early", wasmvm.I32(0))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("early(0) got results %#v, want i32 42", results)
	}
	results = callExport(t, inst, "early", wasmvm.I32(9))
	if len(results) != 1 || results[0] != wasmvm.I32(9) {
		t.Fatalf("early(9) got results %#v, want i32 9", results)
	}
}

func TestIfElse(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "abs") (param i32) (result i32)
				local.get 0
				i32.const 0
				i32.lt_s
				if (result i32)
					i32.const 0
					local.get 0
					i32.sub
				else
					local.get 0
				end))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "abs", wasmvm.I32(-7))
	if len(results) != 1 || results[0] != wasmvm.I32(7) {
		t.Fatalf("abs(-7) got results %#v, want i32 7", results)
	}
	results = callExport(t, inst, "abs", wasmvm.I32(5))
	if len(results) != 1 || results[0] != wasmvm.I32(5) {
		t.Fatalf("abs(5) got results %#v, want i32 5", results)
	}
}

func TestBlockBranch(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "skip") (result i32)
				block (result i32)
					i32.const 99
					br 0
					i32.const 10
				end))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "skip")
	if len(results) != 1 || results[0] != wasmvm.I32(99) {
		t.Fatalf("skip got results %#v, want i32 99", results)
	}
}

func TestBlockBranchIf(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "clamp_zero") (param i32) (result i32)
				block (result i32)
					local.get 0
					local.get 0
					i32.const 0
					i32.ge_s
					br_if 0
					drop
					i32.const 0
				end))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "clamp_zero", wasmvm.I32(12))
	if len(results) != 1 || results[0] != wasmvm.I32(12) {
		t.Fatalf("clamp_zero(12) got results %#v, want i32 12", results)
	}
	results = callExport(t, inst, "clamp_zero", wasmvm.I32(-3))
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("clamp_zero(-3) got results %#v, want i32 0", results)
	}
}

// TestBlockBranchTable checks that br_table selects table targets by an i32
// selector and falls back to the default target for out-of-range selectors.
func TestBlockBranchTable(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "choose") (param i32) (result i32)
				block $default
					block $one
						block $zero
							local.get 0
							br_table $zero $one $default
						end
						i32.const 0
						return
					end
					i32.const 1
					return
				end
				i32.const 9))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	tests := []struct {
		arg  wasmvm.Value
		want wasmvm.Value
	}{
		{wasmvm.I32(0), wasmvm.I32(0)},
		{wasmvm.I32(1), wasmvm.I32(1)},
		{wasmvm.I32(2), wasmvm.I32(9)},
		{wasmvm.I32(-1), wasmvm.I32(9)},
	}
	for _, tt := range tests {
		results := callExport(t, inst, "choose", tt.arg)
		if len(results) != 1 || results[0] != tt.want {
			t.Fatalf("choose(%v) got results %#v, want %v", tt.arg, results, tt.want)
		}
	}
}

// TestLoopBranch checks that br to a loop label jumps back to the loop body,
// while br_if to an outer block exits the loop.
func TestLoopBranch(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "sum") (param $n i32) (result i32)
				(local $acc i32)
				block $exit
					loop $again
						local.get $n
						i32.eqz
						br_if $exit
						local.get $acc
						local.get $n
						i32.add
						local.set $acc
						local.get $n
						i32.const 1
						i32.sub
						local.set $n
						br $again
					end
				end
				local.get $acc))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	tests := []struct {
		arg  wasmvm.Value
		want wasmvm.Value
	}{
		{wasmvm.I32(0), wasmvm.I32(0)},
		{wasmvm.I32(1), wasmvm.I32(1)},
		{wasmvm.I32(5), wasmvm.I32(15)},
	}
	for _, tt := range tests {
		results := callExport(t, inst, "sum", tt.arg)
		if len(results) != 1 || results[0] != tt.want {
			t.Fatalf("sum(%v) got results %#v, want %v", tt.arg, results, tt.want)
		}
	}
}

func TestModuleGlobals(t *testing.T) {
	// Module-defined globals are instantiated once and then accessed through
	// global.get/global.set while functions execute.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(global $g (mut i32) (i32.const 7))
			(global $h i64 (i64.const 11))
			(func (export "get_g") (result i32)
				global.get $g)
			(func (export "set_g") (param i32)
				local.get 0
				global.set $g)
			(func (export "get_h") (result i64)
				global.get $h))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "get_g")
	if len(results) != 1 || results[0] != wasmvm.I32(7) {
		t.Fatalf("get_g got results %#v, want i32 7", results)
	}

	callExport(t, inst, "set_g", wasmvm.I32(42))
	results = callExport(t, inst, "get_g")
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("get_g after set got results %#v, want i32 42", results)
	}

	results = callExport(t, inst, "get_h")
	if len(results) != 1 || results[0] != wasmvm.I64(11) {
		t.Fatalf("get_h got results %#v, want i64 11", results)
	}
}

func TestExportedGlobalValue(t *testing.T) {
	// ExportedGlobal exposes the same global state that executing code observes
	// through global.get/global.set.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(global $g (export "g") (mut i32) (i32.const 7))
			(func (export "set_g") (param i32)
				local.get 0
				global.set $g))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	g, ok := inst.ExportedGlobal("g")
	if !ok {
		t.Fatal("missing g export")
	}
	if got, err := g.Value(); err != nil || got != wasmvm.I32(7) {
		t.Fatalf("g.Value() = %#v, %v; want i32 7, nil", got, err)
	}

	callExport(t, inst, "set_g", wasmvm.I32(42))
	if got, err := g.Value(); err != nil || got != wasmvm.I32(42) {
		t.Fatalf("g.Value() after set = %#v, %v; want i32 42, nil", got, err)
	}

	if _, ok := inst.ExportedGlobal("missing"); ok {
		t.Fatal("ExportedGlobal reported missing global as present")
	}
}

// TestGlobalImportSharesExportedGlobal checks that an exported global can be
// imported into another instance and remains shared.
func TestGlobalImportSharesExportedGlobal(t *testing.T) {
	rt := wasmvm.NewRuntime()
	exporter, err := rt.Instantiate(parseWAT(t, `
		(module
			(global $g (export "g") (mut i32) (i32.const 7))
			(func (export "set") (param i32)
				local.get 0
				global.set $g))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate exporter failed: %v", err)
	}
	global, ok := exporter.ExportedGlobal("g")
	if !ok {
		t.Fatal("missing g export")
	}

	importer, err := rt.Instantiate(parseWAT(t, `
		(module
			(import "env" "g" (global $g (mut i32)))
			(func (export "get") (result i32)
				global.get $g)
			(func (export "set") (param i32)
				local.get 0
				global.set $g))
	`), wasmvm.Imports{"env": {"g": global}})
	if err != nil {
		t.Fatalf("Instantiate importer failed: %v", err)
	}

	results := callExport(t, importer, "get")
	if len(results) != 1 || results[0] != wasmvm.I32(7) {
		t.Fatalf("importer get got results %#v, want i32 7", results)
	}
	callExport(t, exporter, "set", wasmvm.I32(11))
	results = callExport(t, importer, "get")
	if len(results) != 1 || results[0] != wasmvm.I32(11) {
		t.Fatalf("importer get after exporter set got results %#v, want i32 11", results)
	}
	callExport(t, importer, "set", wasmvm.I32(23))
	if got, err := global.Value(); err != nil || got != wasmvm.I32(23) {
		t.Fatalf("exported global after importer set = %#v, %v; want i32 23, nil", got, err)
	}
}

func TestGlobalInitializerReadsEarlierImmutableGlobal(t *testing.T) {
	// Global initializer expressions can read earlier immutable globals and use
	// the numeric constant-expression operators currently supported by wasmvm.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(global $base i32 (i32.const 5))
			(global $sum i32
				global.get $base
				i32.const 6
				i32.add)
			(global $scale i64
				i64.const 3
				i64.const 4
				i64.mul)
			(func (export "sum") (result i32)
				global.get $sum)
			(func (export "scale") (result i64)
				global.get $scale))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "sum")
	if len(results) != 1 || results[0] != wasmvm.I32(11) {
		t.Fatalf("sum got results %#v, want i32 11", results)
	}

	results = callExport(t, inst, "scale")
	if len(results) != 1 || results[0] != wasmvm.I64(12) {
		t.Fatalf("scale got results %#v, want i64 12", results)
	}
}

func TestMemoryI32LoadStore(t *testing.T) {
	// Module-defined memories are instantiated as zeroed bytes and accessed
	// through i32.load/i32.store with the static offset immediate applied.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(func (export "roundtrip") (param i32 i32) (result i32)
				local.get 0
				local.get 1
				i32.store offset=4
				local.get 0
				i32.load offset=4)
			(func (export "zero") (result i32)
				i32.const 32
				i32.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "zero")
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("zero got results %#v, want i32 0", results)
	}

	results = callExport(t, inst, "roundtrip", wasmvm.I32(12), wasmvm.I32(0x12345678))
	if len(results) != 1 || results[0] != wasmvm.I32(0x12345678) {
		t.Fatalf("roundtrip got results %#v, want i32 0x12345678", results)
	}
}

func TestActiveDataSegments(t *testing.T) {
	// Active data segments are copied into memory during instantiation, and
	// offset expressions may read immutable globals.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(global $off i32 (i32.const 16))
			(data (i32.const 4) "ABCD")
			(data (global.get $off) "WXYZ")
			(func (export "load0") (result i32)
				i32.const 4
				i32.load)
			(func (export "load1") (result i32)
				i32.const 16
				i32.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "load0")
	if len(results) != 1 || results[0] != wasmvm.I32(0x44434241) {
		t.Fatalf("load0 got results %#v, want i32 0x44434241", results)
	}

	results = callExport(t, inst, "load1")
	if len(results) != 1 || results[0] != wasmvm.I32(0x5a595857) {
		t.Fatalf("load1 got results %#v, want i32 0x5a595857", results)
	}
}

func TestI32NarrowMemoryOps(t *testing.T) {
	// Narrow i32 loads extend to i32, and narrow stores truncate the low-order
	// bytes of the stored value.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data (i32.const 0) "\ff\80\34\12")
			(func (export "load8_s") (result i32)
				i32.const 0
				i32.load8_s)
			(func (export "load8_u") (result i32)
				i32.const 0
				i32.load8_u)
			(func (export "load16_s") (result i32)
				i32.const 0
				i32.load16_s)
			(func (export "load16_u") (result i32)
				i32.const 2
				i32.load16_u)
			(func (export "store8") (result i32)
				i32.const 8
				i32.const 0x12345678
				i32.store8
				i32.const 8
				i32.load8_u)
			(func (export "store16") (result i32)
				i32.const 10
				i32.const 0x12345678
				i32.store16
				i32.const 10
				i32.load16_u))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		want int32
	}{
		{name: "load8_s", want: -1},
		{name: "load8_u", want: 255},
		{name: "load16_s", want: -32513},
		{name: "load16_u", want: 0x1234},
		{name: "store8", want: 0x78},
		{name: "store16", want: 0x5678},
	} {
		results := callExport(t, inst, tt.name)
		if len(results) != 1 || results[0] != wasmvm.I32(tt.want) {
			t.Fatalf("%s got results %#v, want i32 %d", tt.name, results, tt.want)
		}
	}
}

func TestScalarMemoryOps(t *testing.T) {
	// The remaining scalar numeric load/store instructions share the same
	// memory resolver path with their own value encodings.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data (i32.const 0) "\01\02\03\04\05\06\07\08")
			(func (export "load_i64") (result i64)
				i32.const 0
				i64.load)
			(func (export "roundtrip_i64") (param i64) (result i64)
				i32.const 16
				local.get 0
				i64.store
				i32.const 16
				i64.load)
			(func (export "roundtrip_f32") (param f32) (result f32)
				i32.const 32
				local.get 0
				f32.store
				i32.const 32
				f32.load)
			(func (export "roundtrip_f64") (param f64) (result f64)
				i32.const 48
				local.get 0
				f64.store
				i32.const 48
				f64.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "load_i64")
	if len(results) != 1 || results[0] != wasmvm.I64(0x0807060504030201) {
		t.Fatalf("load_i64 got results %#v, want i64 0x0807060504030201", results)
	}

	results = callExport(t, inst, "roundtrip_i64", wasmvm.I64(-1234567890123))
	if len(results) != 1 || results[0] != wasmvm.I64(-1234567890123) {
		t.Fatalf("roundtrip_i64 got results %#v, want i64 -1234567890123", results)
	}

	results = callExport(t, inst, "roundtrip_f32", wasmvm.F32(12.5))
	if len(results) != 1 || results[0] != wasmvm.F32(12.5) {
		t.Fatalf("roundtrip_f32 got results %#v, want f32 12.5", results)
	}

	results = callExport(t, inst, "roundtrip_f64", wasmvm.F64(-9.25))
	if len(results) != 1 || results[0] != wasmvm.F64(-9.25) {
		t.Fatalf("roundtrip_f64 got results %#v, want f64 -9.25", results)
	}
}

// TestV128Const checks that the VM can return a v128.const value.
func TestV128Const(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "const") (result v128)
				v128.const i32x4 0x11223344 0x55667788 0x99aabbcc 0xddeeff00))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "const")
	want := [16]byte{
		0x44, 0x33, 0x22, 0x11,
		0x88, 0x77, 0x66, 0x55,
		0xcc, 0xbb, 0xaa, 0x99,
		0x00, 0xff, 0xee, 0xdd,
	}
	if len(results) != 1 || results[0].Type != wasmir.ValueTypeV128 || results[0].V128 != want {
		t.Fatalf("const got results %#v, want v128 %#v", results, want)
	}
}

// TestV128Splat checks scalar-to-vector SIMD splat instructions.
func TestV128Splat(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i8_splat") (result v128)
				i32.const 0x12345678
				i8x16.splat)
			(func (export "i16_splat") (result v128)
				i32.const 0x12345678
				i16x8.splat)
			(func (export "i32_splat") (result v128)
				i32.const 0x12345678
				i32x4.splat)
			(func (export "i64_splat") (result v128)
				i64.const 0x1234567801020304
				i64x2.splat)
			(func (export "f32_splat") (result v128)
				f32.const 1.0
				f32x4.splat)
			(func (export "f64_splat") (result v128)
				f64.const 1.0
				f64x2.splat))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	checkV128Export(t, inst, "i8_splat", [16]byte{
		0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78,
		0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78,
	})
	checkV128Export(t, inst, "i16_splat", [16]byte{
		0x78, 0x56, 0x78, 0x56, 0x78, 0x56, 0x78, 0x56,
		0x78, 0x56, 0x78, 0x56, 0x78, 0x56, 0x78, 0x56,
	})
	checkV128Export(t, inst, "i32_splat", [16]byte{
		0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12,
		0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12,
	})
	checkV128Export(t, inst, "i64_splat", [16]byte{
		0x04, 0x03, 0x02, 0x01, 0x78, 0x56, 0x34, 0x12,
		0x04, 0x03, 0x02, 0x01, 0x78, 0x56, 0x34, 0x12,
	})
	checkV128Export(t, inst, "f32_splat", [16]byte{
		0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f,
		0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f,
	})
	checkV128Export(t, inst, "f64_splat", [16]byte{
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
	})
}

// TestV128MemoryOps checks v128 load/store and load-splat instructions.
func TestV128MemoryOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data (i32.const 0) "\ff\ff\ff\ff")
			(data (i32.const 4) "\00\00\ce\41")
			(data (i32.const 8) "\00\00\00\00\00\ff\8f\40")
			(data (i32.const 16) "\ff\ff\ff\ff\ff\ff\ff\ff")
			(func (export "load") (result v128)
				i32.const 4
				v128.load)
			(func (export "load8_splat") (result v128)
				i32.const 6
				v128.load8_splat)
			(func (export "load16_splat") (result v128)
				i32.const 6
				v128.load16_splat)
			(func (export "load32_splat") (result v128)
				i32.const 4
				v128.load32_splat)
			(func (export "load64_splat") (result v128)
				i32.const 0
				v128.load64_splat)
			(func (export "store") (result v128)
				i32.const 4
				v128.const i32x4 0x11223344 0x55667788 0x99aabbcc 0xddeeff00
				v128.store
				i32.const 4
				v128.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	checkV128Export(t, inst, "load", [16]byte{
		0x00, 0x00, 0xce, 0x41, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xff, 0x8f, 0x40, 0xff, 0xff, 0xff, 0xff,
	})
	checkV128Export(t, inst, "load8_splat", [16]byte{
		0xce, 0xce, 0xce, 0xce, 0xce, 0xce, 0xce, 0xce,
		0xce, 0xce, 0xce, 0xce, 0xce, 0xce, 0xce, 0xce,
	})
	checkV128Export(t, inst, "load16_splat", [16]byte{
		0xce, 0x41, 0xce, 0x41, 0xce, 0x41, 0xce, 0x41,
		0xce, 0x41, 0xce, 0x41, 0xce, 0x41, 0xce, 0x41,
	})
	checkV128Export(t, inst, "load32_splat", [16]byte{
		0x00, 0x00, 0xce, 0x41, 0x00, 0x00, 0xce, 0x41,
		0x00, 0x00, 0xce, 0x41, 0x00, 0x00, 0xce, 0x41,
	})
	checkV128Export(t, inst, "load64_splat", [16]byte{
		0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0xce, 0x41,
		0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0xce, 0x41,
	})
	checkV128Export(t, inst, "store", [16]byte{
		0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55,
		0xcc, 0xbb, 0xaa, 0x99, 0x00, 0xff, 0xee, 0xdd,
	})
}

// checkV128Export calls name and verifies that it returns the expected v128.
func checkV128Export(t *testing.T, inst *wasmvm.ModuleInstance, name string, want [16]byte) {
	t.Helper()

	results := callExport(t, inst, name)
	if len(results) != 1 || results[0].Type != wasmir.ValueTypeV128 || results[0].V128 != want {
		t.Fatalf("%s got results %#v, want v128 %#v", name, results, want)
	}
}

// TestV128LaneOps checks SIMD lane extract, replace, and shuffle instructions.
func TestV128LaneOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i8_extract_s") (result i32)
				v128.const i32x4 0x00000001 0x0000000f 0x000000ff 0x0000017f
				i8x16.extract_lane_s 8)
			(func (export "i8_extract_u") (result i32)
				v128.const i32x4 0x00000001 0x0000000f 0x000000ff 0x0000017f
				i8x16.extract_lane_u 8)
			(func (export "i16_extract_s") (result i32)
				v128.const i32x4 0x00000001 0x0000000f 0x0000ffff 0x0000017f
				i16x8.extract_lane_s 4)
			(func (export "i16_extract_u") (result i32)
				v128.const i32x4 0x00000001 0x0000000f 0x0000ffff 0x0000017f
				i16x8.extract_lane_u 4)
			(func (export "i32_extract") (result i32)
				v128.const i32x4 0x00000001 0x0000000f 0x0000ffff 0x0000017f
				i32x4.extract_lane 2)
			(func (export "i64_extract") (result i64)
				v128.const i32x4 0x0000000f 0x00000000 0x0000ffff 0x0000017f
				i64x2.extract_lane 0)
			(func (export "f32_extract") (result f32)
				v128.const i32x4 0x00000001 0x3fc00000 0x0000ffff 0x0000017f
				f32x4.extract_lane 1)
			(func (export "f64_extract") (result f64)
				v128.const i32x4 0x00000000 0x40120000 0x0000ffff 0x0000017f
				f64x2.extract_lane 0)
			(func (export "i8_replace") (result v128)
				v128.const i32x4 0x00000001 0x0000000f 0x000000ff 0x0000017f
				i32.const 0xe5
				i8x16.replace_lane 8)
			(func (export "i16_replace") (result v128)
				v128.const i32x4 0x00000001 0x0000000f 0x0000ffff 0x0000017f
				i32.const 0xe5e6
				i16x8.replace_lane 4)
			(func (export "i32_replace") (result v128)
				v128.const i32x4 0x00000001 0x0000000f 0x0000ffff 0x0000017f
				i32.const 0x12345678
				i32x4.replace_lane 2)
			(func (export "i64_replace") (result v128)
				v128.const i32x4 0x0000000f 0x00000000 0x0000ffff 0x0000017f
				i64.const 0x0000123400005678
				i64x2.replace_lane 0)
			(func (export "f32_replace") (result v128)
				v128.const i32x4 0x00000001 0x00000000 0x0000ffff 0x0000017f
				f32.const 1.5
				f32x4.replace_lane 1)
			(func (export "f64_replace") (result v128)
				v128.const i32x4 0x0000789a 0xff880330 0x0000ffff 0x0000017f
				f64.const 4.5
				f64x2.replace_lane 0)
			(func (export "shuffle") (result v128)
				v128.const i32x4 0xff00ff01 0xff00ff0f 0xff00ffff 0xff00ff7f
				v128.const i32x4 0x00550055 0x00550055 0x00550055 0x00550155
				i8x16.shuffle 16 1 18 3 20 5 22 7 24 9 26 11 28 13 30 15))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	scalarChecks := []struct {
		name string
		want wasmvm.Value
	}{
		{"i8_extract_s", wasmvm.I32(-1)},
		{"i8_extract_u", wasmvm.I32(255)},
		{"i16_extract_s", wasmvm.I32(-1)},
		{"i16_extract_u", wasmvm.I32(65535)},
		{"i32_extract", wasmvm.I32(65535)},
		{"i64_extract", wasmvm.I64(15)},
		{"f32_extract", wasmvm.F32(1.5)},
		{"f64_extract", wasmvm.F64(4.5)},
	}
	for _, check := range scalarChecks {
		results := callExport(t, inst, check.name)
		if len(results) != 1 || results[0] != check.want {
			t.Fatalf("%s got results %#v, want %#v", check.name, results, check.want)
		}
	}

	checkV128Export(t, inst, "i8_replace", v128I32x4(0x00000001, 0x0000000f, 0x000000e5, 0x0000017f))
	checkV128Export(t, inst, "i16_replace", v128I32x4(0x00000001, 0x0000000f, 0x0000e5e6, 0x0000017f))
	checkV128Export(t, inst, "i32_replace", v128I32x4(0x00000001, 0x0000000f, 0x12345678, 0x0000017f))
	checkV128Export(t, inst, "i64_replace", v128I32x4(0x00005678, 0x00001234, 0x0000ffff, 0x0000017f))
	checkV128Export(t, inst, "f32_replace", v128I32x4(0x00000001, math.Float32bits(1.5), 0x0000ffff, 0x0000017f))
	checkV128Export(t, inst, "f64_replace", v128I32x4(0x00000000, 0x40120000, 0x0000ffff, 0x0000017f))
	checkV128Export(t, inst, "shuffle", v128I32x4(0xff55ff55, 0xff55ff55, 0xff55ff55, 0xff55ff55))
}

// TestV128ShiftAndBitselect checks integer SIMD shifts and v128.bitselect.
func TestV128ShiftAndBitselect(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i8_shl") (result v128)
				v128.const i32x4 0xff000001 0xe0000002 0x00000003 0x00000004
				i32.const 11
				i8x16.shl)
			(func (export "i8_shr_s") (result v128)
				v128.const i32x4 0xff00000f 0xe00f7002 0x0f000003 0x000ff004
				i32.const 11
				i8x16.shr_s)
			(func (export "i8_shr_u") (result v128)
				v128.const i32x4 0xff00000f 0xe00f7002 0x0f000003 0x000ff004
				i32.const 11
				i8x16.shr_u)
			(func (export "i16_shl") (result v128)
				v128.const i32x4 0xff000071 0xe0000702 0x00000003 0x00000004
				i32.const 19
				i16x8.shl)
			(func (export "i16_shr_s") (result v128)
				v128.const i32x4 0xff00000f 0xe00f7002 0x0f000003 0x000ff004
				i32.const 19
				i16x8.shr_s)
			(func (export "i16_shr_u") (result v128)
				v128.const i32x4 0xff00000f 0xe00f7002 0x0f000003 0x000ff004
				i32.const 19
				i16x8.shr_u)
			(func (export "i32_shl") (result v128)
				v128.const i32x4 0xff0ff071 0xe0077702 0xe0004003 0x00002004
				i32.const 35
				i32x4.shl)
			(func (export "i32_shr_s") (result v128)
				v128.const i32x4 0xff00000f 0xe00f7002 0x0f000003 0x000ff004
				i32.const 35
				i32x4.shr_s)
			(func (export "i32_shr_u") (result v128)
				v128.const i32x4 0xff00000f 0xe00f7002 0x0f000003 0x000ff004
				i32.const 35
				i32x4.shr_u)
			(func (export "i64_shl") (result v128)
				v128.const i32x4 0xff000055 0xe0000702 0xe0004003 0x00002004
				i32.const 67
				i64x2.shl)
			(func (export "i64_shr_s") (result v128)
				v128.const i32x4 0xff00000f 0xe00f7002 0x0f000003 0x000ff004
				i32.const 67
				i64x2.shr_s)
			(func (export "i64_shr_u") (result v128)
				v128.const i32x4 0xff00000f 0xe00f7002 0x0f000003 0x000ff004
				i32.const 67
				i64x2.shr_u)
			(func (export "bitselect") (result v128)
				v128.const i32x4 0x00ff0001 0x00040002 0x55555555 0x00000004
				v128.const i32x4 0x00020001 0x00fe0002 0xaaaaaaaa 0x55000004
				v128.const i32x4 0xffffffff 0x00000000 0x55555555 0x55000004
				v128.bitselect))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	checkV128Export(t, inst, "i8_shl", v128I32x4(0xf8000008, 0x00000010, 0x00000018, 0x00000020))
	checkV128Export(t, inst, "i8_shr_s", v128I32x4(0xff000001, 0xfc010e00, 0x01000000, 0x0001fe00))
	checkV128Export(t, inst, "i8_shr_u", v128I32x4(0x1f000001, 0x1c010e00, 0x01000000, 0x00011e00))
	checkV128Export(t, inst, "i16_shl", v128I32x4(0xf8000388, 0x00003810, 0x00000018, 0x00000020))
	checkV128Export(t, inst, "i16_shr_s", v128I32x4(0xffe00001, 0xfc010e00, 0x01e00000, 0x0001fe00))
	checkV128Export(t, inst, "i16_shr_u", v128I32x4(0x1fe00001, 0x1c010e00, 0x01e00000, 0x00011e00))
	checkV128Export(t, inst, "i32_shl", v128I32x4(0xf87f8388, 0x003bb810, 0x00020018, 0x00010020))
	checkV128Export(t, inst, "i32_shr_s", v128I32x4(0xffe00001, 0xfc01ee00, 0x01e00000, 0x0001fe00))
	checkV128Export(t, inst, "i32_shr_u", v128I32x4(0x1fe00001, 0x1c01ee00, 0x01e00000, 0x0001fe00))
	checkV128Export(t, inst, "i64_shl", v128I32x4(0xf80002a8, 0x00003817, 0x00020018, 0x00010027))
	checkV128Export(t, inst, "i64_shr_s", v128I32x4(0x5fe00001, 0xfc01ee00, 0x81e00000, 0x0001fe00))
	checkV128Export(t, inst, "i64_shr_u", v128I32x4(0x5fe00001, 0x1c01ee00, 0x81e00000, 0x0001fe00))
	checkV128Export(t, inst, "bitselect", v128I32x4(0x00ff0001, 0x00fe0002, 0xffffffff, 0x00000004))
}

// TestV128Compare checks SIMD integer and floating-point comparison masks.
func TestV128Compare(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i8_lt_s") (result v128)
				v128.const i32x4 0xff000001 0xe0000002 0x00008003 0x00000004
				v128.const i32x4 0x02000001 0xe000ff02 0x00000003 0x00008104
				i8x16.lt_s)
			(func (export "i16_ge_u") (result v128)
				v128.const i32x4 0xff000001 0xe0000002 0x00008003 0x00000004
				v128.const i32x4 0x02000001 0xe000ff02 0x00000003 0x00008104
				i16x8.ge_u)
			(func (export "i32_eq") (result v128)
				v128.const i32x4 0xff000001 0xe0000002 0x00000003 0x77000004
				v128.const i32x4 0x05000001 0x0e002002 0x44000003 0x00000004
				i32x4.eq)
			(func (export "i64_lt_s") (result v128)
				v128.const i32x4 0xffffffff 0xffffffff 0x00000000 0x80000000
				v128.const i32x4 0x00000000 0x00000000 0xffffffff 0x7fffffff
				i64x2.lt_s)
			(func (export "f32_eq") (result v128)
				v128.const i32x4 0x00000000 0xffc00000 0x449a5000 0x449a5000
				v128.const i32x4 0x80000000 0xffc00000 0x449a5000 0x3f800000
				f32x4.eq)
			(func (export "f64_gt") (result v128)
				v128.const i32x4 0x00000000 0x3ff80000 0x00000000 0x3ff80000
				v128.const i32x4 0x00000000 0xfff80000 0x00000000 0x3ff00000
				f64x2.gt))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	checkV128Export(t, inst, "i8_lt_s", v128I32x4(0xff000000, 0x00000000, 0x0000ff00, 0x00000000))
	checkV128Export(t, inst, "i16_ge_u", v128I32x4(0xffffffff, 0xffff0000, 0xffffffff, 0xffff0000))
	checkV128Export(t, inst, "i32_eq", v128I32x4(0x00000000, 0x00000000, 0x00000000, 0x00000000))
	checkV128Export(t, inst, "i64_lt_s", v128I32x4(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff))
	checkV128Export(t, inst, "f32_eq", v128I32x4(0xffffffff, 0x00000000, 0xffffffff, 0x00000000))
	checkV128Export(t, inst, "f64_gt", v128I32x4(0x00000000, 0x00000000, 0xffffffff, 0xffffffff))
}

// TestV128Unary checks SIMD unary operations, tests, bitmasks, and conversions.
func TestV128Unary(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i8_neg") (result v128)
				v128.const i32x4 0x00000001 0x00000002 0x00000003 0x00000004
				i8x16.neg)
			(func (export "i16_neg") (result v128)
				v128.const i32x4 0x0000ffff 0x00007fff 0x00000003 0x00000004
				i16x8.neg)
			(func (export "i32_neg") (result v128)
				v128.const i32x4 0x00000001 0x00000002 0x00000003 0x00000004
				i32x4.neg)
			(func (export "i64_neg") (result v128)
				v128.const i32x4 0x00000001 0x00000002 0x00000003 0x00000004
				i64x2.neg)
			(func (export "vnot") (result v128)
				v128.const i32x4 0x00ff0001 0x00550002 0x00000003 0x00000004
				v128.not)
			(func (export "any_true") (result i32)
				v128.const i32x4 0x00ff0001 0x00550002 0x00000003 0x00000004
				v128.any_true)
			(func (export "all_true") (result i32)
				v128.const i32x4 0x00040004 0x00030003 0x00020002 0x00010001
				i16x8.all_true)
			(func (export "bitmask") (result i32)
				v128.const i32x4 0x80008000 0x80008000 0x80008000 0x80008000
				i16x8.bitmask)
			(func (export "f32_neg") (result v128)
				v128.const i32x4 0x80000000 0xffc00000 0x449a5000 0xbf800000
				f32x4.neg)
			(func (export "f64_abs") (result v128)
				v128.const i32x4 0x00000000 0xc0934a00 0x00000000 0x3ff00000
				f64x2.abs)
			(func (export "f32_sqrt") (result v128)
				v128.const i32x4 0xbf800000 0xffc00000 0x40800000 0x41100000
				f32x4.sqrt)
			(func (export "convert_s") (result v128)
				v128.const i32x4 0x00000001 0xffffffff 0x00000000 0x00000003
				f32x4.convert_i32x4_s)
			(func (export "trunc_sat_u") (result v128)
				v128.const i32x4 0x3fc00000 0x40900000 0xffc00000 0x449a599a
				i32x4.trunc_sat_f32x4_u))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	checkV128Export(t, inst, "i8_neg", v128I32x4(0x000000ff, 0x000000fe, 0x000000fd, 0x000000fc))
	checkV128Export(t, inst, "i16_neg", v128I32x4(0x00000001, 0x00008001, 0x0000fffd, 0x0000fffc))
	checkV128Export(t, inst, "i32_neg", v128I32x4(0xffffffff, 0xfffffffe, 0xfffffffd, 0xfffffffc))
	checkV128Export(t, inst, "i64_neg", v128I32x4(0xffffffff, 0xfffffffd, 0xfffffffd, 0xfffffffb))
	checkV128Export(t, inst, "vnot", v128I32x4(0xff00fffe, 0xffaafffd, 0xfffffffc, 0xfffffffb))
	checkV128Export(t, inst, "f32_neg", v128I32x4(0x00000000, 0x7fc00000, 0xc49a5000, 0x3f800000))
	checkV128Export(t, inst, "f64_abs", v128I32x4(0x00000000, 0x40934a00, 0x00000000, 0x3ff00000))
	checkV128Export(t, inst, "f32_sqrt", v128I32x4(0x7fc00000, 0x7fc00000, 0x40000000, 0x40400000))
	checkV128Export(t, inst, "convert_s", v128I32x4(0x3f800000, 0xbf800000, 0x00000000, 0x40400000))
	checkV128Export(t, inst, "trunc_sat_u", v128I32x4(0x00000001, 0x00000004, 0x00000000, 0x000004d2))

	scalarChecks := []struct {
		name string
		want wasmvm.Value
	}{
		{"any_true", wasmvm.I32(1)},
		{"all_true", wasmvm.I32(1)},
		{"bitmask", wasmvm.I32(255)},
	}
	for _, check := range scalarChecks {
		results := callExport(t, inst, check.name)
		if len(results) != 1 || results[0] != check.want {
			t.Fatalf("%s got results %#v, want %#v", check.name, results, check.want)
		}
	}
}

// TestV128Binary checks SIMD binary arithmetic, bitwise ops, and swizzle.
func TestV128Binary(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "i8_add") (result v128)
				v128.const i32x4 0x00ff0001 0x04000002 0x00000003 0x00000004
				v128.const i32x4 0x00020001 0xfe000002 0x00000003 0x00000004
				i8x16.add)
			(func (export "i16_mul") (result v128)
				v128.const i32x4 0x00ff0001 0x00040002 0x00000003 0x00000004
				v128.const i32x4 0x00020001 0x00fe0002 0x00000003 0x00000004
				i16x8.mul)
			(func (export "i64_mul") (result v128)
				v128.const i32x4 0x00ff0001 0x00040002 0x00000003 0x00000004
				v128.const i32x4 0x00020001 0x00fe0002 0x00000003 0x00000004
				i64x2.mul)
			(func (export "i8_add_sat_s") (result v128)
				v128.const i32x4 0x00000001 0x0000007f 0x00000003 0x00000080
				v128.const i32x4 0x00000001 0x00000002 0x00000003 0x000000ff
				i8x16.add_sat_s)
			(func (export "i16_sub_sat_u") (result v128)
				v128.const i32x4 0x00ffffff 0x0400ffff 0x00000003 0x00000004
				v128.const i32x4 0x00020001 0xfe000002 0x00000003 0x00000004
				i16x8.sub_sat_u)
			(func (export "vxor") (result v128)
				v128.const i32x4 0x00ff0001 0x00040002 0x44000003 0x00000004
				v128.const i32x4 0x00020001 0x00fe0002 0x00000003 0x55000004
				v128.xor)
			(func (export "f32_min") (result v128)
				v128.const i32x4 0x80000000 0xffc00000 0x449a5000 0xbf800000
				v128.const i32x4 0x00000000 0x3f800000 0x449a5000 0x3f800000
				f32x4.min)
			(func (export "f32_div") (result v128)
				v128.const i32x4 0x80000000 0xffc00000 0x3fc00000 0xc0400000
				v128.const i32x4 0x00000000 0x3f800000 0x3f800000 0x3fc00000
				f32x4.div)
			(func (export "f64_add") (result v128)
				v128.const i32x4 0x00000000 0x3ff80000 0x00000000 0xfff80000
				v128.const i32x4 0x00000000 0xc0934a00 0x00000000 0x3ff00000
				f64x2.add)
			(func (export "swizzle") (result v128)
				v128.const i32x4 0x04030201 0x08070605 0x0c0b0a09 0x100f0e0d
				v128.const i8x16 0 4 8 12 5 9 13 1 10 14 6 2 15 3 7 11
				i8x16.swizzle))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	checkV128Export(t, inst, "i8_add", v128I32x4(0x00010002, 0x02000004, 0x00000006, 0x00000008))
	checkV128Export(t, inst, "i16_mul", v128I32x4(0x01fe0001, 0x03f80004, 0x00000009, 0x00000010))
	checkV128Export(t, inst, "i64_mul", v128I32x4(0x01010001, 0x03040202, 0x00000009, 0x00000018))
	checkV128Export(t, inst, "i8_add_sat_s", v128I32x4(0x00000002, 0x0000007f, 0x00000006, 0x00000080))
	checkV128Export(t, inst, "i16_sub_sat_u", v128I32x4(0x00fdfffe, 0x0000fffd, 0x00000000, 0x00000000))
	checkV128Export(t, inst, "vxor", v128I32x4(0x00fd0000, 0x00fa0000, 0x44000000, 0x55000000))
	checkV128Export(t, inst, "f32_min", v128I32x4(0x80000000, 0x7fc00000, 0x449a5000, 0xbf800000))
	checkV128Export(t, inst, "f32_div", v128I32x4(0x7fc00000, 0x7fc00000, 0x3fc00000, 0xc0000000))
	checkV128Export(t, inst, "f64_add", v128I32x4(0x00000000, 0xc0934400, 0x00000000, 0x7ff80000))
	checkV128Export(t, inst, "swizzle", v128I32x4(0x0d090501, 0x020e0a06, 0x03070f0b, 0x0c080410))
}

// v128I32x4 builds the byte representation of a v128 value from i32 lanes.
func v128I32x4(l0, l1, l2, l3 uint32) [16]byte {
	return [16]byte{
		byte(l0), byte(l0 >> 8), byte(l0 >> 16), byte(l0 >> 24),
		byte(l1), byte(l1 >> 8), byte(l1 >> 16), byte(l1 >> 24),
		byte(l2), byte(l2 >> 8), byte(l2 >> 16), byte(l2 >> 24),
		byte(l3), byte(l3 >> 8), byte(l3 >> 16), byte(l3 >> 24),
	}
}

// TestMemory64ScalarOps checks that memory64 load/store instructions consume
// i64 address operands and that active data offsets can also be i64.
func TestMemory64ScalarOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory i64 1)
			(data (i64.const 0) "\01\02\03\04\05\06\07\08")
			(func (export "load_i32") (result i32)
				i64.const 0
				i32.load)
			(func (export "load_i64") (result i64)
				i64.const 0
				i64.load)
			(func (export "roundtrip_f64") (param f64) (result f64)
				i64.const 16
				local.get 0
				f64.store
				i64.const 16
				f64.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "load_i32")
	if len(results) != 1 || results[0] != wasmvm.I32(0x04030201) {
		t.Fatalf("load_i32 got results %#v, want i32 0x04030201", results)
	}

	results = callExport(t, inst, "load_i64")
	if len(results) != 1 || results[0] != wasmvm.I64(0x0807060504030201) {
		t.Fatalf("load_i64 got results %#v, want i64 0x0807060504030201", results)
	}

	results = callExport(t, inst, "roundtrip_f64", wasmvm.F64(-9.25))
	if len(results) != 1 || results[0] != wasmvm.F64(-9.25) {
		t.Fatalf("roundtrip_f64 got results %#v, want f64 -9.25", results)
	}
}

// TestMemory64SizeAndGrow checks that memory.size and memory.grow use i64
// operands and results for memory64 memories.
func TestMemory64SizeAndGrow(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory i64 1 2)
			(func (export "size") (result i64)
				memory.size)
			(func (export "grow") (param i64) (result i64)
				local.get 0
				memory.grow))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "size")
	if len(results) != 1 || results[0] != wasmvm.I64(1) {
		t.Fatalf("size got results %#v, want i64 1", results)
	}

	results = callExport(t, inst, "grow", wasmvm.I64(1))
	if len(results) != 1 || results[0] != wasmvm.I64(1) {
		t.Fatalf("grow got results %#v, want old size i64 1", results)
	}

	results = callExport(t, inst, "grow", wasmvm.I64(1))
	if len(results) != 1 || results[0] != wasmvm.I64(-1) {
		t.Fatalf("failed grow got results %#v, want i64 -1", results)
	}
}

func TestI64NarrowMemoryOps(t *testing.T) {
	// Narrow i64 loads extend to i64, and narrow i64 stores truncate the
	// low-order bytes of the stored value.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data (i32.const 0) "\ff\80\34\12\ff\ff\ff\80")
			(func (export "load8_s") (result i64)
				i32.const 0
				i64.load8_s)
			(func (export "load8_u") (result i64)
				i32.const 0
				i64.load8_u)
			(func (export "load16_s") (result i64)
				i32.const 0
				i64.load16_s)
			(func (export "load16_u") (result i64)
				i32.const 2
				i64.load16_u)
			(func (export "load32_s") (result i64)
				i32.const 4
				i64.load32_s)
			(func (export "load32_u") (result i64)
				i32.const 4
				i64.load32_u)
			(func (export "store8") (result i64)
				i32.const 16
				i64.const 0x123456789abcdef0
				i64.store8
				i32.const 16
				i64.load8_u)
			(func (export "store16") (result i64)
				i32.const 18
				i64.const 0x123456789abcdef0
				i64.store16
				i32.const 18
				i64.load16_u)
			(func (export "store32") (result i64)
				i32.const 20
				i64.const 0x123456789abcdef0
				i64.store32
				i32.const 20
				i64.load32_u))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	for _, tt := range []struct {
		name string
		want int64
	}{
		{name: "load8_s", want: -1},
		{name: "load8_u", want: 255},
		{name: "load16_s", want: -32513},
		{name: "load16_u", want: 0x1234},
		{name: "load32_s", want: -2130706433},
		{name: "load32_u", want: 0x80ffffff},
		{name: "store8", want: 0xf0},
		{name: "store16", want: 0xdef0},
		{name: "store32", want: 0x9abcdef0},
	} {
		results := callExport(t, inst, tt.name)
		if len(results) != 1 || results[0] != wasmvm.I64(tt.want) {
			t.Fatalf("%s got results %#v, want i64 %d", tt.name, results, tt.want)
		}
	}
}

func TestMemorySizeAndGrow(t *testing.T) {
	// memory.grow returns the old size on success, -1 on failure, and newly
	// allocated pages are zero-initialized.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1 3)
			(func (export "size") (result i32)
				memory.size)
			(func (export "grow") (param i32) (result i32)
				local.get 0
				memory.grow)
			(func (export "load_grown_page") (result i32)
				i32.const 70000
				i32.load)
			(func (export "store_grown_page") (result i32)
				i32.const 70000
				i32.const 99
				i32.store
				i32.const 70000
				i32.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "size")
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("initial size got results %#v, want i32 1", results)
	}

	results = callExport(t, inst, "grow", wasmvm.I32(1))
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("grow(1) got results %#v, want old size i32 1", results)
	}

	results = callExport(t, inst, "size")
	if len(results) != 1 || results[0] != wasmvm.I32(2) {
		t.Fatalf("size after grow got results %#v, want i32 2", results)
	}

	results = callExport(t, inst, "load_grown_page")
	if len(results) != 1 || results[0] != wasmvm.I32(0) {
		t.Fatalf("load_grown_page got results %#v, want zero-filled i32 0", results)
	}

	results = callExport(t, inst, "store_grown_page")
	if len(results) != 1 || results[0] != wasmvm.I32(99) {
		t.Fatalf("store_grown_page got results %#v, want i32 99", results)
	}

	results = callExport(t, inst, "grow", wasmvm.I32(2))
	if len(results) != 1 || results[0] != wasmvm.I32(-1) {
		t.Fatalf("grow past max got results %#v, want i32 -1", results)
	}

	results = callExport(t, inst, "size")
	if len(results) != 1 || results[0] != wasmvm.I32(2) {
		t.Fatalf("size after failed grow got results %#v, want i32 2", results)
	}
}

// TestMemoryImportSharesExportedMemory checks that a memory exported from one
// instance can be imported into another instance and remains shared.
func TestMemoryImportSharesExportedMemory(t *testing.T) {
	rt := wasmvm.NewRuntime()
	exporter, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory (export "memory") 1)
			(func (export "grow") (result i32)
				i32.const 1
				memory.grow))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate exporter failed: %v", err)
	}
	mem, ok := exporter.ExportedMemory("memory")
	if !ok {
		t.Fatal("missing memory export")
	}

	results := callExport(t, exporter, "grow")
	if len(results) != 1 || results[0] != wasmvm.I32(1) {
		t.Fatalf("exporter grow got results %#v, want old size i32 1", results)
	}
	if got, want := mem.Size(), uint64(2); got != want {
		t.Fatalf("exported memory size = %d, want %d", got, want)
	}

	importer, err := rt.Instantiate(parseWAT(t, `
		(module
			(import "env" "memory" (memory 2))
			(func (export "size") (result i32)
				memory.size)
			(func (export "grow") (result i32)
				i32.const 1
				memory.grow))
	`), wasmvm.Imports{"env": {"memory": mem}})
	if err != nil {
		t.Fatalf("Instantiate importer failed: %v", err)
	}

	results = callExport(t, importer, "size")
	if len(results) != 1 || results[0] != wasmvm.I32(2) {
		t.Fatalf("importer size got results %#v, want i32 2", results)
	}
	results = callExport(t, importer, "grow")
	if len(results) != 1 || results[0] != wasmvm.I32(2) {
		t.Fatalf("importer grow got results %#v, want old size i32 2", results)
	}
	if got, want := mem.Size(), uint64(3); got != want {
		t.Fatalf("shared memory size = %d, want %d", got, want)
	}
}

// TestMemoryCopyAndFill checks the bulk memory instructions that move or
// initialize byte ranges inside an instantiated memory.
func TestMemoryCopyAndFill(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data (i32.const 0) "abcdef")
			(func (export "copy_overlap") (result i32)
				i32.const 2
				i32.const 0
				i32.const 4
				memory.copy
				i32.const 0
				i32.load)
			(func (export "fill") (result i32)
				i32.const 8
				i32.const 127
				i32.const 4
				memory.fill
				i32.const 8
				i32.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "copy_overlap")
	if len(results) != 1 || results[0] != wasmvm.I32(0x62616261) {
		t.Fatalf("copy_overlap got results %#v, want i32 0x62616261", results)
	}

	results = callExport(t, inst, "fill")
	if len(results) != 1 || results[0] != wasmvm.I32(0x7f7f7f7f) {
		t.Fatalf("fill got results %#v, want i32 0x7f7f7f7f", results)
	}
}

// TestPassiveDataMemoryInit checks that memory.init copies from a passive data
// segment into memory and honors the source offset operand.
func TestPassiveDataMemoryInit(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data "abcdef")
			(func (export "init") (result i32)
				i32.const 8
				i32.const 1
				i32.const 4
				memory.init 0
				i32.const 8
				i32.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "init")
	if len(results) != 1 || results[0] != wasmvm.I32(0x65646362) {
		t.Fatalf("init got results %#v, want i32 0x65646362", results)
	}
}

// The execution-error tests below use hand-built wasmir modules instead of WAT.
// WAT parsing validates stack shape and function indices before the runtime
// sees the code, but these tests specifically check the diagnostics produced
// when the VM encounters invalid runtime state.

func TestExecutionErrorInstructionContext(t *testing.T) {
	// A binary instruction with no operands should report the instruction's pc
	// and opcode in addition to the low-level stack underflow.
	err := callInvalidRuntimeModule(t, []wasmir.Instruction{
		{Kind: wasmir.InstrI32Add},
		{Kind: wasmir.InstrEnd},
	}, []wasmir.ValueType{wasmir.ValueTypeI32})

	if got, want := err.Error(), "pc 0 i32.add: operand stack underflow"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestExecutionErrorUnreachableContext checks that unreachable traps include
// the failing instruction location.
func TestExecutionErrorUnreachableContext(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "trap")
				unreachable))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	trap, ok := inst.ExportedFunc("trap")
	if !ok {
		t.Fatal("missing trap export")
	}
	_, err = trap.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 0 unreachable: unreachable executed"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestExecutionErrorCallContext(t *testing.T) {
	// A call to an invalid function index should report the call instruction's
	// pc and opcode along with the resolver error.
	err := callInvalidRuntimeModule(t, []wasmir.Instruction{
		{Kind: wasmir.InstrCall, FuncIndex: 3},
		{Kind: wasmir.InstrEnd},
	}, nil)

	if got, want := err.Error(), "pc 0 call: call function index 3 out of range"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestExecutionErrorReturnCallContext checks that return_call reports the
// instruction location when direct call resolution fails.
func TestExecutionErrorReturnCallContext(t *testing.T) {
	err := callInvalidRuntimeModule(t, []wasmir.Instruction{
		{Kind: wasmir.InstrReturnCall, FuncIndex: 3},
		{Kind: wasmir.InstrEnd},
	}, nil)

	if got, want := err.Error(), "pc 0 return_call: call function index 3 out of range"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestExecutionErrorTableOutOfBoundsContext checks that table access traps
// include the failing instruction location.
func TestExecutionErrorTableOutOfBoundsContext(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(table 1 funcref)
			(func (export "get_oob")
				i32.const 1
				table.get
				drop))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("get_oob")
	if !ok {
		t.Fatal("missing get_oob export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 1 table.get: table access out of bounds"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestExecutionErrorResultContext(t *testing.T) {
	// A function that declares a result but leaves the stack empty should report
	// the final end instruction as the failing execution point.
	err := callInvalidRuntimeModule(t, []wasmir.Instruction{
		{Kind: wasmir.InstrEnd},
	}, []wasmir.ValueType{wasmir.ValueTypeI32})

	if got, want := err.Error(), "pc 0 end: operand stack underflow"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestExecutionErrorSelectTypeContext(t *testing.T) {
	// A select with mismatched candidate value types should report the select
	// instruction as the failing execution point.
	err := callInvalidRuntimeModule(t, []wasmir.Instruction{
		{Kind: wasmir.InstrI32Const, I32Const: 10},
		{Kind: wasmir.InstrI64Const, I64Const: 20},
		{Kind: wasmir.InstrI32Const, I32Const: 1},
		{Kind: wasmir.InstrSelect},
		{Kind: wasmir.InstrEnd},
	}, []wasmir.ValueType{wasmir.ValueTypeI32})

	if got, want := err.Error(), "pc 3 select: select got i32 and i64 operands"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestExecutionErrorGlobalSetImmutableContext(t *testing.T) {
	// Setting an immutable global should report the global.set instruction as
	// the failing execution point.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(&wasmir.Module{
		Types: []wasmir.TypeDef{{
			Kind: wasmir.TypeDefKindFunc,
		}},
		Globals: []wasmir.Global{{
			Type: wasmir.ValueTypeI32,
			Init: []wasmir.Instruction{{Kind: wasmir.InstrI32Const, I32Const: 0}},
		}},
		Funcs: []wasmir.Function{{
			TypeIdx: 0,
			Body: []wasmir.Instruction{
				{Kind: wasmir.InstrI32Const, I32Const: 1},
				{Kind: wasmir.InstrGlobalSet, GlobalIndex: 0},
				{Kind: wasmir.InstrEnd},
			},
		}},
		Exports: []wasmir.Export{{
			Name:  "run",
			Kind:  wasmir.ExternalKindFunction,
			Index: 0,
		}},
	}, nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("run")
	if !ok {
		t.Fatal("missing run export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 1 global.set: global 0 is immutable"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestExecutionErrorMemoryOutOfBoundsContext(t *testing.T) {
	// An out-of-bounds memory store should report the store instruction as the
	// failing execution point.
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(func (export "store_oob")
				i32.const 65533
				i32.const 1
				i32.store))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("store_oob")
	if !ok {
		t.Fatal("missing store_oob export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 2 i32.store: memory access out of bounds"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestExecutionErrorMemory64OutOfBoundsContext checks that out-of-bounds
// memory64 operations still report the failing instruction location.
func TestExecutionErrorMemory64OutOfBoundsContext(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory i64 1)
			(func (export "load_oob") (result i32)
				i64.const 65533
				i32.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("load_oob")
	if !ok {
		t.Fatal("missing load_oob export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 1 i32.load: memory access out of bounds"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func TestInstantiateRejectsOutOfBoundsDataSegment(t *testing.T) {
	// Active data segments are bounds-checked while the instance memory is
	// initialized.
	rt := wasmvm.NewRuntime()
	_, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data (i32.const 65534) "ABCD"))
	`), nil)
	if err == nil {
		t.Fatal("Instantiate succeeded unexpectedly")
	}
	if got, want := err.Error(), "data[0]: memory access out of bounds"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestStartFunctionRunsDuringInstantiate checks that instantiation executes
// the module's start function before exported functions are called.
func TestStartFunctionRunsDuringInstantiate(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(global $g (mut i32) (i32.const 0))
			(func $start
				i32.const 0
				i32.const 42
				i32.store
				i32.const 7
				global.set $g)
			(start $start)
			(func (export "get_mem") (result i32)
				i32.const 0
				i32.load)
			(func (export "get_global") (result i32)
				global.get $g))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	if got, want := callExport(t, inst, "get_mem")[0].I32, int32(42); got != want {
		t.Fatalf("get_mem() = %d, want %d", got, want)
	}
	if got, want := callExport(t, inst, "get_global")[0].I32, int32(7); got != want {
		t.Fatalf("get_global() = %d, want %d", got, want)
	}
}

// TestStartFunctionTrapFailsInstantiate checks that a trap in the module's
// start function fails instantiation.
func TestStartFunctionTrapFailsInstantiate(t *testing.T) {
	rt := wasmvm.NewRuntime()
	_, err := rt.Instantiate(parseWAT(t, `
		(module
			(func $start unreachable)
			(start $start))
	`), nil)
	if err == nil {
		t.Fatal("Instantiate succeeded unexpectedly")
	}
	if got, want := err.Error(), "start function: pc 0 unreachable: unreachable executed"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestExecutionErrorMemoryFillOutOfBoundsContext checks that memory.fill traps
// include the failing instruction location.
func TestExecutionErrorMemoryFillOutOfBoundsContext(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(func (export "fill_oob")
				i32.const 65535
				i32.const 1
				i32.const 2
				memory.fill))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("fill_oob")
	if !ok {
		t.Fatal("missing fill_oob export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 3 memory.fill: memory access out of bounds"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestExecutionErrorMemoryInitAfterDataDropContext checks that data.drop makes
// a data segment unavailable for later memory.init operations.
func TestExecutionErrorMemoryInitAfterDataDropContext(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data "abc")
			(func (export "drop_then_init")
				data.drop 0
				i32.const 0
				i32.const 0
				i32.const 1
				memory.init 0))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("drop_then_init")
	if !ok {
		t.Fatal("missing drop_then_init export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 4 memory.init: data segment 0 is dropped"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

// TestExecutionErrorMemoryInitSourceOutOfBoundsContext checks that memory.init
// reports data segment source-range failures with instruction context.
func TestExecutionErrorMemoryInitSourceOutOfBoundsContext(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory 1)
			(data "abc")
			(func (export "init_oob")
				i32.const 0
				i32.const 2
				i32.const 2
				memory.init 0))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("init_oob")
	if !ok {
		t.Fatal("missing init_oob export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	if got, want := err.Error(), "pc 3 memory.init: data segment access out of bounds"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
}

func invalidRuntimeModule(body []wasmir.Instruction, results []wasmir.ValueType) *wasmir.Module {
	return &wasmir.Module{
		Types: []wasmir.TypeDef{{
			Kind:    wasmir.TypeDefKindFunc,
			Results: results,
		}},
		Funcs: []wasmir.Function{{
			TypeIdx: 0,
			Body:    body,
		}},
		Exports: []wasmir.Export{{
			Name:  "run",
			Kind:  wasmir.ExternalKindFunction,
			Index: 0,
		}},
	}
}

func callInvalidRuntimeModule(t *testing.T, body []wasmir.Instruction, results []wasmir.ValueType) error {
	t.Helper()

	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(invalidRuntimeModule(body, results), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("run")
	if !ok {
		t.Fatal("missing run export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call succeeded unexpectedly")
	}
	return err
}

// callInvalidCallRefRuntimeModule runs a deliberately unvalidated module whose
// call_ref operand is a function reference with the wrong runtime signature.
func callInvalidCallRefRuntimeModule(t *testing.T) error {
	t.Helper()

	rt := wasmvm.NewRuntime()
	i32 := wasmir.ValueTypeI32
	m := &wasmir.Module{
		Types: []wasmir.TypeDef{
			{Kind: wasmir.TypeDefKindFunc, Params: []wasmir.ValueType{i32, i32}, Results: []wasmir.ValueType{i32}},
			{Kind: wasmir.TypeDefKindFunc, Params: []wasmir.ValueType{i32}, Results: []wasmir.ValueType{i32}},
			{Kind: wasmir.TypeDefKindFunc, Results: []wasmir.ValueType{i32}},
		},
		Funcs: []wasmir.Function{
			{
				TypeIdx: 2,
				Body: []wasmir.Instruction{
					{Kind: wasmir.InstrI32Const, I32Const: 1},
					{Kind: wasmir.InstrI32Const, I32Const: 2},
					{Kind: wasmir.InstrRefFunc, FuncIndex: 1},
					{Kind: wasmir.InstrCallRef, CallTypeIndex: 0},
					{Kind: wasmir.InstrEnd},
				},
			},
			{
				TypeIdx: 1,
				Body: []wasmir.Instruction{
					{Kind: wasmir.InstrLocalGet, LocalIndex: 0},
					{Kind: wasmir.InstrI32Const, I32Const: 1},
					{Kind: wasmir.InstrI32Add},
					{Kind: wasmir.InstrEnd},
				},
			},
		},
		Exports: []wasmir.Export{{
			Name:  "run",
			Kind:  wasmir.ExternalKindFunction,
			Index: 0,
		}},
	}
	inst, err := rt.Instantiate(m, nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	run, ok := inst.ExportedFunc("run")
	if !ok {
		t.Fatal("missing run export")
	}
	_, err = run.Call()
	if err == nil {
		t.Fatal("Call with mismatched call_ref target succeeded unexpectedly")
	}
	return err
}
