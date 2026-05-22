package wasmvm_test

import (
	"strings"
	"testing"

	"github.com/eliben/watgo/wasmvm"
)

// TestExceptionThrowCatchAndThrowRef checks the VM exception instructions
// directly through the public wasmvm API.
func TestExceptionThrowCatchAndThrowRef(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(tag $e (param i32))

			(func (export "throw") (param i32)
				local.get 0
				throw $e)

			(func (export "catch_payload") (param i32) (result i32)
				(block $h (result i32)
					try_table (result i32) (catch $e $h)
						local.get 0
						throw $e
						i32.const -1
					end
					return)
				i32.const 1
				i32.add)

			(func (export "catch_ref_rethrow") (param i32) (result i32)
				(local $exn exnref)
				(block $h (result i32 exnref)
					try_table (result i32) (catch_ref $e $h)
						local.get 0
						throw $e
						i32.const -1
					end
					return)
				local.set $exn
				local.set 0
				i32.const 0
				local.get 0
				i32.eq
				if
					local.get $exn
					throw_ref
				end
				local.get 0))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	throw, ok := inst.ExportedFunc("throw")
	if !ok {
		t.Fatal("missing throw export")
	}
	_, err = throw.Call(wasmvm.I32(7))
	if err == nil {
		t.Fatal("throw succeeded unexpectedly")
	}
	if !strings.Contains(err.Error(), "wasm exception") {
		t.Fatalf("throw error = %q, want wasm exception", err)
	}

	results := callExport(t, inst, "catch_payload", wasmvm.I32(41))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("catch_payload got results %#v, want i32 42", results)
	}

	results = callExport(t, inst, "catch_ref_rethrow", wasmvm.I32(42))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("catch_ref_rethrow got results %#v, want i32 42", results)
	}
	rethrow, ok := inst.ExportedFunc("catch_ref_rethrow")
	if !ok {
		t.Fatal("missing catch_ref_rethrow export")
	}
	_, err = rethrow.Call(wasmvm.I32(0))
	if err == nil {
		t.Fatal("catch_ref_rethrow succeeded unexpectedly")
	}
	if !strings.Contains(err.Error(), "wasm exception") {
		t.Fatalf("catch_ref_rethrow error = %q, want wasm exception", err)
	}
}

// TestSharedTagImportExport checks that an exported tag preserves exception
// identity when imported by another module.
func TestSharedTagImportExport(t *testing.T) {
	rt := wasmvm.NewRuntime()

	owner, err := rt.Instantiate(parseWAT(t, `
		(module
			(tag $e (export "e") (param i32))
			(func (export "throw") (param i32)
				local.get 0
				throw $e))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate owner failed: %v", err)
	}
	tag, ok := owner.ExportedTag("e")
	if !ok {
		t.Fatal("missing e tag export")
	}
	throw, ok := owner.ExportedFunc("throw")
	if !ok {
		t.Fatal("missing throw export")
	}

	catcher, err := rt.Instantiate(parseWAT(t, `
		(module
			(tag $e (import "owner" "e") (param i32))
			(func $throw (import "owner" "throw") (param i32))

			(func (export "catch_imported_throw") (param i32) (result i32)
				(block $h (result i32)
					try_table (result i32) (catch $e $h)
						local.get 0
						call $throw
						i32.const -1
					end
					return)
				i32.const 1
				i32.add))
	`), wasmvm.Imports{
		"owner": {
			"e":     tag,
			"throw": throw,
		},
	})
	if err != nil {
		t.Fatalf("Instantiate catcher failed: %v", err)
	}

	results := callExport(t, catcher, "catch_imported_throw", wasmvm.I32(41))
	if len(results) != 1 || results[0] != wasmvm.I32(42) {
		t.Fatalf("catch_imported_throw got results %#v, want i32 42", results)
	}
}
