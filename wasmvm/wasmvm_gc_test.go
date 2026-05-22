package wasmvm_test

import (
	"testing"

	"github.com/eliben/watgo/wasmvm"
)

// TestGCRefI31 checks i31 construction and signed/unsigned extraction.
func TestGCRefI31(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(func (export "get_u") (param i32) (result i32)
				local.get 0
				ref.i31
				i31.get_u)
			(func (export "get_s") (param i32) (result i32)
				local.get 0
				ref.i31
				i31.get_s))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "get_u", 0x7fffffff, wasmvm.I32(-1))
	expectI32Result(t, inst, "get_s", -0x40000000, wasmvm.I32(0x40000000))
}

// TestGCRefEq checks equality over the GC eq hierarchy.
func TestGCRefEq(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $s (struct))
			(type $a (array i8))

			(func (export "same_i31") (param i32 i32) (result i32)
				local.get 0
				ref.i31
				local.get 1
				ref.i31
				ref.eq)
			(func (export "same_struct_ref") (result i32)
				(local $r (ref $s))
				struct.new_default $s
				local.set $r
				local.get $r
				local.get $r
				ref.eq)
			(func (export "different_struct_refs") (result i32)
				struct.new_default $s
				struct.new_default $s
				ref.eq)
			(func (export "different_array_refs") (result i32)
				i32.const 0
				array.new_default $a
				i32.const 0
				array.new_default $a
				ref.eq))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "same_i31", 1, wasmvm.I32(7), wasmvm.I32(7))
	expectI32Result(t, inst, "same_i31", 0, wasmvm.I32(7), wasmvm.I32(8))
	expectI32Result(t, inst, "same_struct_ref", 1)
	expectI32Result(t, inst, "different_struct_refs", 0)
	expectI32Result(t, inst, "different_array_refs", 0)
}

// TestGCRefTest checks runtime type tests for the first supported GC refs.
func TestGCRefTest(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $s (struct))
			(type $a (array i8))

			(func (export "i31_is_i31") (result i32)
				i32.const 1
				ref.i31
				ref.test i31ref)
			(func (export "struct_is_struct") (result i32)
				struct.new_default $s
				ref.test structref)
			(func (export "array_is_array") (result i32)
				i32.const 0
				array.new_default $a
				ref.test arrayref)
			(func (export "i31_is_struct") (result i32)
				i32.const 1
				ref.i31
				ref.test structref)
			(func (export "null_is_i31ref") (result i32)
				ref.null i31
				ref.test i31ref))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "i31_is_i31", 1)
	expectI32Result(t, inst, "struct_is_struct", 1)
	expectI32Result(t, inst, "array_is_array", 1)
	expectI32Result(t, inst, "i31_is_struct", 0)
	expectI32Result(t, inst, "null_is_i31ref", 1)
}

// TestGCRefCast checks successful casts and cast-failure traps.
func TestGCRefCast(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $s (struct (field i32)))

			(func (export "cast_i31") (param i32) (result i32)
				local.get 0
				ref.i31
				ref.cast i31ref
				i31.get_u)
			(func (export "cast_struct") (result i32)
				i32.const 42
				struct.new $s
				ref.cast (ref $s)
				struct.get $s 0)
			(func (export "cast_i31_to_struct")
				i32.const 1
				ref.i31
				ref.cast structref
				drop))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "cast_i31", 42, wasmvm.I32(42))
	expectI32Result(t, inst, "cast_struct", 42)

	f, ok := inst.ExportedFunc("cast_i31_to_struct")
	if !ok {
		t.Fatal("missing cast_i31_to_struct export")
	}
	if _, err := f.Call(); err == nil {
		t.Fatal("cast_i31_to_struct succeeded unexpectedly")
	}
}

// TestGCBranchCast checks that br_on_cast branches only when the runtime
// object kind matches the target type, and that the branch path carries the
// narrowed reference needed by later GC instructions.
func TestGCBranchCast(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $s (struct (field i32)))
			(type $a (array i8))

			(func $pick (param i32) (result anyref)
				local.get 0
				if (result anyref)
					i32.const 9
					i32.const 2
					array.new $a
				else
					i32.const 41
					struct.new $s
				end)

			(func (export "struct_or_minus_one") (param i32) (result i32)
				(block $h (result (ref $s))
					local.get 0
					call $pick
					br_on_cast $h anyref (ref $s)
					drop
					i32.const -1
					return)
				struct.get $s 0)

			(func (export "array_len_or_minus_one") (param i32) (result i32)
				(block $h (result (ref $a))
					local.get 0
					call $pick
					br_on_cast $h anyref (ref $a)
					drop
					i32.const -1
					return)
				array.len))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "struct_or_minus_one", 41, wasmvm.I32(0))
	expectI32Result(t, inst, "struct_or_minus_one", -1, wasmvm.I32(1))
	expectI32Result(t, inst, "array_len_or_minus_one", -1, wasmvm.I32(0))
	expectI32Result(t, inst, "array_len_or_minus_one", 2, wasmvm.I32(1))
}

// TestGCBranchCastFail checks that br_on_cast_fail branches only when the
// runtime object kind does not match the target type.
func TestGCBranchCastFail(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $s (struct))
			(type $a (array i8))

			(func $pick (param i32) (result anyref)
				local.get 0
				if (result anyref)
					i32.const 0
					array.new_default $a
				else
					struct.new_default $s
				end)

			(func (export "array_fails_struct_cast") (param i32) (result i32)
				(block $fail (result anyref)
					local.get 0
					call $pick
					br_on_cast_fail $fail anyref (ref struct)
					drop
					i32.const 0
					return)
				drop
				i32.const 1))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "array_fails_struct_cast", 0, wasmvm.I32(0))
	expectI32Result(t, inst, "array_fails_struct_cast", 1, wasmvm.I32(1))
}

// TestGCStructNewAndGet checks struct allocation and field reads.
func TestGCStructNewAndGet(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $s (struct (field i32) (field i8) (field i16)))

			(func (export "get_i32") (result i32)
				i32.const 42
				i32.const -1
				i32.const 0x8001
				struct.new $s
				struct.get $s 0)
			(func (export "get_i8_u") (result i32)
				i32.const 42
				i32.const -1
				i32.const 0x8001
				struct.new $s
				struct.get_u $s 1)
			(func (export "get_i16_s") (result i32)
				i32.const 42
				i32.const -1
				i32.const 0x8001
				struct.new $s
				struct.get_s $s 2)
			(func (export "default_i32") (result i32)
				struct.new_default $s
				struct.get $s 0))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "get_i32", 42)
	expectI32Result(t, inst, "get_i8_u", 255)
	expectI32Result(t, inst, "get_i16_s", -32767)
	expectI32Result(t, inst, "default_i32", 0)
}

// TestGCArrayNewLenAndGet checks array allocation, length, and element reads.
func TestGCArrayNewLenAndGet(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $i8s (array i8))
			(type $i16s (array i16))

			(func (export "len") (result i32)
				i32.const 7
				i32.const 3
				array.new $i8s
				array.len)
			(func (export "get_i8_u") (result i32)
				i32.const -1
				i32.const 3
				array.new $i8s
				i32.const 1
				array.get_u $i8s)
			(func (export "get_i16_s") (result i32)
				i32.const 0x8001
				i32.const 3
				array.new $i16s
				i32.const 2
				array.get_s $i16s)
			(func (export "default_i8") (result i32)
				i32.const 2
				array.new_default $i8s
				i32.const 0
				array.get_u $i8s))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "len", 3)
	expectI32Result(t, inst, "get_i8_u", 255)
	expectI32Result(t, inst, "get_i16_s", -32767)
	expectI32Result(t, inst, "default_i8", 0)
}

// TestGCStructSet checks that struct.set mutates a mutable field and that
// packed fields are truncated before storage.
func TestGCStructSet(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $s (struct (field i32) (field (mut i8))))

			(func (export "set_i8") (result i32) (local (ref $s))
				i32.const 11
				i32.const 0
				struct.new $s
				local.tee 0
				i32.const -1
				struct.set $s 1
				local.get 0
				struct.get_u $s 1)
			(func (export "set_i32") (result i32) (local (ref $s))
				i32.const 41
				i32.const 0
				struct.new $s
				local.set 0
				local.get 0
				i32.const 255
				struct.set $s 1
				local.get 0
				struct.get_s $s 1))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "set_i8", 255)
	expectI32Result(t, inst, "set_i32", -1)
}

// TestGCArrayNewFixedAndSet checks array.new_fixed element order and array.set
// mutation of a mutable packed element.
func TestGCArrayNewFixedAndSet(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $a (array (mut i16)))

			(func (export "get_fixed") (result i32) (local (ref $a))
				i32.const 10
				i32.const 20
				i32.const 30
				array.new_fixed $a 3
				i32.const 1
				array.get_s $a)
			(func (export "set") (result i32) (local (ref $a))
				i32.const 10
				i32.const 20
				i32.const 30
				array.new_fixed $a 3
				local.set 0
				local.get 0
				i32.const 2
				i32.const 0x8001
				array.set $a
				local.get 0
				i32.const 2
				array.get_s $a))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "get_fixed", 20)
	expectI32Result(t, inst, "set", -32767)
}

// TestGCArrayFill checks that array.fill mutates a range and normalizes packed
// values before storing them.
func TestGCArrayFill(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $a (array (mut i8)))

			(global $g (mut (ref $a)) (array.new_default $a (i32.const 5)))

			(func (export "fill")
				global.get $g
				i32.const 1
				i32.const -1
				i32.const 3
				array.fill $a)
			(func (export "get") (param i32) (result i32)
				global.get $g
				local.get 0
				array.get_u $a)
			(func (export "fill_empty_at_end")
				global.get $g
				i32.const 5
				i32.const 7
				i32.const 0
				array.fill $a))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	callExport(t, inst, "fill")
	expectI32Result(t, inst, "get", 0, wasmvm.I32(0))
	expectI32Result(t, inst, "get", 255, wasmvm.I32(1))
	expectI32Result(t, inst, "get", 255, wasmvm.I32(3))
	expectI32Result(t, inst, "get", 0, wasmvm.I32(4))
	callExport(t, inst, "fill_empty_at_end")
}

// TestGCArrayBulkOperations checks array data/element initialization and
// overlapping array copies.
func TestGCArrayBulkOperations(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $bytes (array (mut i8)))
			(type $refs (array (mut i31ref)))

			(data $d "\10\20\30\40")
			(elem $e i31ref
				(ref.i31 (i32.const 7))
				(ref.i31 (i32.const 8))
				(ref.i31 (i32.const 9)))

			(global $bytes (mut (ref $bytes)) (array.new_default $bytes (i32.const 4)))
			(global $refs (mut (ref $refs)) (array.new_default $refs (i32.const 3)))

			(func (export "new_data") (result i32)
				i32.const 1
				i32.const 2
				array.new_data $bytes $d
				i32.const 1
				array.get_u $bytes)
			(func (export "init_data") (result i32)
				global.get $bytes
				i32.const 1
				i32.const 2
				i32.const 2
				array.init_data $bytes $d
				global.get $bytes
				i32.const 2
				array.get_u $bytes)
			(func (export "new_elem") (result i32)
				i32.const 1
				i32.const 2
				array.new_elem $refs $e
				i32.const 0
				array.get $refs
				i31.get_u)
			(func (export "init_elem") (result i32)
				global.get $refs
				i32.const 0
				i32.const 2
				i32.const 1
				array.init_elem $refs $e
				global.get $refs
				i32.const 0
				array.get $refs
				i31.get_u)
			(func (export "copy_overlap") (result i32)
				i32.const 0
				i32.const 4
				array.new_data $bytes $d
				global.set $bytes
				global.get $bytes
				i32.const 1
				global.get $bytes
				i32.const 0
				i32.const 3
				array.copy $bytes $bytes
				global.get $bytes
				i32.const 3
				array.get_u $bytes))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "new_data", 0x30)
	expectI32Result(t, inst, "init_data", 0x40)
	expectI32Result(t, inst, "new_elem", 8)
	expectI32Result(t, inst, "init_elem", 9)
	expectI32Result(t, inst, "copy_overlap", 0x30)
}

// TestGCConstExprAggregateNew checks that struct and array allocations can be
// used in module-level constant expressions such as global initializers.
func TestGCConstExprAggregateNew(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(type $s (struct (field i32) (field (mut i32))))
			(type $a (array i32))

			(global $s1 (ref $s) (struct.new $s (i32.const 7) (i32.const 11)))
			(global $s2 (ref $s) (struct.new_default $s))
			(global $a1 (ref $a) (array.new $a (i32.const 5) (i32.const 3)))
			(global $a2 (ref $a) (array.new_fixed $a 3 (i32.const 8) (i32.const 9) (i32.const 10)))

			(func (export "struct_new") (result i32)
				global.get $s1
				struct.get $s 1)
			(func (export "struct_default") (result i32)
				global.get $s2
				struct.get $s 0)
			(func (export "array_new") (result i32)
				global.get $a1
				i32.const 2
				array.get $a)
			(func (export "array_fixed") (result i32)
				global.get $a2
				i32.const 1
				array.get $a))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	expectI32Result(t, inst, "struct_new", 11)
	expectI32Result(t, inst, "struct_default", 0)
	expectI32Result(t, inst, "array_new", 5)
	expectI32Result(t, inst, "array_fixed", 9)
}
