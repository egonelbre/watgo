package wasmvm_test

import (
	"testing"

	"github.com/eliben/watgo/wasmvm"
)

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

	expectI32Result(t, inst, "zero", wasmvm.I32(0))

	expectI32Result(t, inst, "roundtrip", wasmvm.I32(0x12345678), wasmvm.I32(12), wasmvm.I32(0x12345678))
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

	expectI32Result(t, inst, "load0", wasmvm.I32(0x44434241))

	expectI32Result(t, inst, "load1", wasmvm.I32(0x5a595857))
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
		expectI32Result(t, inst, tt.name, wasmvm.I32(tt.want))
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

	expectI32Result(t, inst, "load_i32", wasmvm.I32(0x04030201))

	results := callExport(t, inst, "load_i64")
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

// TestMemory64BulkOps checks that memory64 bulk-memory instructions consume
// i64 destination and length operands where required.
func TestMemory64BulkOps(t *testing.T) {
	rt := wasmvm.NewRuntime()
	inst, err := rt.Instantiate(parseWAT(t, `
		(module
			(memory i64 1)
			(data (i64.const 0) "abcdef")
			(data "wxyz")
			(func (export "fill") (result i32)
				i64.const 16
				i32.const 127
				i64.const 4
				memory.fill
				i64.const 16
				i32.load)
			(func (export "copy") (result i32)
				i64.const 32
				i64.const 0
				i64.const 4
				memory.copy
				i64.const 32
				i32.load)
			(func (export "init") (result i32)
				i64.const 48
				i32.const 1
				i32.const 3
				memory.init 1
				i64.const 48
				i32.load))
	`), nil)
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}

	results := callExport(t, inst, "fill")
	if len(results) != 1 || results[0] != wasmvm.I32(0x7f7f7f7f) {
		t.Fatalf("fill got results %#v, want i32 0x7f7f7f7f", results)
	}
	results = callExport(t, inst, "copy")
	if len(results) != 1 || results[0] != wasmvm.I32(0x64636261) {
		t.Fatalf("copy got results %#v, want i32 0x64636261", results)
	}
	results = callExport(t, inst, "init")
	if len(results) != 1 || results[0] != wasmvm.I32(0x007a7978) {
		t.Fatalf("init got results %#v, want i32 0x007a7978", results)
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

// TestExecutionErrorMemoryInitAfterDataDropContext checks that memory.init
// reports a dropped data segment as an out-of-bounds access.
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
	if got, want := err.Error(), "pc 4 memory.init: data segment out of bounds"; got != want {
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
