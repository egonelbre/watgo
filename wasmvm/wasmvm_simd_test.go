package wasmvm_test

import (
	"math"
	"testing"

	"github.com/eliben/watgo/wasmir"
	"github.com/eliben/watgo/wasmvm"
)

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
