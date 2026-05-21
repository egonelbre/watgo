package typeequiv

import (
	"testing"

	"github.com/eliben/watgo/wasmir"
)

// TestTypesSimpleFunctionEquivalence checks that names are irrelevant and only
// the function shape participates in simple type equivalence.
func TestTypesSimpleFunctionEquivalence(t *testing.T) {
	m := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.ValueTypeF32, wasmir.ValueTypeF32}, []wasmir.ValueType{wasmir.ValueTypeF32}),
		funcType([]wasmir.ValueType{wasmir.ValueTypeF32, wasmir.ValueTypeF32}, []wasmir.ValueType{wasmir.ValueTypeF32}),
		funcType([]wasmir.ValueType{wasmir.ValueTypeF32}, []wasmir.ValueType{wasmir.ValueTypeF32}),
	)

	if !Types(m, 0, m, 1) {
		t.Fatal("equivalent simple function types reported as different")
	}
	if Types(m, 0, m, 2) {
		t.Fatal("different simple function types reported as equivalent")
	}
	if !FuncTypeAndSignature(m, 0, []wasmir.ValueType{wasmir.ValueTypeF32, wasmir.ValueTypeF32}, []wasmir.ValueType{wasmir.ValueTypeF32}) {
		t.Fatal("matching explicit signature reported as different")
	}
	if FuncTypeAndSignature(m, 0, []wasmir.ValueType{wasmir.ValueTypeF32}, []wasmir.ValueType{wasmir.ValueTypeF32}) {
		t.Fatal("different explicit signature reported as equivalent")
	}
}

// TestTypesCrossModuleIndexedRefs checks that indexed reference types are
// compared structurally across module type graphs.
func TestTypesCrossModuleIndexedRefs(t *testing.T) {
	a := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.RefTypeIndexed(1, false)}, nil),
		funcType([]wasmir.ValueType{wasmir.ValueTypeI32}, []wasmir.ValueType{wasmir.ValueTypeF32}),
	)
	b := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.RefTypeIndexed(1, false)}, nil),
		funcType([]wasmir.ValueType{wasmir.ValueTypeI32}, []wasmir.ValueType{wasmir.ValueTypeF32}),
	)
	c := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.RefTypeIndexed(1, false)}, nil),
		funcType([]wasmir.ValueType{wasmir.ValueTypeI64}, []wasmir.ValueType{wasmir.ValueTypeF32}),
	)

	if !Types(a, 0, b, 0) {
		t.Fatal("equivalent cross-module indexed reference types reported as different")
	}
	if Types(a, 0, c, 0) {
		t.Fatal("different cross-module indexed reference types reported as equivalent")
	}
}

// TestTypesRecursiveGroupEquivalence checks recursive groups that refer to
// peers by relative position.
func TestTypesRecursiveGroupEquivalence(t *testing.T) {
	a := moduleWithTypes(
		recFuncType(2, []wasmir.ValueType{wasmir.ValueTypeI32, wasmir.RefTypeIndexed(1, false)}, nil),
		funcType([]wasmir.ValueType{wasmir.ValueTypeI32, wasmir.RefTypeIndexed(0, false)}, nil),
	)
	b := moduleWithTypes(
		recFuncType(2, []wasmir.ValueType{wasmir.ValueTypeI32, wasmir.RefTypeIndexed(1, false)}, nil),
		funcType([]wasmir.ValueType{wasmir.ValueTypeI32, wasmir.RefTypeIndexed(0, false)}, nil),
	)
	c := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.ValueTypeI32, wasmir.RefTypeIndexed(0, false)}, nil),
	)

	if !Types(a, 0, b, 0) {
		t.Fatal("equivalent recursive groups reported as different")
	}
	if !Types(a, 1, b, 1) {
		t.Fatal("equivalent recursive group peers reported as different")
	}
	if Types(a, 0, b, 1) {
		t.Fatal("different recursive group positions reported as equivalent")
	}
	if Types(a, 0, c, 0) {
		t.Fatal("recursive group and singleton type reported as equivalent")
	}
}

// TestTypesStructAndFieldEquivalence checks the struct and field entry points,
// including mutable field equivalence over indexed reference types.
func TestTypesStructAndFieldEquivalence(t *testing.T) {
	a := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.ValueTypeI32}, nil),
		structType(fieldType(wasmir.RefTypeIndexed(0, true), true)),
	)
	b := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.ValueTypeI32}, nil),
		structType(fieldType(wasmir.RefTypeIndexed(0, true), true)),
	)
	c := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.ValueTypeI64}, nil),
		structType(fieldType(wasmir.RefTypeIndexed(0, true), true)),
	)
	d := moduleWithTypes(
		funcType([]wasmir.ValueType{wasmir.ValueTypeI32}, nil),
		structType(fieldType(wasmir.RefTypeIndexed(0, true), false)),
	)

	if !Types(a, 1, b, 1) {
		t.Fatal("equivalent struct types reported as different")
	}
	if Types(a, 1, c, 1) {
		t.Fatal("struct fields with different indexed refs reported as equivalent")
	}
	if !FieldTypes(a, a.Types[1].Fields[0], b, b.Types[1].Fields[0]) {
		t.Fatal("equivalent mutable fields reported as different")
	}
	if FieldTypes(a, a.Types[1].Fields[0], d, d.Types[1].Fields[0]) {
		t.Fatal("fields with different mutability reported as equivalent")
	}
}

// TestRecGroupInfo checks that grouped and ungrouped type indices report their
// containing recursive group.
func TestRecGroupInfo(t *testing.T) {
	m := moduleWithTypes(
		funcType(nil, nil),
		recFuncType(2, []wasmir.ValueType{wasmir.RefTypeIndexed(2, true)}, nil),
		funcType([]wasmir.ValueType{wasmir.RefTypeIndexed(1, true)}, nil),
	)

	if start, size, pos := RecGroupInfo(m, 0); start != 0 || size != 1 || pos != 0 {
		t.Fatalf("RecGroupInfo(0) = (%d, %d, %d), want (0, 1, 0)", start, size, pos)
	}
	if start, size, pos := RecGroupInfo(m, 1); start != 1 || size != 2 || pos != 0 {
		t.Fatalf("RecGroupInfo(1) = (%d, %d, %d), want (1, 2, 0)", start, size, pos)
	}
	if start, size, pos := RecGroupInfo(m, 2); start != 1 || size != 2 || pos != 1 {
		t.Fatalf("RecGroupInfo(2) = (%d, %d, %d), want (1, 2, 1)", start, size, pos)
	}
}

// moduleWithTypes builds a minimal module containing types.
func moduleWithTypes(types ...wasmir.TypeDef) *wasmir.Module {
	return &wasmir.Module{Types: types}
}

// funcType builds a function type definition.
func funcType(params []wasmir.ValueType, results []wasmir.ValueType) wasmir.TypeDef {
	return wasmir.TypeDef{
		Kind:    wasmir.TypeDefKindFunc,
		Params:  params,
		Results: results,
	}
}

// recFuncType builds the first function type definition in a recursive group.
func recFuncType(groupSize uint32, params []wasmir.ValueType, results []wasmir.ValueType) wasmir.TypeDef {
	td := funcType(params, results)
	td.RecGroupSize = groupSize
	return td
}

// structType builds a struct type definition.
func structType(fields ...wasmir.FieldType) wasmir.TypeDef {
	return wasmir.TypeDef{
		Kind:   wasmir.TypeDefKindStruct,
		Fields: fields,
	}
}

// fieldType builds an unpacked field type definition.
func fieldType(t wasmir.ValueType, mutable bool) wasmir.FieldType {
	return wasmir.FieldType{Type: t, Mutable: mutable}
}
