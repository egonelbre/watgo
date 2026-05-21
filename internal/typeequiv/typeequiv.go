// Package typeequiv compares WebAssembly type definitions structurally.
package typeequiv

import "github.com/eliben/watgo/wasmir"

// FuncTypes reports whether the two type indices name equivalent function
// types in their respective modules.
func FuncTypes(moduleA *wasmir.Module, typeA uint32, moduleB *wasmir.Module, typeB uint32) bool {
	if moduleA == nil || moduleB == nil || int(typeA) >= len(moduleA.Types) || int(typeB) >= len(moduleB.Types) {
		return false
	}
	if moduleA.Types[typeA].Kind != wasmir.TypeDefKindFunc || moduleB.Types[typeB].Kind != wasmir.TypeDefKindFunc {
		return false
	}
	c := newChecker(moduleA, moduleB)
	return c.typeIndicesEquivalent(typeA, typeB)
}

// Types reports whether the two type indices name equivalent type definitions
// in their respective modules.
func Types(moduleA *wasmir.Module, typeA uint32, moduleB *wasmir.Module, typeB uint32) bool {
	c := newChecker(moduleA, moduleB)
	return c.typeIndicesEquivalent(typeA, typeB)
}

// FuncTypeAndSignature reports whether the function type named by typeIndex in
// module is equivalent to the explicit parameter and result type lists.
func FuncTypeAndSignature(module *wasmir.Module, typeIndex uint32, params []wasmir.ValueType, results []wasmir.ValueType) bool {
	if module == nil || int(typeIndex) >= len(module.Types) || module.Types[typeIndex].Kind != wasmir.TypeDefKindFunc {
		return false
	}
	ft := module.Types[typeIndex]
	c := newChecker(module, module)
	return c.valueTypeListsEquivalent(ft.Params, params) && c.valueTypeListsEquivalent(ft.Results, results)
}

// FieldTypes reports whether two struct or array field types are equivalent in
// their respective modules.
func FieldTypes(moduleA *wasmir.Module, fieldA wasmir.FieldType, moduleB *wasmir.Module, fieldB wasmir.FieldType) bool {
	c := newChecker(moduleA, moduleB)
	return c.fieldTypesEquivalent(fieldA, fieldB)
}

// RecGroupInfo returns the recursive group containing idx. Non-grouped types
// behave like singleton groups.
func RecGroupInfo(m *wasmir.Module, idx uint32) (start uint32, size uint32, pos uint32) {
	return recGroupInfo(m, idx)
}

type typePair struct {
	a uint32
	b uint32
}

type checker struct {
	moduleA *wasmir.Module
	moduleB *wasmir.Module

	groupVisiting map[typePair]bool
	groupMemo     map[typePair]bool
	typeVisiting  map[typePair]bool
	typeMemo      map[typePair]bool
}

// newChecker returns a recursive type-equivalence checker for a fixed pair of
// modules.
func newChecker(moduleA *wasmir.Module, moduleB *wasmir.Module) checker {
	return checker{
		moduleA:       moduleA,
		moduleB:       moduleB,
		groupVisiting: make(map[typePair]bool),
		groupMemo:     make(map[typePair]bool),
		typeVisiting:  make(map[typePair]bool),
		typeMemo:      make(map[typePair]bool),
	}
}

// typeIndicesEquivalent reports whether two type indices name equivalent type
// definitions in the checker's module pair.
func (c *checker) typeIndicesEquivalent(a uint32, b uint32) bool {
	startA, sizeA, posA := recGroupInfo(c.moduleA, a)
	startB, sizeB, posB := recGroupInfo(c.moduleB, b)
	if posA != posB || sizeA != sizeB {
		return false
	}
	if sizeA > 1 {
		groupKey := typePair{a: startA, b: startB}
		if eq, ok := c.groupMemo[groupKey]; ok {
			return eq
		}
		if c.groupVisiting[groupKey] {
			return true
		}
		c.groupVisiting[groupKey] = true
		defer delete(c.groupVisiting, groupKey)
		for i := uint32(0); i < sizeA; i++ {
			if !c.typeIndicesEquivalentInGroup(startA, startB, sizeA, startA+i, startB+i) {
				c.groupMemo[groupKey] = false
				return false
			}
		}
		c.groupMemo[groupKey] = true
		return true
	}
	return c.typeIndicesEquivalentBody(a, b)
}

// typeIndicesEquivalentInGroup compares two type indices while treating
// references into the current recursive group pair by relative position.
func (c *checker) typeIndicesEquivalentInGroup(groupA, groupB, groupSize, a, b uint32) bool {
	if a == b && c.moduleA == c.moduleB && groupA == groupB {
		return true
	}
	if c.outOfRange(a, b) {
		return false
	}
	key := typePair{a: a, b: b}
	if eq, ok := c.typeMemo[key]; ok {
		return eq
	}
	if c.typeVisiting[key] {
		return true
	}
	c.typeVisiting[key] = true
	defer delete(c.typeVisiting, key)

	ta := c.moduleA.Types[a]
	tb := c.moduleB.Types[b]
	if ta.SubType != tb.SubType || ta.Final != tb.Final || ta.Kind != tb.Kind || len(ta.SuperTypes) != len(tb.SuperTypes) {
		c.typeMemo[key] = false
		return false
	}
	for i := range ta.SuperTypes {
		if !c.typeIndexRefsEquivalentInGroup(groupA, groupB, groupSize, ta.SuperTypes[i], tb.SuperTypes[i]) {
			c.typeMemo[key] = false
			return false
		}
	}

	eq := c.typeDefsEquivalentInGroup(ta, tb, groupA, groupB, groupSize)
	c.typeMemo[key] = eq
	return eq
}

// typeIndexRefsEquivalentInGroup compares references that may point into the
// active recursive group pair.
func (c *checker) typeIndexRefsEquivalentInGroup(groupA, groupB, groupSize, a, b uint32) bool {
	inA := a >= groupA && a < groupA+groupSize
	inB := b >= groupB && b < groupB+groupSize
	if inA || inB {
		if !inA || !inB {
			return false
		}
		if a-groupA != b-groupB {
			return false
		}
		return c.typeIndicesEquivalentInGroup(groupA, groupB, groupSize, a, b)
	}
	return c.typeIndicesEquivalent(a, b)
}

// typeIndicesEquivalentBody compares two type definitions outside a
// multi-entry recursive group context.
func (c *checker) typeIndicesEquivalentBody(a uint32, b uint32) bool {
	if a == b && c.moduleA == c.moduleB {
		return true
	}
	if c.outOfRange(a, b) {
		return false
	}
	key := typePair{a: a, b: b}
	if eq, ok := c.typeMemo[key]; ok {
		return eq
	}
	if c.typeVisiting[key] {
		return true
	}
	c.typeVisiting[key] = true
	defer delete(c.typeVisiting, key)

	ta := c.moduleA.Types[a]
	tb := c.moduleB.Types[b]
	if ta.SubType != tb.SubType || ta.Final != tb.Final || ta.Kind != tb.Kind || len(ta.SuperTypes) != len(tb.SuperTypes) {
		c.typeMemo[key] = false
		return false
	}
	for i := range ta.SuperTypes {
		if !c.typeIndicesEquivalent(ta.SuperTypes[i], tb.SuperTypes[i]) {
			c.typeMemo[key] = false
			return false
		}
	}

	eq := c.typeDefsEquivalent(ta, tb)
	c.typeMemo[key] = eq
	return eq
}

// typeDefsEquivalentInGroup compares the body of two type definitions in a
// recursive-group context.
func (c *checker) typeDefsEquivalentInGroup(a wasmir.TypeDef, b wasmir.TypeDef, groupA, groupB, groupSize uint32) bool {
	switch a.Kind {
	case wasmir.TypeDefKindFunc:
		if len(a.Params) != len(b.Params) || len(a.Results) != len(b.Results) {
			return false
		}
		for i := range a.Params {
			if !c.valueTypesEquivalentInRecGroup(a.Params[i], b.Params[i], groupA, groupB, groupSize) {
				return false
			}
		}
		for i := range a.Results {
			if !c.valueTypesEquivalentInRecGroup(a.Results[i], b.Results[i], groupA, groupB, groupSize) {
				return false
			}
		}
		return true
	case wasmir.TypeDefKindStruct:
		if len(a.Fields) != len(b.Fields) {
			return false
		}
		for i := range a.Fields {
			if !c.fieldTypesEquivalentInRecGroup(a.Fields[i], b.Fields[i], groupA, groupB, groupSize) {
				return false
			}
		}
		return true
	case wasmir.TypeDefKindArray:
		return c.fieldTypesEquivalentInRecGroup(a.ElemField, b.ElemField, groupA, groupB, groupSize)
	default:
		return false
	}
}

// typeDefsEquivalent compares the body of two type definitions.
func (c *checker) typeDefsEquivalent(a wasmir.TypeDef, b wasmir.TypeDef) bool {
	switch a.Kind {
	case wasmir.TypeDefKindFunc:
		return c.valueTypeListsEquivalent(a.Params, b.Params) && c.valueTypeListsEquivalent(a.Results, b.Results)
	case wasmir.TypeDefKindStruct:
		if len(a.Fields) != len(b.Fields) {
			return false
		}
		for i := range a.Fields {
			if !c.fieldTypesEquivalent(a.Fields[i], b.Fields[i]) {
				return false
			}
		}
		return true
	case wasmir.TypeDefKindArray:
		return c.fieldTypesEquivalent(a.ElemField, b.ElemField)
	default:
		return false
	}
}

// valueTypeListsEquivalent compares two value-type slices pairwise.
func (c *checker) valueTypeListsEquivalent(a []wasmir.ValueType, b []wasmir.ValueType) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !c.valueTypesEquivalent(a[i], b[i]) {
			return false
		}
	}
	return true
}

// valueTypesEquivalentInRecGroup compares two value types in a recursive-group
// context.
func (c *checker) valueTypesEquivalentInRecGroup(a wasmir.ValueType, b wasmir.ValueType, groupA, groupB, groupSize uint32) bool {
	if a.Kind != b.Kind {
		return false
	}
	if a.Kind != wasmir.ValueKindRef {
		return a == b
	}
	if a.Nullable != b.Nullable {
		return false
	}
	if a.UsesTypeIndex() && b.UsesTypeIndex() {
		return c.typeIndexRefsEquivalentInGroup(groupA, groupB, groupSize, a.HeapType.TypeIndex, b.HeapType.TypeIndex)
	}
	return a.HeapType.Kind == b.HeapType.Kind
}

// valueTypesEquivalent compares two value types.
func (c *checker) valueTypesEquivalent(a wasmir.ValueType, b wasmir.ValueType) bool {
	if a.Kind != b.Kind {
		return false
	}
	if a.Kind != wasmir.ValueKindRef {
		return a == b
	}
	if a.Nullable != b.Nullable {
		return false
	}
	if a.UsesTypeIndex() && b.UsesTypeIndex() {
		return c.typeIndicesEquivalent(a.HeapType.TypeIndex, b.HeapType.TypeIndex)
	}
	return a.HeapType.Kind == b.HeapType.Kind
}

// fieldTypesEquivalentInRecGroup compares two struct or array field types in a
// recursive-group context.
func (c *checker) fieldTypesEquivalentInRecGroup(a wasmir.FieldType, b wasmir.FieldType, groupA, groupB, groupSize uint32) bool {
	if a.Mutable != b.Mutable || a.Packed != b.Packed {
		return false
	}
	if a.Packed != wasmir.PackedTypeNone {
		return true
	}
	return c.valueTypesEquivalentInRecGroup(a.Type, b.Type, groupA, groupB, groupSize)
}

// fieldTypesEquivalent compares two struct or array field types.
func (c *checker) fieldTypesEquivalent(a wasmir.FieldType, b wasmir.FieldType) bool {
	if a.Mutable != b.Mutable || a.Packed != b.Packed {
		return false
	}
	if a.Packed != wasmir.PackedTypeNone {
		return true
	}
	return c.valueTypesEquivalent(a.Type, b.Type)
}

// outOfRange reports whether either type index is outside its module.
func (c *checker) outOfRange(a uint32, b uint32) bool {
	return c.moduleA == nil || c.moduleB == nil || int(a) >= len(c.moduleA.Types) || int(b) >= len(c.moduleB.Types)
}

// recGroupInfo returns the recursive group containing idx. Non-grouped types
// behave like singleton groups.
func recGroupInfo(m *wasmir.Module, idx uint32) (start uint32, size uint32, pos uint32) {
	if m == nil || int(idx) >= len(m.Types) {
		return idx, 1, 0
	}
	for s := idx; ; {
		if groupSize := m.Types[s].RecGroupSize; groupSize > 0 && idx < s+groupSize {
			return s, groupSize, idx - s
		}
		if s == 0 {
			break
		}
		s--
	}
	return idx, 1, 0
}
