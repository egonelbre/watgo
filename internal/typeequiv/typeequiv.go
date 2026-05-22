// Package typeequiv compares WebAssembly type definitions structurally.
//
// WebAssembly type checks often cannot use Go equality on wasmir.TypeDef or
// wasmir.ValueType values directly. Indexed reference types may use different
// numeric type indices while still naming structurally equivalent recursive
// type definitions. For example, two modules can each define:
//
//	(type $callback (func (param i32) (result f32)))
//	(type $user (func (param (ref $callback))))
//
// The two $user types are equivalent across modules even though each module's
// `(ref $callback)` points into a different Module.Types slice. Types compares
// those graphs by structure, including recursive groups.
//
// Recursive groups compare by shape and relative position. If one module has:
//
//	(rec
//	  (type $a (func (param (ref $b))))
//	  (type $b (func (param (ref $a)))))
//
// and another module has the same two-entry cycle with different absolute type
// indices, corresponding entries are equivalent. Entries at different relative
// positions in the group are not equivalent.
//
// Most predicates in this package are intentionally about equivalence rather
// than subtyping. TypeSubtype is the exception: it layers declared-supertype
// reachability over structural equivalence for runtime and linking checks that
// accept subtypes.
package typeequiv

import "github.com/eliben/watgo/wasmir"

// Types reports whether the two type indices name equivalent type definitions
// in their respective modules.
//
// This accepts any type-definition kind: function, struct, or array. It returns
// false for missing modules, out-of-range indices, different type-definition
// kinds, different subtype wrappers, different declared supertypes, or
// different structural bodies. Use this for call_indirect/call_ref checks,
// wasm-to-wasm function import linking, validation-time type equivalence, and
// GC composite type equivalence.
func Types(moduleA *wasmir.Module, typeA uint32, moduleB *wasmir.Module, typeB uint32) bool {
	c := newChecker(moduleA, moduleB)
	return c.typeIndicesEquivalent(typeA, typeB)
}

// TypeSubtype reports whether sub names a type that is equivalent to, or a
// declared subtype of, super across the two module type spaces.
func TypeSubtype(moduleSub *wasmir.Module, sub uint32, moduleSuper *wasmir.Module, super uint32) bool {
	// WebAssembly subtyping is declared explicitly: a type can list one or more
	// direct supertypes. To answer a subtype query, this function walks the
	// transitive closure of those declared supertypes in moduleSub.
	//
	// The stack below is a small depth-first worklist of moduleSub type indices
	// still to inspect.
	//
	// seen prevents cycles or repeated work in recursive subtype graphs. Each
	// visited supertype is compared to moduleSuper.super with Types, because
	// the target supertype may live in a different module and therefore may
	// have a different numeric type index even when it is structurally
	// equivalent.
	if Types(moduleSub, sub, moduleSuper, super) {
		return true
	}
	if moduleSub == nil || int(sub) >= len(moduleSub.Types) || moduleSuper == nil || int(super) >= len(moduleSuper.Types) {
		return false
	}

	seen := map[uint32]bool{}
	stack := []uint32{sub}
	for len(stack) > 0 {
		idx := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[idx] || int(idx) >= len(moduleSub.Types) {
			continue
		}
		seen[idx] = true
		for _, declaredSuper := range moduleSub.Types[idx].SuperTypes {
			if Types(moduleSub, declaredSuper, moduleSuper, super) {
				return true
			}
			stack = append(stack, declaredSuper)
		}
	}
	return false
}

// FuncTypeAndSignature reports whether the function type named by typeIndex in
// module is equivalent to the explicit parameter and result type lists.
//
// This is for host functions, whose public API carries params/results directly
// rather than a source module and type index. Indexed reference types in params
// and results are interpreted in module's type index space. It returns false if
// module is nil, typeIndex is out of range, or typeIndex does not name a
// function type.
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
//
// Field mutability and packed representation must match exactly. For unpacked
// fields, the field value types are compared structurally, so indexed reference
// types are resolved against their respective modules. Packed fields carry no
// value type payload for this comparison once their packed kind matches.
func FieldTypes(moduleA *wasmir.Module, fieldA wasmir.FieldType, moduleB *wasmir.Module, fieldB wasmir.FieldType) bool {
	c := newChecker(moduleA, moduleB)
	return c.fieldTypesEquivalent(fieldA, fieldB)
}

// RecGroupInfo returns the recursive group containing idx. Non-grouped types
// behave like singleton groups.
//
// start is the first type index in the group, size is the number of entries in
// the group, and pos is idx's zero-based position within the group. If m is nil
// or idx is out of range, the result treats idx as a singleton group.
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
