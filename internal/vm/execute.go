package vm

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"slices"

	"github.com/eliben/watgo/wasmir"
)

const (
	// minInt32/maxInt32 and minInt64/maxInt64 are used when saturating
	// float-to-signed-integer conversions clamp out-of-range values.
	minInt32 = -1 << 31
	maxInt32 = 1<<31 - 1
	minInt64 = -1 << 63
	maxInt64 = 1<<63 - 1

	// The float thresholds below describe the half-open valid ranges for
	// trapping float-to-integer truncation:
	//
	//   - signed i32: [minInt32Float, two31Float)
	//   - unsigned i32: [0, two32Float)
	//   - signed i64: [minInt64Float, two63Float)
	//   - unsigned i64: [0, two64Float)
	//
	// They are also used as saturation cutoffs. Powers of two are exactly
	// representable in binary floating point, which makes these comparisons
	// stable across f32 and f64 inputs after promotion to float64.
	minInt32Float = -2147483648.0
	two31Float    = 2147483648.0
	two32Float    = 4294967296.0
	minInt64Float = -9223372036854775808.0
	two63Float    = 9223372036854775808.0
	two64Float    = 18446744073709551616.0

	// canonicalF32NaNBits/canonicalF64NaNBits are the WebAssembly canonical
	// NaN encodings used when a floating-point operation produces a new NaN.
	canonicalF32NaNBits = 0x7fc00000
	canonicalF64NaNBits = 0x7ff8000000000000
)

// instructionError adds interpreter location to low-level execution errors.
//
// The helpers below report compact errors such as stack underflow or operand
// type mismatch. Wrapping them with pc and opcode here makes those errors
// useful at the VM boundary, while keeping the no-error path free of formatting
// or allocation work.
type instructionError struct {
	pc   int
	kind wasmir.InstrKind
	err  error
}

// Error returns the execution error annotated with program counter and opcode.
func (e instructionError) Error() string {
	return fmt.Sprintf("pc %d %s: %v", e.pc, instrName(e.kind), e.err)
}

// Unwrap returns the low-level error that occurred while executing the
// instruction.
func (e instructionError) Unwrap() error {
	return e.err
}

// instructionErrorAt is called only on instruction failure paths.
func instructionErrorAt(pc int, kind wasmir.InstrKind, err error) error {
	return instructionError{pc: pc, kind: kind, err: err}
}

// Value is one runtime WebAssembly value.
type Value struct {
	// Type is the WebAssembly value type carried by this value.
	Type wasmir.ValueType

	// I32 is the payload for wasmir.ValueTypeI32 values.
	I32 int32

	// I64 is the payload for wasmir.ValueTypeI64 values.
	I64 int64

	// F32 is the payload for wasmir.ValueTypeF32 values.
	F32 float32

	// F64 is the payload for wasmir.ValueTypeF64 values.
	F64 float64

	// V128 is the payload for wasmir.ValueTypeV128 values.
	V128 [16]byte

	// Ref is the payload for reference-typed values.
	Ref Reference
}

// RefKind classifies the reference payload carried by a runtime Value.
type RefKind uint8

const (
	// RefKindNull is the null reference.
	RefKindNull RefKind = iota

	// RefKindFunc is a reference to a function in the instance function index
	// space.
	RefKindFunc

	// RefKindExtern is an opaque externref value supplied by the host.
	RefKindExtern
)

// Reference is one runtime reference value.
type Reference struct {
	// Kind identifies the concrete reference payload.
	Kind RefKind

	// FuncIndex is set when Kind is RefKindFunc.
	FuncIndex uint32

	// ExternID is set when Kind is RefKindExtern. The VM treats this as an
	// opaque identity token and never interprets it.
	ExternID uint64
}

// Resolver is the VM's narrow bridge to host-owned imports.
type Resolver interface {
	// CallFunc invokes an imported host function at index with already checked
	// arguments in parameter order.
	CallFunc(index uint32, args []Value) ([]Value, error)

	// Memory resolves an imported memory in the module's memory index space.
	Memory(index uint32, def wasmir.Memory) (*Memory, error)
}

// checkArgs verifies call argument count and value types.
func checkArgs(params []wasmir.ValueType, args []Value) error {
	if len(args) != len(params) {
		return fmt.Errorf("got %d arguments, want %d", len(args), len(params))
	}
	for i, want := range params {
		if !runtimeTypeMatches(args[i].Type, want) {
			return fmt.Errorf("argument %d has type %s, want %s", i, args[i].Type, want)
		}
	}
	return nil
}

// checkResults verifies result count and value types.
func checkResults(want []wasmir.ValueType, got []Value) error {
	if len(got) != len(want) {
		return fmt.Errorf("got %d results, want %d", len(got), len(want))
	}
	for i := range want {
		if !runtimeTypeMatches(got[i].Type, want[i]) {
			return fmt.Errorf("result %d has type %s, want %s", i, got[i].Type, want[i])
		}
	}
	return nil
}

// runtimeTypeMatches checks runtime value compatibility after validation has
// already enforced the full WebAssembly static typing rules.
func runtimeTypeMatches(got, want wasmir.ValueType) bool {
	if got == want {
		return true
	}
	return got.IsRef() && want.IsRef()
}

// executor is one active module-defined function frame.
type executor struct {
	// fn is the compiled function being interpreted by this frame.
	fn *function

	// ft is fn's validated WebAssembly signature.
	ft wasmir.TypeDef

	// inst is the VM-owned instantiated module state used by this frame.
	inst *Instance

	// pc is the current instruction index in fn.code. It is stored on the frame
	// so error wrapping and control-flow instructions can share the same
	// location state.
	pc int

	// locals is the function's local array: parameters first, followed by
	// zero-initialized non-parameter locals.
	locals []Value

	// stack is the operand stack for this frame.
	stack []Value
}

// executeFunction interprets one compiled module-defined function body.
func executeFunction(fn *function, ft wasmir.TypeDef, args []Value, inst *Instance) ([]Value, error) {
	if fn == nil {
		return nil, fmt.Errorf("defined function has no compiled code")
	}

	e := executor{
		fn:    fn,
		ft:    ft,
		inst:  inst,
		stack: make([]Value, 0),
	}
	if err := e.initLocals(args); err != nil {
		return nil, err
	}
	return e.run()
}

// initLocals builds the frame's local array from call arguments and declared
// non-parameter locals.
func (e *executor) initLocals(args []Value) error {
	e.locals = append([]Value{}, args...)
	for _, vt := range e.fn.locals {
		v, err := zeroValue(vt)
		if err != nil {
			return err
		}
		e.locals = append(e.locals, v)
	}
	return nil
}

// run interprets fn.code until it reaches return, the final end instruction, or
// an execution error.
func (e *executor) run() ([]Value, error) {
	for e.pc = 0; e.pc < len(e.fn.code); e.pc++ {
		ins := e.fn.code[e.pc]
		switch ins.kind {
		case wasmir.InstrBlock, wasmir.InstrLoop, wasmir.InstrNop:
		case wasmir.InstrUnreachable:
			return nil, e.instructionError(fmt.Errorf("unreachable executed"))
		case wasmir.InstrIf:
			// The condition has already been validated as i32. A true condition
			// enters the then arm. A false condition skips to the else marker
			// if present, or to the matching end otherwise.
			cond, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if cond == 0 {
				e.pc = ins.target
				continue
			}
		case wasmir.InstrElse:
			// Reaching else normally means the then arm completed without
			// branching. Skip the else arm.
			e.pc = ins.target
		case wasmir.InstrLocalGet:
			if int(ins.index) >= len(e.locals) {
				return nil, e.instructionError(fmt.Errorf("local index %d out of range", ins.index))
			}
			e.push(e.locals[ins.index])
		case wasmir.InstrLocalSet:
			if int(ins.index) >= len(e.locals) {
				return nil, e.instructionError(fmt.Errorf("local index %d out of range", ins.index))
			}
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !runtimeTypeMatches(v.Type, e.locals[ins.index].Type) {
				return nil, e.instructionError(fmt.Errorf("local.set %d got %s, want %s", ins.index, v.Type, e.locals[ins.index].Type))
			}
			e.locals[ins.index] = v
		case wasmir.InstrLocalTee:
			if int(ins.index) >= len(e.locals) {
				return nil, e.instructionError(fmt.Errorf("local index %d out of range", ins.index))
			}
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !runtimeTypeMatches(v.Type, e.locals[ins.index].Type) {
				return nil, e.instructionError(fmt.Errorf("local.tee %d got %s, want %s", ins.index, v.Type, e.locals[ins.index].Type))
			}
			e.locals[ins.index] = v
			e.push(v)
		case wasmir.InstrGlobalGet:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			v, err := e.inst.globalGetValue(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(v)
		case wasmir.InstrGlobalSet:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.globalSet(ins.index, v); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrI32Const:
			e.push(Value{Type: wasmir.ValueTypeI32, I32: int32(ins.bits)})
		case wasmir.InstrI32Load, wasmir.InstrI32Load8S, wasmir.InstrI32Load8U,
			wasmir.InstrI32Load16S, wasmir.InstrI32Load16U:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			size := memoryAccessSize(ins.kind)
			raw, err := e.inst.memoryLoad(ins.index, effective, size)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: extendI32Load(ins.kind, raw)})
		case wasmir.InstrI32Store, wasmir.InstrI32Store8, wasmir.InstrI32Store16:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			value, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.memoryStore(ins.index, effective, memoryAccessSize(ins.kind), uint64(uint32(value))); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrI64Load, wasmir.InstrI64Load8S, wasmir.InstrI64Load8U,
			wasmir.InstrI64Load16S, wasmir.InstrI64Load16U,
			wasmir.InstrI64Load32S, wasmir.InstrI64Load32U:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			size := memoryAccessSize(ins.kind)
			raw, err := e.inst.memoryLoad(ins.index, effective, size)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI64, I64: extendI64Load(ins.kind, raw)})
		case wasmir.InstrI64Store, wasmir.InstrI64Store8, wasmir.InstrI64Store16, wasmir.InstrI64Store32:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			value, err := e.popI64()
			if err != nil {
				return nil, e.instructionError(err)
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.memoryStore(ins.index, effective, memoryAccessSize(ins.kind), uint64(value)); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrF32Load:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			raw, err := e.inst.memoryLoad(ins.index, effective, 4)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeF32, F32: math.Float32frombits(uint32(raw))})
		case wasmir.InstrF32Store:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			value, err := e.popF32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.memoryStore(ins.index, effective, 4, uint64(math.Float32bits(value))); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrF64Load:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			raw, err := e.inst.memoryLoad(ins.index, effective, 8)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeF64, F64: math.Float64frombits(raw)})
		case wasmir.InstrF64Store:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			value, err := e.popF64()
			if err != nil {
				return nil, e.instructionError(err)
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.memoryStore(ins.index, effective, 8, math.Float64bits(value)); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrV128Load:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			bytes, err := e.inst.memory(ins.index, effective, 16)
			if err != nil {
				return nil, e.instructionError(err)
			}
			var value [16]byte
			copy(value[:], bytes)
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrV128Store:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			value, err := e.popV128()
			if err != nil {
				return nil, e.instructionError(err)
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			bytes, err := e.inst.memory(ins.index, effective, 16)
			if err != nil {
				return nil, e.instructionError(err)
			}
			copy(bytes, value[:])
		case wasmir.InstrV128Load8Splat, wasmir.InstrV128Load16Splat,
			wasmir.InstrV128Load32Splat, wasmir.InstrV128Load64Splat:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			effective, err := e.popMemoryAddress(ins.index, uint64(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			value, err := e.evalV128LoadSplat(ins.kind, ins.index, effective)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrMemorySize:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			pages, err := e.inst.memorySize(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.pushMemoryIndexResult(ins.index, pages); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrMemoryGrow:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			delta, err := e.popMemoryIndexOperand(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			oldPages, ok, err := e.inst.memoryGrow(ins.index, delta)
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !ok {
				if err := e.pushMemoryIndexResult(ins.index, ^uint64(0)); err != nil {
					return nil, e.instructionError(err)
				}
				continue
			}
			if err := e.pushMemoryIndexResult(ins.index, oldPages); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrMemoryCopy:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			size, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			src, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			dst, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.memoryCopy(ins.index, uint64(uint32(dst)), uint32(ins.bits), uint64(uint32(src)), uint64(uint32(size))); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrMemoryFill:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			size, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			value, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			dst, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.memoryFill(ins.index, uint64(uint32(dst)), uint64(uint32(size)), byte(value)); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrMemoryInit:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			size, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			src, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			dst, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.memoryInit(ins.index, uint32(ins.bits), uint64(uint32(dst)), uint64(uint32(src)), uint64(uint32(size))); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrDataDrop:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			if err := e.inst.dataDrop(ins.index); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrTableSize:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			size, err := e.inst.tableSize(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: int32(uint32(size))})
		case wasmir.InstrTableGet:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			elemIndex, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			v, err := e.inst.tableGet(ins.index, uint64(uint32(elemIndex)))
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(v)
		case wasmir.InstrTableSet:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			elemIndex, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.tableSet(ins.index, uint64(uint32(elemIndex)), v); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrTableGrow:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			delta, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			init, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			oldSize, ok, err := e.inst.tableGrow(ins.index, init, uint64(uint32(delta)))
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !ok {
				e.push(Value{Type: wasmir.ValueTypeI32, I32: -1})
				continue
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: int32(uint32(oldSize))})
		case wasmir.InstrTableFill:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			size, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			value, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			dst, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.tableFill(ins.index, uint64(uint32(dst)), uint64(uint32(size)), value); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrTableCopy:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			size, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			src, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			dst, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.tableCopy(ins.index, uint64(uint32(dst)), uint32(ins.bits), uint64(uint32(src)), uint64(uint32(size))); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrTableInit:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			size, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			src, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			dst, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := e.inst.tableInit(ins.index, uint32(ins.bits), uint64(uint32(dst)), uint64(uint32(src)), uint64(uint32(size))); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrElemDrop:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			if err := e.inst.elemDrop(ins.index); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrI32Add, wasmir.InstrI32Sub, wasmir.InstrI32Mul,
			wasmir.InstrI32DivS, wasmir.InstrI32DivU, wasmir.InstrI32RemS, wasmir.InstrI32RemU,
			wasmir.InstrI32And, wasmir.InstrI32Or, wasmir.InstrI32Xor,
			wasmir.InstrI32Shl, wasmir.InstrI32ShrS, wasmir.InstrI32ShrU,
			wasmir.InstrI32Rotl, wasmir.InstrI32Rotr,
			wasmir.InstrI32Eq, wasmir.InstrI32Ne,
			wasmir.InstrI32LtS, wasmir.InstrI32LtU, wasmir.InstrI32LeS, wasmir.InstrI32LeU,
			wasmir.InstrI32GtS, wasmir.InstrI32GtU, wasmir.InstrI32GeS, wasmir.InstrI32GeU:
			v, err := e.evalI32Binary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: v})
		case wasmir.InstrI32Eqz:
			v, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: boolI32(v == 0)})
		case wasmir.InstrI32Clz, wasmir.InstrI32Ctz, wasmir.InstrI32Popcnt,
			wasmir.InstrI32Extend8S, wasmir.InstrI32Extend16S:
			v, err := e.evalI32Unary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: v})
		case wasmir.InstrI64Const:
			e.push(Value{Type: wasmir.ValueTypeI64, I64: ins.bits})
		case wasmir.InstrI64Add, wasmir.InstrI64Sub, wasmir.InstrI64Mul,
			wasmir.InstrI64DivS, wasmir.InstrI64DivU, wasmir.InstrI64RemS, wasmir.InstrI64RemU,
			wasmir.InstrI64And, wasmir.InstrI64Or, wasmir.InstrI64Xor,
			wasmir.InstrI64Shl, wasmir.InstrI64ShrS, wasmir.InstrI64ShrU,
			wasmir.InstrI64Rotl, wasmir.InstrI64Rotr:
			v, err := e.evalI64Binary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI64, I64: v})
		case wasmir.InstrI64Eq, wasmir.InstrI64Ne,
			wasmir.InstrI64LtS, wasmir.InstrI64LtU, wasmir.InstrI64LeS, wasmir.InstrI64LeU,
			wasmir.InstrI64GtS, wasmir.InstrI64GtU, wasmir.InstrI64GeS, wasmir.InstrI64GeU:
			v, err := e.evalI64Compare(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: v})
		case wasmir.InstrI64Eqz:
			v, err := e.popI64()
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: boolI32(v == 0)})
		case wasmir.InstrI64Clz, wasmir.InstrI64Ctz, wasmir.InstrI64Popcnt,
			wasmir.InstrI64Extend8S, wasmir.InstrI64Extend16S, wasmir.InstrI64Extend32S:
			v, err := e.evalI64Unary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI64, I64: v})
		case wasmir.InstrI32WrapI64,
			wasmir.InstrI32TruncF32S, wasmir.InstrI32TruncF32U,
			wasmir.InstrI32TruncF64S, wasmir.InstrI32TruncF64U,
			wasmir.InstrI32TruncSatF32S, wasmir.InstrI32TruncSatF32U,
			wasmir.InstrI32TruncSatF64S, wasmir.InstrI32TruncSatF64U,
			wasmir.InstrI64ExtendI32S, wasmir.InstrI64ExtendI32U,
			wasmir.InstrI64TruncF32S, wasmir.InstrI64TruncF32U,
			wasmir.InstrI64TruncF64S, wasmir.InstrI64TruncF64U,
			wasmir.InstrI64TruncSatF32S, wasmir.InstrI64TruncSatF32U,
			wasmir.InstrI64TruncSatF64S, wasmir.InstrI64TruncSatF64U,
			wasmir.InstrF32ConvertI32S, wasmir.InstrF32ConvertI32U,
			wasmir.InstrF32ConvertI64S, wasmir.InstrF32ConvertI64U,
			wasmir.InstrF32DemoteF64,
			wasmir.InstrF64ConvertI32S, wasmir.InstrF64ConvertI32U,
			wasmir.InstrF64ConvertI64S, wasmir.InstrF64ConvertI64U,
			wasmir.InstrF64PromoteF32,
			wasmir.InstrI32ReinterpretF32, wasmir.InstrI64ReinterpretF64,
			wasmir.InstrF32ReinterpretI32, wasmir.InstrF64ReinterpretI64:
			v, err := e.evalConversion(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(v)
		case wasmir.InstrF32Const:
			e.push(Value{Type: wasmir.ValueTypeF32, F32: math.Float32frombits(uint32(ins.bits))})
		case wasmir.InstrF32Abs, wasmir.InstrF32Neg, wasmir.InstrF32Sqrt,
			wasmir.InstrF32Ceil, wasmir.InstrF32Floor, wasmir.InstrF32Trunc, wasmir.InstrF32Nearest:
			v, err := e.evalF32Unary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeF32, F32: v})
		case wasmir.InstrF32Add, wasmir.InstrF32Sub, wasmir.InstrF32Mul, wasmir.InstrF32Div,
			wasmir.InstrF32Min, wasmir.InstrF32Max, wasmir.InstrF32Copysign:
			v, err := e.evalF32Binary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeF32, F32: v})
		case wasmir.InstrF32Eq, wasmir.InstrF32Ne,
			wasmir.InstrF32Lt, wasmir.InstrF32Le, wasmir.InstrF32Gt, wasmir.InstrF32Ge:
			v, err := e.evalF32Compare(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: v})
		case wasmir.InstrF64Const:
			e.push(Value{Type: wasmir.ValueTypeF64, F64: math.Float64frombits(uint64(ins.bits))})
		case wasmir.InstrV128Const:
			if int(ins.index) >= len(e.fn.v128Consts) {
				return nil, e.instructionError(fmt.Errorf("v128.const index %d out of range", ins.index))
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: e.fn.v128Consts[ins.index]})
		case wasmir.InstrI8x16Splat, wasmir.InstrI16x8Splat, wasmir.InstrI32x4Splat,
			wasmir.InstrI64x2Splat, wasmir.InstrF32x4Splat, wasmir.InstrF64x2Splat:
			value, err := e.evalV128Splat(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrI8x16ExtractLaneS, wasmir.InstrI8x16ExtractLaneU,
			wasmir.InstrI16x8ExtractLaneS, wasmir.InstrI16x8ExtractLaneU,
			wasmir.InstrI32x4ExtractLane, wasmir.InstrI64x2ExtractLane,
			wasmir.InstrF32x4ExtractLane, wasmir.InstrF64x2ExtractLane:
			value, err := e.evalV128ExtractLane(ins.kind, ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(value)
		case wasmir.InstrI8x16ReplaceLane, wasmir.InstrI16x8ReplaceLane,
			wasmir.InstrI32x4ReplaceLane, wasmir.InstrI64x2ReplaceLane,
			wasmir.InstrF32x4ReplaceLane, wasmir.InstrF64x2ReplaceLane:
			value, err := e.evalV128ReplaceLane(ins.kind, ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrI8x16Shuffle:
			value, err := e.evalI8x16Shuffle(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrV128AnyTrue,
			wasmir.InstrI8x16AllTrue, wasmir.InstrI16x8AllTrue, wasmir.InstrI32x4AllTrue, wasmir.InstrI64x2AllTrue,
			wasmir.InstrI8x16Bitmask, wasmir.InstrI16x8Bitmask, wasmir.InstrI32x4Bitmask, wasmir.InstrI64x2Bitmask:
			value, err := e.evalV128Test(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: value})
		case wasmir.InstrV128Not,
			wasmir.InstrI8x16Neg, wasmir.InstrI16x8Neg, wasmir.InstrI32x4Neg, wasmir.InstrI64x2Neg,
			wasmir.InstrF32x4Abs, wasmir.InstrF32x4Neg, wasmir.InstrF32x4Sqrt,
			wasmir.InstrF64x2Abs, wasmir.InstrF64x2Neg, wasmir.InstrF64x2Sqrt,
			wasmir.InstrF32x4ConvertI32x4S, wasmir.InstrF32x4ConvertI32x4U,
			wasmir.InstrI32x4TruncSatF32x4S, wasmir.InstrI32x4TruncSatF32x4U:
			value, err := e.evalV128Unary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrI8x16Swizzle,
			wasmir.InstrV128And, wasmir.InstrV128AndNot, wasmir.InstrV128Or, wasmir.InstrV128Xor,
			wasmir.InstrI8x16Add, wasmir.InstrI8x16AddSatS, wasmir.InstrI8x16AddSatU,
			wasmir.InstrI8x16Sub, wasmir.InstrI8x16SubSatS, wasmir.InstrI8x16SubSatU,
			wasmir.InstrI16x8Add, wasmir.InstrI16x8AddSatS, wasmir.InstrI16x8AddSatU,
			wasmir.InstrI16x8Sub, wasmir.InstrI16x8SubSatS, wasmir.InstrI16x8SubSatU, wasmir.InstrI16x8Mul,
			wasmir.InstrI32x4Add, wasmir.InstrI32x4Sub, wasmir.InstrI32x4Mul,
			wasmir.InstrI64x2Add, wasmir.InstrI64x2Sub, wasmir.InstrI64x2Mul,
			wasmir.InstrF32x4Min, wasmir.InstrF32x4Max,
			wasmir.InstrF32x4Add, wasmir.InstrF32x4Sub, wasmir.InstrF32x4Div, wasmir.InstrF32x4Mul,
			wasmir.InstrF64x2Min, wasmir.InstrF64x2Max,
			wasmir.InstrF64x2Add, wasmir.InstrF64x2Sub, wasmir.InstrF64x2Div, wasmir.InstrF64x2Mul:
			value, err := e.evalV128Binary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrI8x16Shl, wasmir.InstrI8x16ShrS, wasmir.InstrI8x16ShrU,
			wasmir.InstrI16x8Shl, wasmir.InstrI16x8ShrS, wasmir.InstrI16x8ShrU,
			wasmir.InstrI32x4Shl, wasmir.InstrI32x4ShrS, wasmir.InstrI32x4ShrU,
			wasmir.InstrI64x2Shl, wasmir.InstrI64x2ShrS, wasmir.InstrI64x2ShrU:
			value, err := e.evalV128Shift(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrV128Bitselect:
			value, err := e.evalV128Bitselect()
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrI8x16Eq, wasmir.InstrI8x16Ne, wasmir.InstrI8x16LtS, wasmir.InstrI8x16LtU,
			wasmir.InstrI8x16GtS, wasmir.InstrI8x16GtU, wasmir.InstrI8x16LeS, wasmir.InstrI8x16LeU,
			wasmir.InstrI8x16GeS, wasmir.InstrI8x16GeU,
			wasmir.InstrI16x8Eq, wasmir.InstrI16x8Ne, wasmir.InstrI16x8LtS, wasmir.InstrI16x8LtU,
			wasmir.InstrI16x8GtS, wasmir.InstrI16x8GtU, wasmir.InstrI16x8LeS, wasmir.InstrI16x8LeU,
			wasmir.InstrI16x8GeS, wasmir.InstrI16x8GeU,
			wasmir.InstrI32x4Eq, wasmir.InstrI32x4Ne, wasmir.InstrI32x4LtS, wasmir.InstrI32x4LtU,
			wasmir.InstrI32x4GtS, wasmir.InstrI32x4GtU, wasmir.InstrI32x4LeS, wasmir.InstrI32x4LeU,
			wasmir.InstrI32x4GeS, wasmir.InstrI32x4GeU,
			wasmir.InstrI64x2Eq, wasmir.InstrI64x2Ne, wasmir.InstrI64x2LtS,
			wasmir.InstrI64x2GtS, wasmir.InstrI64x2LeS, wasmir.InstrI64x2GeS,
			wasmir.InstrF32x4Eq, wasmir.InstrF32x4Ne, wasmir.InstrF32x4Lt,
			wasmir.InstrF32x4Gt, wasmir.InstrF32x4Le, wasmir.InstrF32x4Ge,
			wasmir.InstrF64x2Eq, wasmir.InstrF64x2Ne, wasmir.InstrF64x2Lt,
			wasmir.InstrF64x2Gt, wasmir.InstrF64x2Le, wasmir.InstrF64x2Ge:
			value, err := e.evalV128Compare(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeV128, V128: value})
		case wasmir.InstrF64Abs, wasmir.InstrF64Neg, wasmir.InstrF64Sqrt,
			wasmir.InstrF64Ceil, wasmir.InstrF64Floor, wasmir.InstrF64Trunc, wasmir.InstrF64Nearest:
			v, err := e.evalF64Unary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeF64, F64: v})
		case wasmir.InstrF64Add, wasmir.InstrF64Sub, wasmir.InstrF64Mul, wasmir.InstrF64Div,
			wasmir.InstrF64Min, wasmir.InstrF64Max, wasmir.InstrF64Copysign:
			v, err := e.evalF64Binary(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeF64, F64: v})
		case wasmir.InstrF64Eq, wasmir.InstrF64Ne,
			wasmir.InstrF64Lt, wasmir.InstrF64Le, wasmir.InstrF64Gt, wasmir.InstrF64Ge:
			v, err := e.evalF64Compare(ins.kind)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: v})
		case wasmir.InstrDrop:
			if _, err := e.pop(); err != nil {
				return nil, e.instructionError(err)
			}
		case wasmir.InstrSelect:
			if ins.bits < 0 {
				cond, err := e.popI32()
				if err != nil {
					return nil, e.instructionError(err)
				}
				v2, err := e.pop()
				if err != nil {
					return nil, e.instructionError(err)
				}
				v1, err := e.pop()
				if err != nil {
					return nil, e.instructionError(err)
				}
				if v1.Type != v2.Type {
					return nil, e.instructionError(fmt.Errorf("select got %s and %s operands", v1.Type, v2.Type))
				}
				if cond != 0 {
					e.push(v1)
				} else {
					e.push(v2)
				}
				continue
			}
			v, err := e.evalTypedSelect(uint32(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.push(v)
		case wasmir.InstrRefNull:
			refTypeIndex := int(ins.index)
			if refTypeIndex >= len(e.fn.refTypes) {
				return nil, e.instructionError(fmt.Errorf("ref.null type index %d out of range", ins.index))
			}
			e.push(Value{Type: e.fn.refTypes[refTypeIndex], Ref: Reference{Kind: RefKindNull}})
		case wasmir.InstrRefFunc:
			if e.inst == nil {
				return nil, e.instructionError(fmt.Errorf("instance is nil"))
			}
			if _, err := e.inst.FuncType(ins.index); err != nil {
				return nil, e.instructionError(err)
			}
			e.push(Value{Type: wasmir.RefTypeFunc(false), Ref: Reference{Kind: RefKindFunc, FuncIndex: ins.index}})
		case wasmir.InstrRefIsNull:
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !v.Type.IsRef() {
				return nil, e.instructionError(fmt.Errorf("ref.is_null got %s operand", v.Type))
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: boolI32(v.Ref.Kind == RefKindNull)})
		case wasmir.InstrRefAsNonNull:
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !v.Type.IsRef() {
				return nil, e.instructionError(fmt.Errorf("ref.as_non_null got %s operand", v.Type))
			}
			if v.Ref.Kind == RefKindNull {
				return nil, e.instructionError(fmt.Errorf("ref.as_non_null to null reference"))
			}
			v.Type.Nullable = false
			e.push(v)
		case wasmir.InstrRefEq:
			v2, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			v1, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !v1.Type.IsRef() || !v2.Type.IsRef() {
				return nil, e.instructionError(fmt.Errorf("ref.eq got %s and %s operands", v1.Type, v2.Type))
			}
			e.push(Value{Type: wasmir.ValueTypeI32, I32: boolI32(refsEqual(v1.Ref, v2.Ref))})
		case wasmir.InstrExternConvertAny:
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !v.Type.IsRef() {
				return nil, e.instructionError(fmt.Errorf("extern.convert_any got %s operand", v.Type))
			}
			v.Type = wasmir.RefTypeExtern(v.Type.Nullable)
			e.push(v)
		case wasmir.InstrAnyConvertExtern:
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !v.Type.IsRef() {
				return nil, e.instructionError(fmt.Errorf("any.convert_extern got %s operand", v.Type))
			}
			v.Type = wasmir.RefTypeAny(v.Type.Nullable)
			e.push(v)
		case wasmir.InstrCall:
			results, err := e.callFunction(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.stack = append(e.stack, results...)
		case wasmir.InstrCallIndirect:
			results, err := e.callIndirectFunction(ins.index, uint32(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.stack = append(e.stack, results...)
		case wasmir.InstrCallRef:
			results, err := e.callRefFunction(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			e.stack = append(e.stack, results...)
		case wasmir.InstrReturnCall:
			results, err := e.callFunction(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := checkResults(e.ft.Results, results); err != nil {
				return nil, e.instructionError(err)
			}
			return results, nil
		case wasmir.InstrReturnCallIndirect:
			results, err := e.callIndirectFunction(ins.index, uint32(ins.bits))
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := checkResults(e.ft.Results, results); err != nil {
				return nil, e.instructionError(err)
			}
			return results, nil
		case wasmir.InstrReturnCallRef:
			results, err := e.callRefFunction(ins.index)
			if err != nil {
				return nil, e.instructionError(err)
			}
			if err := checkResults(e.ft.Results, results); err != nil {
				return nil, e.instructionError(err)
			}
			return results, nil
		case wasmir.InstrBr:
			e.pc = ins.target
		case wasmir.InstrBrIf:
			// br_if consumes only the condition. Any branch result values are
			// already below it on the operand stack and are left there for the
			// target block's end to consume.
			cond, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if cond != 0 {
				e.pc = ins.target
			}
		case wasmir.InstrBrOnNull:
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !v.Type.IsRef() {
				return nil, e.instructionError(fmt.Errorf("br_on_null got %s operand", v.Type))
			}
			if v.Ref.Kind == RefKindNull {
				e.pc = ins.target
				continue
			}
			v.Type.Nullable = false
			e.push(v)
		case wasmir.InstrBrOnNonNull:
			v, err := e.pop()
			if err != nil {
				return nil, e.instructionError(err)
			}
			if !v.Type.IsRef() {
				return nil, e.instructionError(fmt.Errorf("br_on_non_null got %s operand", v.Type))
			}
			if v.Ref.Kind != RefKindNull {
				v.Type.Nullable = false
				e.push(v)
				e.pc = ins.target
				continue
			}
		case wasmir.InstrBrTable:
			// br_table consumes only the i32 selector. Branch result values, if
			// any, are already below it on the operand stack and are left there
			// for the selected target block's end to consume.
			selector, err := e.popI32()
			if err != nil {
				return nil, e.instructionError(err)
			}
			tableIndex := int(ins.index)
			if tableIndex >= len(e.fn.branchTables) {
				return nil, e.instructionError(fmt.Errorf("br_table index %d out of range", ins.index))
			}
			targets := e.fn.branchTables[tableIndex]
			if len(targets) == 0 {
				return nil, e.instructionError(fmt.Errorf("br_table has no default target"))
			}
			targetIndex := uint32(selector)
			defaultIndex := len(targets) - 1
			if uint64(targetIndex) < uint64(defaultIndex) {
				e.pc = targets[int(targetIndex)]
			} else {
				e.pc = targets[defaultIndex]
			}
		case wasmir.InstrReturn:
			results, err := e.popResults(e.ft.Results)
			if err != nil {
				return nil, e.instructionError(err)
			}
			return results, nil
		case wasmir.InstrEnd:
			if e.pc != len(e.fn.code)-1 {
				// Non-final end closes structured control. Branch targets skip
				// over it, and ordinary fallthrough can treat it as a no-op
				// because validation has already established the operand stack
				// contract.
				continue
			}
			results, err := e.popResults(e.ft.Results)
			if err != nil {
				return nil, e.instructionError(err)
			}
			return results, nil
		default:
			return nil, e.instructionError(fmt.Errorf("unsupported instruction"))
		}
	}
	return nil, fmt.Errorf("function ended without end")
}

// zeroValue constructs the default local value for a supported value type.
func zeroValue(vt wasmir.ValueType) (Value, error) {
	switch vt {
	case wasmir.ValueTypeI32:
		return Value{Type: wasmir.ValueTypeI32}, nil
	case wasmir.ValueTypeI64:
		return Value{Type: wasmir.ValueTypeI64}, nil
	case wasmir.ValueTypeF32:
		return Value{Type: wasmir.ValueTypeF32}, nil
	case wasmir.ValueTypeF64:
		return Value{Type: wasmir.ValueTypeF64}, nil
	case wasmir.ValueTypeV128:
		return Value{Type: wasmir.ValueTypeV128}, nil
	default:
		if vt.IsRef() {
			return Value{Type: vt, Ref: Reference{Kind: RefKindNull}}, nil
		}
		return Value{}, fmt.Errorf("unsupported local type %s", vt)
	}
}

// instructionError wraps err with the current program counter and opcode.
func (e *executor) instructionError(err error) error {
	return instructionErrorAt(e.pc, e.fn.code[e.pc].kind, err)
}

// push appends v to the operand stack.
func (e *executor) push(v Value) {
	e.stack = append(e.stack, v)
}

// pop removes and returns the top operand stack value.
func (e *executor) pop() (Value, error) {
	if len(e.stack) == 0 {
		return Value{}, fmt.Errorf("operand stack underflow")
	}
	v := e.stack[len(e.stack)-1]
	e.stack = e.stack[:len(e.stack)-1]
	return v, nil
}

// callFunction pops arguments for the target function and invokes it through
// the instance dispatcher.
func (e *executor) callFunction(index uint32) ([]Value, error) {
	if e.inst == nil {
		return nil, fmt.Errorf("instance is nil")
	}
	calleeType, err := e.inst.FuncType(index)
	if err != nil {
		return nil, err
	}
	callArgs, err := e.popArgs(calleeType.Params)
	if err != nil {
		return nil, err
	}
	return e.inst.CallFunc(index, callArgs)
}

// callIndirectFunction resolves a table element to a function reference,
// checks that its runtime type matches callTypeIndex, and invokes it.
func (e *executor) callIndirectFunction(tableIndex uint32, callTypeIndex uint32) ([]Value, error) {
	if e.inst == nil {
		return nil, fmt.Errorf("instance is nil")
	}
	elemIndex, err := e.popI32()
	if err != nil {
		return nil, err
	}
	ref, err := e.inst.tableGet(tableIndex, uint64(uint32(elemIndex)))
	if err != nil {
		return nil, err
	}
	if !ref.Type.IsRef() {
		return nil, fmt.Errorf("call_indirect table element has type %s", ref.Type)
	}
	if ref.Ref.Kind == RefKindNull {
		return nil, fmt.Errorf("indirect call to null reference")
	}
	if ref.Ref.Kind != RefKindFunc {
		return nil, fmt.Errorf("indirect call to non-function reference")
	}
	if err := e.checkFunctionReferenceType(ref.Ref.FuncIndex, callTypeIndex); err != nil {
		return nil, err
	}
	return e.callFunction(ref.Ref.FuncIndex)
}

// callRefFunction pops a function reference operand, checks its runtime type,
// and invokes it.
func (e *executor) callRefFunction(callTypeIndex uint32) ([]Value, error) {
	if e.inst == nil {
		return nil, fmt.Errorf("instance is nil")
	}
	ref, err := e.pop()
	if err != nil {
		return nil, err
	}
	if !ref.Type.IsRef() {
		return nil, fmt.Errorf("call_ref got %s operand", ref.Type)
	}
	if ref.Ref.Kind == RefKindNull {
		return nil, fmt.Errorf("call_ref to null reference")
	}
	if ref.Ref.Kind != RefKindFunc {
		return nil, fmt.Errorf("call_ref to non-function reference")
	}
	if err := e.checkFunctionReferenceType(ref.Ref.FuncIndex, callTypeIndex); err != nil {
		return nil, err
	}
	return e.callFunction(ref.Ref.FuncIndex)
}

// checkFunctionReferenceType verifies the runtime type check for a resolved
// function reference.
func (e *executor) checkFunctionReferenceType(funcIndex uint32, callTypeIndex uint32) error {
	want, err := e.inst.callType(callTypeIndex)
	if err != nil {
		return err
	}
	if want.Kind != wasmir.TypeDefKindFunc {
		return fmt.Errorf("type index %d is not a function type", callTypeIndex)
	}
	got, err := e.inst.FuncType(funcIndex)
	if err != nil {
		return err
	}
	if got.Kind != wasmir.TypeDefKindFunc || !slices.Equal(got.Params, want.Params) || !slices.Equal(got.Results, want.Results) {
		return fmt.Errorf("indirect call type mismatch")
	}
	return nil
}

// evalTypedSelect pops a typed select's operands and returns the selected value
// normalized to the explicit select result type.
func (e *executor) evalTypedSelect(selectTypeIndex uint32) (Value, error) {
	if int(selectTypeIndex) >= len(e.fn.selectTypes) {
		return Value{}, fmt.Errorf("select type index %d out of range", selectTypeIndex)
	}
	want := e.fn.selectTypes[selectTypeIndex]
	cond, err := e.popI32()
	if err != nil {
		return Value{}, err
	}
	v2, err := e.pop()
	if err != nil {
		return Value{}, err
	}
	v1, err := e.pop()
	if err != nil {
		return Value{}, err
	}
	if !runtimeTypeMatches(v1.Type, want) || !runtimeTypeMatches(v2.Type, want) {
		return Value{}, fmt.Errorf("select expects operands of type %s", want)
	}
	if cond != 0 {
		v1.Type = want
		return v1, nil
	}
	v2.Type = want
	return v2, nil
}

// refsEqual reports whether two runtime references have the same identity.
func refsEqual(a Reference, b Reference) bool {
	if a.Kind != b.Kind {
		return false
	}
	switch a.Kind {
	case RefKindNull:
		return true
	case RefKindFunc:
		return a.FuncIndex == b.FuncIndex
	case RefKindExtern:
		return a.ExternID == b.ExternID
	default:
		return false
	}
}

// evalI64Binary pops two i64 operands and evaluates an i64 binary instruction.
func (e *executor) evalI64Binary(kind wasmir.InstrKind) (int64, error) {
	rhs, err := e.popI64()
	if err != nil {
		return 0, err
	}
	lhs, err := e.popI64()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrI64Add:
		return lhs + rhs, nil
	case wasmir.InstrI64Sub:
		return lhs - rhs, nil
	case wasmir.InstrI64Mul:
		return lhs * rhs, nil
	case wasmir.InstrI64DivS:
		if rhs == 0 {
			return 0, fmt.Errorf("integer divide by zero")
		}
		if lhs == minInt64 && rhs == -1 {
			return 0, fmt.Errorf("integer overflow")
		}
		return lhs / rhs, nil
	case wasmir.InstrI64DivU:
		if rhs == 0 {
			return 0, fmt.Errorf("integer divide by zero")
		}
		return int64(uint64(lhs) / uint64(rhs)), nil
	case wasmir.InstrI64RemS:
		if rhs == 0 {
			return 0, fmt.Errorf("integer divide by zero")
		}
		return lhs % rhs, nil
	case wasmir.InstrI64RemU:
		if rhs == 0 {
			return 0, fmt.Errorf("integer divide by zero")
		}
		return int64(uint64(lhs) % uint64(rhs)), nil
	case wasmir.InstrI64And:
		return lhs & rhs, nil
	case wasmir.InstrI64Or:
		return lhs | rhs, nil
	case wasmir.InstrI64Xor:
		return lhs ^ rhs, nil
	case wasmir.InstrI64Shl:
		return int64(uint64(lhs) << (uint64(rhs) & 63)), nil
	case wasmir.InstrI64ShrS:
		return lhs >> (uint64(rhs) & 63), nil
	case wasmir.InstrI64ShrU:
		return int64(uint64(lhs) >> (uint64(rhs) & 63)), nil
	case wasmir.InstrI64Rotl:
		return int64(bits.RotateLeft64(uint64(lhs), int(uint64(rhs)&63))), nil
	case wasmir.InstrI64Rotr:
		return int64(bits.RotateLeft64(uint64(lhs), -int(uint64(rhs)&63))), nil
	default:
		return 0, fmt.Errorf("unsupported i64 binary instruction %s", instrName(kind))
	}
}

// evalI64Unary pops one i64 operand and evaluates an i64 unary instruction.
func (e *executor) evalI64Unary(kind wasmir.InstrKind) (int64, error) {
	v, err := e.popI64()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrI64Clz:
		return int64(bits.LeadingZeros64(uint64(v))), nil
	case wasmir.InstrI64Ctz:
		return int64(bits.TrailingZeros64(uint64(v))), nil
	case wasmir.InstrI64Popcnt:
		return int64(bits.OnesCount64(uint64(v))), nil
	case wasmir.InstrI64Extend8S:
		return int64(int8(v)), nil
	case wasmir.InstrI64Extend16S:
		return int64(int16(v)), nil
	case wasmir.InstrI64Extend32S:
		return int64(int32(v)), nil
	default:
		return 0, fmt.Errorf("unsupported i64 unary instruction %s", instrName(kind))
	}
}

// evalI64Compare pops two i64 operands and evaluates an i64 comparison,
// returning the WebAssembly i32 boolean result.
func (e *executor) evalI64Compare(kind wasmir.InstrKind) (int32, error) {
	rhs, err := e.popI64()
	if err != nil {
		return 0, err
	}
	lhs, err := e.popI64()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrI64Eq:
		return boolI32(lhs == rhs), nil
	case wasmir.InstrI64Ne:
		return boolI32(lhs != rhs), nil
	case wasmir.InstrI64LtS:
		return boolI32(lhs < rhs), nil
	case wasmir.InstrI64LtU:
		return boolI32(uint64(lhs) < uint64(rhs)), nil
	case wasmir.InstrI64LeS:
		return boolI32(lhs <= rhs), nil
	case wasmir.InstrI64LeU:
		return boolI32(uint64(lhs) <= uint64(rhs)), nil
	case wasmir.InstrI64GtS:
		return boolI32(lhs > rhs), nil
	case wasmir.InstrI64GtU:
		return boolI32(uint64(lhs) > uint64(rhs)), nil
	case wasmir.InstrI64GeS:
		return boolI32(lhs >= rhs), nil
	case wasmir.InstrI64GeU:
		return boolI32(uint64(lhs) >= uint64(rhs)), nil
	default:
		return 0, fmt.Errorf("unsupported i64 comparison instruction %s", instrName(kind))
	}
}

// evalF32Binary pops two f32 operands and evaluates an f32 arithmetic
// instruction.
func (e *executor) evalF32Binary(kind wasmir.InstrKind) (float32, error) {
	rhs, err := e.popF32()
	if err != nil {
		return 0, err
	}
	lhs, err := e.popF32()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrF32Add, wasmir.InstrF32Sub, wasmir.InstrF32Mul,
		wasmir.InstrF32Div, wasmir.InstrF32Min, wasmir.InstrF32Max:
		return math.Float32frombits(binaryF32(kind, lhs, rhs)), nil
	case wasmir.InstrF32Copysign:
		return math.Float32frombits((math.Float32bits(lhs) &^ (1 << 31)) | (math.Float32bits(rhs) & (1 << 31))), nil
	default:
		return 0, fmt.Errorf("unsupported f32 binary instruction %s", instrName(kind))
	}
}

// evalF32Unary pops one f32 operand and evaluates an f32 unary instruction.
func (e *executor) evalF32Unary(kind wasmir.InstrKind) (float32, error) {
	v, err := e.popF32()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrF32Abs:
		return math.Float32frombits(math.Float32bits(v) &^ (1 << 31)), nil
	case wasmir.InstrF32Neg:
		return math.Float32frombits(math.Float32bits(v) ^ (1 << 31)), nil
	case wasmir.InstrF32Sqrt:
		return float32(math.Sqrt(float64(v))), nil
	case wasmir.InstrF32Ceil:
		return float32(math.Ceil(float64(v))), nil
	case wasmir.InstrF32Floor:
		return float32(math.Floor(float64(v))), nil
	case wasmir.InstrF32Trunc:
		return float32(math.Trunc(float64(v))), nil
	case wasmir.InstrF32Nearest:
		return float32(math.RoundToEven(float64(v))), nil
	default:
		return 0, fmt.Errorf("unsupported f32 unary instruction %s", instrName(kind))
	}
}

// evalF32Compare pops two f32 operands and evaluates an f32 comparison,
// returning the WebAssembly i32 boolean result.
func (e *executor) evalF32Compare(kind wasmir.InstrKind) (int32, error) {
	rhs, err := e.popF32()
	if err != nil {
		return 0, err
	}
	lhs, err := e.popF32()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrF32Eq:
		return boolI32(lhs == rhs), nil
	case wasmir.InstrF32Ne:
		return boolI32(lhs != rhs), nil
	case wasmir.InstrF32Lt:
		return boolI32(lhs < rhs), nil
	case wasmir.InstrF32Le:
		return boolI32(lhs <= rhs), nil
	case wasmir.InstrF32Gt:
		return boolI32(lhs > rhs), nil
	case wasmir.InstrF32Ge:
		return boolI32(lhs >= rhs), nil
	default:
		return 0, fmt.Errorf("unsupported f32 comparison instruction %s", instrName(kind))
	}
}

// evalF64Unary pops one f64 operand and evaluates an f64 unary instruction.
func (e *executor) evalF64Unary(kind wasmir.InstrKind) (float64, error) {
	v, err := e.popF64()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrF64Abs:
		return math.Float64frombits(math.Float64bits(v) &^ (1 << 63)), nil
	case wasmir.InstrF64Neg:
		return math.Float64frombits(math.Float64bits(v) ^ (1 << 63)), nil
	case wasmir.InstrF64Sqrt:
		return math.Sqrt(v), nil
	case wasmir.InstrF64Ceil:
		return math.Ceil(v), nil
	case wasmir.InstrF64Floor:
		return math.Floor(v), nil
	case wasmir.InstrF64Trunc:
		return math.Trunc(v), nil
	case wasmir.InstrF64Nearest:
		return math.RoundToEven(v), nil
	default:
		return 0, fmt.Errorf("unsupported f64 unary instruction %s", instrName(kind))
	}
}

// evalF64Binary pops two f64 operands and evaluates an f64 arithmetic
// instruction.
func (e *executor) evalF64Binary(kind wasmir.InstrKind) (float64, error) {
	rhs, err := e.popF64()
	if err != nil {
		return 0, err
	}
	lhs, err := e.popF64()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrF64Add, wasmir.InstrF64Sub, wasmir.InstrF64Mul,
		wasmir.InstrF64Div, wasmir.InstrF64Min, wasmir.InstrF64Max:
		return math.Float64frombits(binaryF64(kind, lhs, rhs)), nil
	case wasmir.InstrF64Copysign:
		return math.Float64frombits((math.Float64bits(lhs) &^ (1 << 63)) | (math.Float64bits(rhs) & (1 << 63))), nil
	default:
		return 0, fmt.Errorf("unsupported f64 binary instruction %s", instrName(kind))
	}
}

// evalF64Compare pops two f64 operands and evaluates an f64 comparison,
// returning the WebAssembly i32 boolean result.
func (e *executor) evalF64Compare(kind wasmir.InstrKind) (int32, error) {
	rhs, err := e.popF64()
	if err != nil {
		return 0, err
	}
	lhs, err := e.popF64()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrF64Eq:
		return boolI32(lhs == rhs), nil
	case wasmir.InstrF64Ne:
		return boolI32(lhs != rhs), nil
	case wasmir.InstrF64Lt:
		return boolI32(lhs < rhs), nil
	case wasmir.InstrF64Le:
		return boolI32(lhs <= rhs), nil
	case wasmir.InstrF64Gt:
		return boolI32(lhs > rhs), nil
	case wasmir.InstrF64Ge:
		return boolI32(lhs >= rhs), nil
	default:
		return 0, fmt.Errorf("unsupported f64 comparison instruction %s", instrName(kind))
	}
}

// evalI32Binary pops two i32 operands and evaluates an i32 binary instruction.
func (e *executor) evalI32Binary(kind wasmir.InstrKind) (int32, error) {
	rhs, err := e.popI32()
	if err != nil {
		return 0, err
	}
	lhs, err := e.popI32()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrI32Add:
		return lhs + rhs, nil
	case wasmir.InstrI32Sub:
		return lhs - rhs, nil
	case wasmir.InstrI32Mul:
		return lhs * rhs, nil
	case wasmir.InstrI32DivS:
		if rhs == 0 {
			return 0, fmt.Errorf("integer divide by zero")
		}
		if lhs == minInt32 && rhs == -1 {
			return 0, fmt.Errorf("integer overflow")
		}
		return lhs / rhs, nil
	case wasmir.InstrI32DivU:
		if rhs == 0 {
			return 0, fmt.Errorf("integer divide by zero")
		}
		return int32(uint32(lhs) / uint32(rhs)), nil
	case wasmir.InstrI32RemS:
		if rhs == 0 {
			return 0, fmt.Errorf("integer divide by zero")
		}
		return lhs % rhs, nil
	case wasmir.InstrI32RemU:
		if rhs == 0 {
			return 0, fmt.Errorf("integer divide by zero")
		}
		return int32(uint32(lhs) % uint32(rhs)), nil
	case wasmir.InstrI32And:
		return lhs & rhs, nil
	case wasmir.InstrI32Or:
		return lhs | rhs, nil
	case wasmir.InstrI32Xor:
		return lhs ^ rhs, nil
	case wasmir.InstrI32Shl:
		return int32(uint32(lhs) << (uint32(rhs) & 31)), nil
	case wasmir.InstrI32ShrS:
		return lhs >> (uint32(rhs) & 31), nil
	case wasmir.InstrI32ShrU:
		return int32(uint32(lhs) >> (uint32(rhs) & 31)), nil
	case wasmir.InstrI32Rotl:
		return int32(bits.RotateLeft32(uint32(lhs), int(uint32(rhs)&31))), nil
	case wasmir.InstrI32Rotr:
		return int32(bits.RotateLeft32(uint32(lhs), -int(uint32(rhs)&31))), nil
	case wasmir.InstrI32Eq:
		return boolI32(lhs == rhs), nil
	case wasmir.InstrI32Ne:
		return boolI32(lhs != rhs), nil
	case wasmir.InstrI32LtS:
		return boolI32(lhs < rhs), nil
	case wasmir.InstrI32LtU:
		return boolI32(uint32(lhs) < uint32(rhs)), nil
	case wasmir.InstrI32LeS:
		return boolI32(lhs <= rhs), nil
	case wasmir.InstrI32LeU:
		return boolI32(uint32(lhs) <= uint32(rhs)), nil
	case wasmir.InstrI32GtS:
		return boolI32(lhs > rhs), nil
	case wasmir.InstrI32GtU:
		return boolI32(uint32(lhs) > uint32(rhs)), nil
	case wasmir.InstrI32GeS:
		return boolI32(lhs >= rhs), nil
	case wasmir.InstrI32GeU:
		return boolI32(uint32(lhs) >= uint32(rhs)), nil
	default:
		return 0, fmt.Errorf("unsupported i32 binary instruction %s", instrName(kind))
	}
}

// evalI32Unary pops one i32 operand and evaluates an i32 unary instruction.
func (e *executor) evalI32Unary(kind wasmir.InstrKind) (int32, error) {
	v, err := e.popI32()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrI32Clz:
		return int32(bits.LeadingZeros32(uint32(v))), nil
	case wasmir.InstrI32Ctz:
		return int32(bits.TrailingZeros32(uint32(v))), nil
	case wasmir.InstrI32Popcnt:
		return int32(bits.OnesCount32(uint32(v))), nil
	case wasmir.InstrI32Extend8S:
		return int32(int8(v)), nil
	case wasmir.InstrI32Extend16S:
		return int32(int16(v)), nil
	default:
		return 0, fmt.Errorf("unsupported i32 unary instruction %s", instrName(kind))
	}
}

// evalConversion pops the source operand for a numeric conversion or
// reinterpret instruction and returns the converted runtime value.
func (e *executor) evalConversion(kind wasmir.InstrKind) (Value, error) {
	switch kind {
	case wasmir.InstrI32WrapI64:
		v, err := e.popI64()
		return Value{Type: wasmir.ValueTypeI32, I32: int32(v)}, err
	case wasmir.InstrI32TruncF32S:
		v, err := e.popF32()
		if err != nil {
			return Value{}, err
		}
		out, err := truncFloatToI32S(float64(v))
		return Value{Type: wasmir.ValueTypeI32, I32: out}, err
	case wasmir.InstrI32TruncF32U:
		v, err := e.popF32()
		if err != nil {
			return Value{}, err
		}
		out, err := truncFloatToI32U(float64(v))
		return Value{Type: wasmir.ValueTypeI32, I32: out}, err
	case wasmir.InstrI32TruncF64S:
		v, err := e.popF64()
		if err != nil {
			return Value{}, err
		}
		out, err := truncFloatToI32S(v)
		return Value{Type: wasmir.ValueTypeI32, I32: out}, err
	case wasmir.InstrI32TruncF64U:
		v, err := e.popF64()
		if err != nil {
			return Value{}, err
		}
		out, err := truncFloatToI32U(v)
		return Value{Type: wasmir.ValueTypeI32, I32: out}, err
	case wasmir.InstrI32TruncSatF32S:
		v, err := e.popF32()
		return Value{Type: wasmir.ValueTypeI32, I32: truncSatFloatToI32S(float64(v))}, err
	case wasmir.InstrI32TruncSatF32U:
		v, err := e.popF32()
		return Value{Type: wasmir.ValueTypeI32, I32: truncSatFloatToI32U(float64(v))}, err
	case wasmir.InstrI32TruncSatF64S:
		v, err := e.popF64()
		return Value{Type: wasmir.ValueTypeI32, I32: truncSatFloatToI32S(v)}, err
	case wasmir.InstrI32TruncSatF64U:
		v, err := e.popF64()
		return Value{Type: wasmir.ValueTypeI32, I32: truncSatFloatToI32U(v)}, err
	case wasmir.InstrI64ExtendI32S:
		v, err := e.popI32()
		return Value{Type: wasmir.ValueTypeI64, I64: int64(v)}, err
	case wasmir.InstrI64ExtendI32U:
		v, err := e.popI32()
		return Value{Type: wasmir.ValueTypeI64, I64: int64(uint32(v))}, err
	case wasmir.InstrI64TruncF32S:
		v, err := e.popF32()
		if err != nil {
			return Value{}, err
		}
		out, err := truncFloatToI64S(float64(v))
		return Value{Type: wasmir.ValueTypeI64, I64: out}, err
	case wasmir.InstrI64TruncF32U:
		v, err := e.popF32()
		if err != nil {
			return Value{}, err
		}
		out, err := truncFloatToI64U(float64(v))
		return Value{Type: wasmir.ValueTypeI64, I64: out}, err
	case wasmir.InstrI64TruncF64S:
		v, err := e.popF64()
		if err != nil {
			return Value{}, err
		}
		out, err := truncFloatToI64S(v)
		return Value{Type: wasmir.ValueTypeI64, I64: out}, err
	case wasmir.InstrI64TruncF64U:
		v, err := e.popF64()
		if err != nil {
			return Value{}, err
		}
		out, err := truncFloatToI64U(v)
		return Value{Type: wasmir.ValueTypeI64, I64: out}, err
	case wasmir.InstrI64TruncSatF32S:
		v, err := e.popF32()
		return Value{Type: wasmir.ValueTypeI64, I64: truncSatFloatToI64S(float64(v))}, err
	case wasmir.InstrI64TruncSatF32U:
		v, err := e.popF32()
		return Value{Type: wasmir.ValueTypeI64, I64: truncSatFloatToI64U(float64(v))}, err
	case wasmir.InstrI64TruncSatF64S:
		v, err := e.popF64()
		return Value{Type: wasmir.ValueTypeI64, I64: truncSatFloatToI64S(v)}, err
	case wasmir.InstrI64TruncSatF64U:
		v, err := e.popF64()
		return Value{Type: wasmir.ValueTypeI64, I64: truncSatFloatToI64U(v)}, err
	case wasmir.InstrF32ConvertI32S:
		v, err := e.popI32()
		return Value{Type: wasmir.ValueTypeF32, F32: float32(v)}, err
	case wasmir.InstrF32ConvertI32U:
		v, err := e.popI32()
		return Value{Type: wasmir.ValueTypeF32, F32: float32(uint32(v))}, err
	case wasmir.InstrF32ConvertI64S:
		v, err := e.popI64()
		return Value{Type: wasmir.ValueTypeF32, F32: float32(v)}, err
	case wasmir.InstrF32ConvertI64U:
		v, err := e.popI64()
		return Value{Type: wasmir.ValueTypeF32, F32: float32(uint64(v))}, err
	case wasmir.InstrF32DemoteF64:
		v, err := e.popF64()
		return Value{Type: wasmir.ValueTypeF32, F32: float32(v)}, err
	case wasmir.InstrF64ConvertI32S:
		v, err := e.popI32()
		return Value{Type: wasmir.ValueTypeF64, F64: float64(v)}, err
	case wasmir.InstrF64ConvertI32U:
		v, err := e.popI32()
		return Value{Type: wasmir.ValueTypeF64, F64: float64(uint32(v))}, err
	case wasmir.InstrF64ConvertI64S:
		v, err := e.popI64()
		return Value{Type: wasmir.ValueTypeF64, F64: float64(v)}, err
	case wasmir.InstrF64ConvertI64U:
		v, err := e.popI64()
		return Value{Type: wasmir.ValueTypeF64, F64: float64(uint64(v))}, err
	case wasmir.InstrF64PromoteF32:
		v, err := e.popF32()
		return Value{Type: wasmir.ValueTypeF64, F64: float64(v)}, err
	case wasmir.InstrI32ReinterpretF32:
		v, err := e.popF32()
		return Value{Type: wasmir.ValueTypeI32, I32: int32(math.Float32bits(v))}, err
	case wasmir.InstrI64ReinterpretF64:
		v, err := e.popF64()
		return Value{Type: wasmir.ValueTypeI64, I64: int64(math.Float64bits(v))}, err
	case wasmir.InstrF32ReinterpretI32:
		v, err := e.popI32()
		return Value{Type: wasmir.ValueTypeF32, F32: math.Float32frombits(uint32(v))}, err
	case wasmir.InstrF64ReinterpretI64:
		v, err := e.popI64()
		return Value{Type: wasmir.ValueTypeF64, F64: math.Float64frombits(uint64(v))}, err
	default:
		return Value{}, fmt.Errorf("unsupported conversion instruction %s", instrName(kind))
	}
}

// checkedTruncFloat truncates x toward zero and verifies the truncated value is
// in [lower, upper).
func checkedTruncFloat(x, lower, upper float64) (float64, error) {
	if math.IsNaN(x) {
		return 0, fmt.Errorf("invalid conversion to integer")
	}
	if math.IsInf(x, 0) {
		return 0, fmt.Errorf("integer overflow")
	}
	t := math.Trunc(x)
	if t < lower || t >= upper {
		return 0, fmt.Errorf("integer overflow")
	}
	return t, nil
}

// truncFloatToI32S implements trapping signed float-to-i32 truncation.
func truncFloatToI32S(x float64) (int32, error) {
	t, err := checkedTruncFloat(x, minInt32Float, two31Float)
	return int32(t), err
}

// truncFloatToI32U implements trapping unsigned float-to-i32 truncation.
func truncFloatToI32U(x float64) (int32, error) {
	t, err := checkedTruncFloat(x, 0, two32Float)
	return int32(uint32(t)), err
}

// truncFloatToI64S implements trapping signed float-to-i64 truncation.
func truncFloatToI64S(x float64) (int64, error) {
	t, err := checkedTruncFloat(x, minInt64Float, two63Float)
	return int64(t), err
}

// truncFloatToI64U implements trapping unsigned float-to-i64 truncation.
func truncFloatToI64U(x float64) (int64, error) {
	t, err := checkedTruncFloat(x, 0, two64Float)
	return int64(uint64(t)), err
}

// truncSatFloatToI32S implements saturating signed float-to-i32 truncation.
func truncSatFloatToI32S(x float64) int32 {
	if math.IsNaN(x) {
		return 0
	}
	t := math.Trunc(x)
	if t < minInt32Float {
		return minInt32
	}
	if t >= two31Float {
		return maxInt32
	}
	return int32(t)
}

// truncSatFloatToI32U implements saturating unsigned float-to-i32 truncation.
func truncSatFloatToI32U(x float64) int32 {
	if math.IsNaN(x) {
		return 0
	}
	t := math.Trunc(x)
	if t <= 0 {
		return 0
	}
	if t >= two32Float {
		v := ^uint32(0)
		return int32(v)
	}
	return int32(uint32(t))
}

// truncSatFloatToI64S implements saturating signed float-to-i64 truncation.
func truncSatFloatToI64S(x float64) int64 {
	if math.IsNaN(x) {
		return 0
	}
	t := math.Trunc(x)
	if t < minInt64Float {
		return minInt64
	}
	if t >= two63Float {
		return maxInt64
	}
	return int64(t)
}

// truncSatFloatToI64U implements saturating unsigned float-to-i64 truncation.
func truncSatFloatToI64U(x float64) int64 {
	if math.IsNaN(x) {
		return 0
	}
	t := math.Trunc(x)
	if t <= 0 {
		return 0
	}
	if t >= two64Float {
		v := ^uint64(0)
		return int64(v)
	}
	return int64(uint64(t))
}

// boolI32 converts a WebAssembly i32 condition result to 0 or 1.
func boolI32(v bool) int32 {
	if v {
		return 1
	}
	return 0
}

// evalV128Splat evaluates one SIMD splat instruction.
func (e *executor) evalV128Splat(kind wasmir.InstrKind) ([16]byte, error) {
	switch kind {
	case wasmir.InstrI8x16Splat:
		v, err := e.popI32()
		if err != nil {
			return [16]byte{}, err
		}
		return splatV128(1, uint64(uint8(v))), nil
	case wasmir.InstrI16x8Splat:
		v, err := e.popI32()
		if err != nil {
			return [16]byte{}, err
		}
		return splatV128(2, uint64(uint16(v))), nil
	case wasmir.InstrI32x4Splat:
		v, err := e.popI32()
		if err != nil {
			return [16]byte{}, err
		}
		return splatV128(4, uint64(uint32(v))), nil
	case wasmir.InstrI64x2Splat:
		v, err := e.popI64()
		if err != nil {
			return [16]byte{}, err
		}
		return splatV128(8, uint64(v)), nil
	case wasmir.InstrF32x4Splat:
		v, err := e.popF32()
		if err != nil {
			return [16]byte{}, err
		}
		return splatV128(4, uint64(math.Float32bits(v))), nil
	case wasmir.InstrF64x2Splat:
		v, err := e.popF64()
		if err != nil {
			return [16]byte{}, err
		}
		return splatV128(8, math.Float64bits(v)), nil
	default:
		return [16]byte{}, fmt.Errorf("unsupported splat instruction %s", instrName(kind))
	}
}

// evalV128LoadSplat evaluates one SIMD load-splat instruction.
func (e *executor) evalV128LoadSplat(kind wasmir.InstrKind, memoryIndex uint32, address uint64) ([16]byte, error) {
	width := v128LoadSplatWidth(kind)
	if width == 0 {
		return [16]byte{}, fmt.Errorf("unsupported load splat instruction %s", instrName(kind))
	}
	raw, err := e.inst.memoryLoad(memoryIndex, address, width)
	if err != nil {
		return [16]byte{}, err
	}
	return splatV128(width, raw), nil
}

// evalV128ExtractLane evaluates one SIMD lane extraction instruction.
func (e *executor) evalV128ExtractLane(kind wasmir.InstrKind, lane uint32) (Value, error) {
	vec, err := e.popV128()
	if err != nil {
		return Value{}, err
	}

	switch kind {
	case wasmir.InstrI8x16ExtractLaneS:
		offset, err := v128LaneByteOffset(kind, lane, 16, 1)
		if err != nil {
			return Value{}, err
		}
		return Value{Type: wasmir.ValueTypeI32, I32: int32(int8(vec[offset]))}, nil
	case wasmir.InstrI8x16ExtractLaneU:
		offset, err := v128LaneByteOffset(kind, lane, 16, 1)
		if err != nil {
			return Value{}, err
		}
		return Value{Type: wasmir.ValueTypeI32, I32: int32(vec[offset])}, nil
	case wasmir.InstrI16x8ExtractLaneS:
		offset, err := v128LaneByteOffset(kind, lane, 8, 2)
		if err != nil {
			return Value{}, err
		}
		raw := binary.LittleEndian.Uint16(vec[offset : offset+2])
		return Value{Type: wasmir.ValueTypeI32, I32: int32(int16(raw))}, nil
	case wasmir.InstrI16x8ExtractLaneU:
		offset, err := v128LaneByteOffset(kind, lane, 8, 2)
		if err != nil {
			return Value{}, err
		}
		raw := binary.LittleEndian.Uint16(vec[offset : offset+2])
		return Value{Type: wasmir.ValueTypeI32, I32: int32(raw)}, nil
	case wasmir.InstrI32x4ExtractLane:
		offset, err := v128LaneByteOffset(kind, lane, 4, 4)
		if err != nil {
			return Value{}, err
		}
		raw := binary.LittleEndian.Uint32(vec[offset : offset+4])
		return Value{Type: wasmir.ValueTypeI32, I32: int32(raw)}, nil
	case wasmir.InstrI64x2ExtractLane:
		offset, err := v128LaneByteOffset(kind, lane, 2, 8)
		if err != nil {
			return Value{}, err
		}
		raw := binary.LittleEndian.Uint64(vec[offset : offset+8])
		return Value{Type: wasmir.ValueTypeI64, I64: int64(raw)}, nil
	case wasmir.InstrF32x4ExtractLane:
		offset, err := v128LaneByteOffset(kind, lane, 4, 4)
		if err != nil {
			return Value{}, err
		}
		raw := binary.LittleEndian.Uint32(vec[offset : offset+4])
		return Value{Type: wasmir.ValueTypeF32, F32: math.Float32frombits(raw)}, nil
	case wasmir.InstrF64x2ExtractLane:
		offset, err := v128LaneByteOffset(kind, lane, 2, 8)
		if err != nil {
			return Value{}, err
		}
		raw := binary.LittleEndian.Uint64(vec[offset : offset+8])
		return Value{Type: wasmir.ValueTypeF64, F64: math.Float64frombits(raw)}, nil
	default:
		return Value{}, fmt.Errorf("unsupported extract_lane instruction %s", instrName(kind))
	}
}

// evalV128ReplaceLane evaluates one SIMD lane replacement instruction.
func (e *executor) evalV128ReplaceLane(kind wasmir.InstrKind, lane uint32) ([16]byte, error) {
	switch kind {
	case wasmir.InstrI8x16ReplaceLane:
		scalar, err := e.popI32()
		if err != nil {
			return [16]byte{}, err
		}
		vec, err := e.popV128()
		if err != nil {
			return [16]byte{}, err
		}
		offset, err := v128LaneByteOffset(kind, lane, 16, 1)
		if err != nil {
			return [16]byte{}, err
		}
		vec[offset] = byte(scalar)
		return vec, nil
	case wasmir.InstrI16x8ReplaceLane:
		scalar, err := e.popI32()
		if err != nil {
			return [16]byte{}, err
		}
		vec, err := e.popV128()
		if err != nil {
			return [16]byte{}, err
		}
		offset, err := v128LaneByteOffset(kind, lane, 8, 2)
		if err != nil {
			return [16]byte{}, err
		}
		binary.LittleEndian.PutUint16(vec[offset:offset+2], uint16(scalar))
		return vec, nil
	case wasmir.InstrI32x4ReplaceLane:
		scalar, err := e.popI32()
		if err != nil {
			return [16]byte{}, err
		}
		vec, err := e.popV128()
		if err != nil {
			return [16]byte{}, err
		}
		offset, err := v128LaneByteOffset(kind, lane, 4, 4)
		if err != nil {
			return [16]byte{}, err
		}
		binary.LittleEndian.PutUint32(vec[offset:offset+4], uint32(scalar))
		return vec, nil
	case wasmir.InstrI64x2ReplaceLane:
		scalar, err := e.popI64()
		if err != nil {
			return [16]byte{}, err
		}
		vec, err := e.popV128()
		if err != nil {
			return [16]byte{}, err
		}
		offset, err := v128LaneByteOffset(kind, lane, 2, 8)
		if err != nil {
			return [16]byte{}, err
		}
		binary.LittleEndian.PutUint64(vec[offset:offset+8], uint64(scalar))
		return vec, nil
	case wasmir.InstrF32x4ReplaceLane:
		scalar, err := e.popF32()
		if err != nil {
			return [16]byte{}, err
		}
		vec, err := e.popV128()
		if err != nil {
			return [16]byte{}, err
		}
		offset, err := v128LaneByteOffset(kind, lane, 4, 4)
		if err != nil {
			return [16]byte{}, err
		}
		binary.LittleEndian.PutUint32(vec[offset:offset+4], math.Float32bits(scalar))
		return vec, nil
	case wasmir.InstrF64x2ReplaceLane:
		scalar, err := e.popF64()
		if err != nil {
			return [16]byte{}, err
		}
		vec, err := e.popV128()
		if err != nil {
			return [16]byte{}, err
		}
		offset, err := v128LaneByteOffset(kind, lane, 2, 8)
		if err != nil {
			return [16]byte{}, err
		}
		binary.LittleEndian.PutUint64(vec[offset:offset+8], math.Float64bits(scalar))
		return vec, nil
	default:
		return [16]byte{}, fmt.Errorf("unsupported replace_lane instruction %s", instrName(kind))
	}
}

// evalI8x16Shuffle evaluates i8x16.shuffle using the compiled shuffle
// immediate stored on the function.
func (e *executor) evalI8x16Shuffle(shuffleIndex uint32) ([16]byte, error) {
	if int(shuffleIndex) >= len(e.fn.shuffleLanes) {
		return [16]byte{}, fmt.Errorf("shuffle index %d out of range", shuffleIndex)
	}
	laneIndex := e.fn.shuffleLanes[shuffleIndex]
	rhs, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}
	lhs, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}

	var out [16]byte
	for i, lane := range laneIndex {
		if lane < 16 {
			out[i] = lhs[lane]
		} else if lane < 32 {
			out[i] = rhs[lane-16]
		} else {
			return [16]byte{}, fmt.Errorf("shuffle lane %d out of range", lane)
		}
	}
	return out, nil
}

// evalV128Shift evaluates one integer SIMD lane shift instruction.
func (e *executor) evalV128Shift(kind wasmir.InstrKind) ([16]byte, error) {
	count, err := e.popI32()
	if err != nil {
		return [16]byte{}, err
	}
	vec, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}

	switch kind {
	case wasmir.InstrI8x16Shl, wasmir.InstrI8x16ShrS, wasmir.InstrI8x16ShrU:
		return shiftI8x16(kind, vec, uint32(count)&7), nil
	case wasmir.InstrI16x8Shl, wasmir.InstrI16x8ShrS, wasmir.InstrI16x8ShrU:
		return shiftI16x8(kind, vec, uint32(count)&15), nil
	case wasmir.InstrI32x4Shl, wasmir.InstrI32x4ShrS, wasmir.InstrI32x4ShrU:
		return shiftI32x4(kind, vec, uint32(count)&31), nil
	case wasmir.InstrI64x2Shl, wasmir.InstrI64x2ShrS, wasmir.InstrI64x2ShrU:
		return shiftI64x2(kind, vec, uint32(count)&63), nil
	default:
		return [16]byte{}, fmt.Errorf("unsupported SIMD shift instruction %s", instrName(kind))
	}
}

// evalV128Bitselect evaluates v128.bitselect.
func (e *executor) evalV128Bitselect() ([16]byte, error) {
	mask, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}
	b, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}
	a, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}

	var out [16]byte
	for i := range out {
		out[i] = (a[i] & mask[i]) | (b[i] &^ mask[i])
	}
	return out, nil
}

// evalV128Test evaluates a SIMD test instruction that returns an i32 result.
func (e *executor) evalV128Test(kind wasmir.InstrKind) (int32, error) {
	vec, err := e.popV128()
	if err != nil {
		return 0, err
	}

	switch kind {
	case wasmir.InstrV128AnyTrue:
		return boolI32(v128AnyTrue(vec)), nil
	case wasmir.InstrI8x16AllTrue:
		return boolI32(v128AllTrue(vec, 1)), nil
	case wasmir.InstrI16x8AllTrue:
		return boolI32(v128AllTrue(vec, 2)), nil
	case wasmir.InstrI32x4AllTrue:
		return boolI32(v128AllTrue(vec, 4)), nil
	case wasmir.InstrI64x2AllTrue:
		return boolI32(v128AllTrue(vec, 8)), nil
	case wasmir.InstrI8x16Bitmask:
		return int32(v128Bitmask(vec, 1)), nil
	case wasmir.InstrI16x8Bitmask:
		return int32(v128Bitmask(vec, 2)), nil
	case wasmir.InstrI32x4Bitmask:
		return int32(v128Bitmask(vec, 4)), nil
	case wasmir.InstrI64x2Bitmask:
		return int32(v128Bitmask(vec, 8)), nil
	default:
		return 0, fmt.Errorf("unsupported SIMD test instruction %s", instrName(kind))
	}
}

// evalV128Unary evaluates one SIMD unary instruction that returns a v128
// result.
func (e *executor) evalV128Unary(kind wasmir.InstrKind) ([16]byte, error) {
	vec, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}

	switch kind {
	case wasmir.InstrV128Not:
		return notV128(vec), nil
	case wasmir.InstrI8x16Neg:
		return negI8x16(vec), nil
	case wasmir.InstrI16x8Neg:
		return negI16x8(vec), nil
	case wasmir.InstrI32x4Neg:
		return negI32x4(vec), nil
	case wasmir.InstrI64x2Neg:
		return negI64x2(vec), nil
	case wasmir.InstrF32x4Abs, wasmir.InstrF32x4Neg, wasmir.InstrF32x4Sqrt:
		return unaryF32x4(kind, vec), nil
	case wasmir.InstrF64x2Abs, wasmir.InstrF64x2Neg, wasmir.InstrF64x2Sqrt:
		return unaryF64x2(kind, vec), nil
	case wasmir.InstrF32x4ConvertI32x4S, wasmir.InstrF32x4ConvertI32x4U:
		return convertI32x4ToF32x4(kind, vec), nil
	case wasmir.InstrI32x4TruncSatF32x4S, wasmir.InstrI32x4TruncSatF32x4U:
		return truncSatF32x4ToI32x4(kind, vec), nil
	default:
		return [16]byte{}, fmt.Errorf("unsupported SIMD unary instruction %s", instrName(kind))
	}
}

// evalV128Binary evaluates one SIMD binary instruction that returns a v128
// result.
func (e *executor) evalV128Binary(kind wasmir.InstrKind) ([16]byte, error) {
	rhs, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}
	lhs, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}

	switch kind {
	case wasmir.InstrI8x16Swizzle:
		return swizzleI8x16(lhs, rhs), nil
	case wasmir.InstrV128And, wasmir.InstrV128AndNot, wasmir.InstrV128Or, wasmir.InstrV128Xor:
		return bitwiseV128(kind, lhs, rhs), nil
	case wasmir.InstrI8x16Add, wasmir.InstrI8x16AddSatS, wasmir.InstrI8x16AddSatU,
		wasmir.InstrI8x16Sub, wasmir.InstrI8x16SubSatS, wasmir.InstrI8x16SubSatU:
		return binaryI8x16(kind, lhs, rhs), nil
	case wasmir.InstrI16x8Add, wasmir.InstrI16x8AddSatS, wasmir.InstrI16x8AddSatU,
		wasmir.InstrI16x8Sub, wasmir.InstrI16x8SubSatS, wasmir.InstrI16x8SubSatU, wasmir.InstrI16x8Mul:
		return binaryI16x8(kind, lhs, rhs), nil
	case wasmir.InstrI32x4Add, wasmir.InstrI32x4Sub, wasmir.InstrI32x4Mul:
		return binaryI32x4(kind, lhs, rhs), nil
	case wasmir.InstrI64x2Add, wasmir.InstrI64x2Sub, wasmir.InstrI64x2Mul:
		return binaryI64x2(kind, lhs, rhs), nil
	case wasmir.InstrF32x4Min, wasmir.InstrF32x4Max,
		wasmir.InstrF32x4Add, wasmir.InstrF32x4Sub, wasmir.InstrF32x4Div, wasmir.InstrF32x4Mul:
		return binaryF32x4(kind, lhs, rhs), nil
	case wasmir.InstrF64x2Min, wasmir.InstrF64x2Max,
		wasmir.InstrF64x2Add, wasmir.InstrF64x2Sub, wasmir.InstrF64x2Div, wasmir.InstrF64x2Mul:
		return binaryF64x2(kind, lhs, rhs), nil
	default:
		return [16]byte{}, fmt.Errorf("unsupported SIMD binary instruction %s", instrName(kind))
	}
}

// evalV128Compare evaluates one SIMD comparison instruction.
func (e *executor) evalV128Compare(kind wasmir.InstrKind) ([16]byte, error) {
	rhs, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}
	lhs, err := e.popV128()
	if err != nil {
		return [16]byte{}, err
	}

	switch kind {
	case wasmir.InstrI8x16Eq, wasmir.InstrI8x16Ne, wasmir.InstrI8x16LtS, wasmir.InstrI8x16LtU,
		wasmir.InstrI8x16GtS, wasmir.InstrI8x16GtU, wasmir.InstrI8x16LeS, wasmir.InstrI8x16LeU,
		wasmir.InstrI8x16GeS, wasmir.InstrI8x16GeU:
		return compareI8x16(kind, lhs, rhs), nil
	case wasmir.InstrI16x8Eq, wasmir.InstrI16x8Ne, wasmir.InstrI16x8LtS, wasmir.InstrI16x8LtU,
		wasmir.InstrI16x8GtS, wasmir.InstrI16x8GtU, wasmir.InstrI16x8LeS, wasmir.InstrI16x8LeU,
		wasmir.InstrI16x8GeS, wasmir.InstrI16x8GeU:
		return compareI16x8(kind, lhs, rhs), nil
	case wasmir.InstrI32x4Eq, wasmir.InstrI32x4Ne, wasmir.InstrI32x4LtS, wasmir.InstrI32x4LtU,
		wasmir.InstrI32x4GtS, wasmir.InstrI32x4GtU, wasmir.InstrI32x4LeS, wasmir.InstrI32x4LeU,
		wasmir.InstrI32x4GeS, wasmir.InstrI32x4GeU:
		return compareI32x4(kind, lhs, rhs), nil
	case wasmir.InstrI64x2Eq, wasmir.InstrI64x2Ne, wasmir.InstrI64x2LtS,
		wasmir.InstrI64x2GtS, wasmir.InstrI64x2LeS, wasmir.InstrI64x2GeS:
		return compareI64x2(kind, lhs, rhs), nil
	case wasmir.InstrF32x4Eq, wasmir.InstrF32x4Ne, wasmir.InstrF32x4Lt,
		wasmir.InstrF32x4Gt, wasmir.InstrF32x4Le, wasmir.InstrF32x4Ge:
		return compareF32x4(kind, lhs, rhs), nil
	case wasmir.InstrF64x2Eq, wasmir.InstrF64x2Ne, wasmir.InstrF64x2Lt,
		wasmir.InstrF64x2Gt, wasmir.InstrF64x2Le, wasmir.InstrF64x2Ge:
		return compareF64x2(kind, lhs, rhs), nil
	default:
		return [16]byte{}, fmt.Errorf("unsupported SIMD compare instruction %s", instrName(kind))
	}
}

// shiftI8x16 applies an integer SIMD shift to each i8 lane.
func shiftI8x16(kind wasmir.InstrKind, vec [16]byte, count uint32) [16]byte {
	var out [16]byte
	for i, raw := range vec {
		switch kind {
		case wasmir.InstrI8x16Shl:
			out[i] = byte(uint8(raw) << count)
		case wasmir.InstrI8x16ShrS:
			out[i] = byte(int8(raw) >> count)
		case wasmir.InstrI8x16ShrU:
			out[i] = byte(uint8(raw) >> count)
		}
	}
	return out
}

// shiftI16x8 applies an integer SIMD shift to each i16 lane.
func shiftI16x8(kind wasmir.InstrKind, vec [16]byte, count uint32) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 2 {
		raw := binary.LittleEndian.Uint16(vec[i : i+2])
		switch kind {
		case wasmir.InstrI16x8Shl:
			binary.LittleEndian.PutUint16(out[i:i+2], raw<<count)
		case wasmir.InstrI16x8ShrS:
			binary.LittleEndian.PutUint16(out[i:i+2], uint16(int16(raw)>>count))
		case wasmir.InstrI16x8ShrU:
			binary.LittleEndian.PutUint16(out[i:i+2], raw>>count)
		}
	}
	return out
}

// shiftI32x4 applies an integer SIMD shift to each i32 lane.
func shiftI32x4(kind wasmir.InstrKind, vec [16]byte, count uint32) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		raw := binary.LittleEndian.Uint32(vec[i : i+4])
		switch kind {
		case wasmir.InstrI32x4Shl:
			binary.LittleEndian.PutUint32(out[i:i+4], raw<<count)
		case wasmir.InstrI32x4ShrS:
			binary.LittleEndian.PutUint32(out[i:i+4], uint32(int32(raw)>>count))
		case wasmir.InstrI32x4ShrU:
			binary.LittleEndian.PutUint32(out[i:i+4], raw>>count)
		}
	}
	return out
}

// shiftI64x2 applies an integer SIMD shift to each i64 lane.
func shiftI64x2(kind wasmir.InstrKind, vec [16]byte, count uint32) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 8 {
		raw := binary.LittleEndian.Uint64(vec[i : i+8])
		switch kind {
		case wasmir.InstrI64x2Shl:
			binary.LittleEndian.PutUint64(out[i:i+8], raw<<count)
		case wasmir.InstrI64x2ShrS:
			binary.LittleEndian.PutUint64(out[i:i+8], uint64(int64(raw)>>count))
		case wasmir.InstrI64x2ShrU:
			binary.LittleEndian.PutUint64(out[i:i+8], raw>>count)
		}
	}
	return out
}

// v128AnyTrue reports whether any byte in vec is non-zero.
func v128AnyTrue(vec [16]byte) bool {
	for _, b := range vec {
		if b != 0 {
			return true
		}
	}
	return false
}

// v128AllTrue reports whether every integer lane of width bytes is non-zero.
func v128AllTrue(vec [16]byte, width int) bool {
	for i := 0; i < len(vec); i += width {
		switch width {
		case 1:
			if vec[i] == 0 {
				return false
			}
		case 2:
			if binary.LittleEndian.Uint16(vec[i:i+2]) == 0 {
				return false
			}
		case 4:
			if binary.LittleEndian.Uint32(vec[i:i+4]) == 0 {
				return false
			}
		case 8:
			if binary.LittleEndian.Uint64(vec[i:i+8]) == 0 {
				return false
			}
		}
	}
	return true
}

// v128Bitmask extracts each lane's sign bit into an i32 bitmask.
func v128Bitmask(vec [16]byte, width int) uint32 {
	var mask uint32
	for lane, i := 0, 0; i < len(vec); lane, i = lane+1, i+width {
		var sign bool
		switch width {
		case 1:
			sign = vec[i]&0x80 != 0
		case 2:
			sign = binary.LittleEndian.Uint16(vec[i:i+2])&0x8000 != 0
		case 4:
			sign = binary.LittleEndian.Uint32(vec[i:i+4])&0x80000000 != 0
		case 8:
			sign = binary.LittleEndian.Uint64(vec[i:i+8])&0x8000000000000000 != 0
		}
		if sign {
			mask |= 1 << lane
		}
	}
	return mask
}

// notV128 flips every bit in vec.
func notV128(vec [16]byte) [16]byte {
	var out [16]byte
	for i, b := range vec {
		out[i] = ^b
	}
	return out
}

// negI8x16 negates each i8 lane with wrapping arithmetic.
func negI8x16(vec [16]byte) [16]byte {
	var out [16]byte
	for i, b := range vec {
		out[i] = byte(0 - uint8(b))
	}
	return out
}

// negI16x8 negates each i16 lane with wrapping arithmetic.
func negI16x8(vec [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 2 {
		raw := binary.LittleEndian.Uint16(vec[i : i+2])
		binary.LittleEndian.PutUint16(out[i:i+2], 0-raw)
	}
	return out
}

// negI32x4 negates each i32 lane with wrapping arithmetic.
func negI32x4(vec [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		raw := binary.LittleEndian.Uint32(vec[i : i+4])
		binary.LittleEndian.PutUint32(out[i:i+4], 0-raw)
	}
	return out
}

// negI64x2 negates each i64 lane with wrapping arithmetic.
func negI64x2(vec [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 8 {
		raw := binary.LittleEndian.Uint64(vec[i : i+8])
		binary.LittleEndian.PutUint64(out[i:i+8], 0-raw)
	}
	return out
}

// unaryF32x4 applies one f32x4 unary operation lane-wise.
func unaryF32x4(kind wasmir.InstrKind, vec [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		raw := binary.LittleEndian.Uint32(vec[i : i+4])
		var result uint32
		switch kind {
		case wasmir.InstrF32x4Abs:
			result = raw & 0x7fffffff
		case wasmir.InstrF32x4Neg:
			result = raw ^ 0x80000000
		case wasmir.InstrF32x4Sqrt:
			v := math.Float32frombits(raw)
			if math.IsNaN(float64(v)) || v < 0 {
				result = canonicalF32NaNBits
			} else {
				result = math.Float32bits(float32(math.Sqrt(float64(v))))
			}
		}
		binary.LittleEndian.PutUint32(out[i:i+4], result)
	}
	return out
}

// unaryF64x2 applies one f64x2 unary operation lane-wise.
func unaryF64x2(kind wasmir.InstrKind, vec [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 8 {
		raw := binary.LittleEndian.Uint64(vec[i : i+8])
		var result uint64
		switch kind {
		case wasmir.InstrF64x2Abs:
			result = raw & 0x7fffffffffffffff
		case wasmir.InstrF64x2Neg:
			result = raw ^ 0x8000000000000000
		case wasmir.InstrF64x2Sqrt:
			v := math.Float64frombits(raw)
			if math.IsNaN(v) || v < 0 {
				result = canonicalF64NaNBits
			} else {
				result = math.Float64bits(math.Sqrt(v))
			}
		}
		binary.LittleEndian.PutUint64(out[i:i+8], result)
	}
	return out
}

// convertI32x4ToF32x4 converts each i32 lane to an f32 lane.
func convertI32x4ToF32x4(kind wasmir.InstrKind, vec [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		raw := binary.LittleEndian.Uint32(vec[i : i+4])
		var result float32
		switch kind {
		case wasmir.InstrF32x4ConvertI32x4S:
			result = float32(int32(raw))
		case wasmir.InstrF32x4ConvertI32x4U:
			result = float32(raw)
		}
		binary.LittleEndian.PutUint32(out[i:i+4], math.Float32bits(result))
	}
	return out
}

// truncSatF32x4ToI32x4 saturating-truncates each f32 lane to an i32 lane.
func truncSatF32x4ToI32x4(kind wasmir.InstrKind, vec [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(vec[i : i+4]))
		var result int32
		switch kind {
		case wasmir.InstrI32x4TruncSatF32x4S:
			result = truncSatFloatToI32S(float64(v))
		case wasmir.InstrI32x4TruncSatF32x4U:
			result = truncSatFloatToI32U(float64(v))
		}
		binary.LittleEndian.PutUint32(out[i:i+4], uint32(result))
	}
	return out
}

// swizzleI8x16 rearranges lhs bytes according to rhs byte indices.
func swizzleI8x16(lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i, lane := range rhs {
		if lane < 16 {
			out[i] = lhs[lane]
		}
	}
	return out
}

// bitwiseV128 applies one v128 bitwise binary operation byte-wise.
func bitwiseV128(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := range out {
		switch kind {
		case wasmir.InstrV128And:
			out[i] = lhs[i] & rhs[i]
		case wasmir.InstrV128AndNot:
			out[i] = lhs[i] &^ rhs[i]
		case wasmir.InstrV128Or:
			out[i] = lhs[i] | rhs[i]
		case wasmir.InstrV128Xor:
			out[i] = lhs[i] ^ rhs[i]
		}
	}
	return out
}

// binaryI8x16 applies one integer binary operation to each i8 lane.
func binaryI8x16(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := range out {
		a := lhs[i]
		b := rhs[i]
		switch kind {
		case wasmir.InstrI8x16Add:
			out[i] = a + b
		case wasmir.InstrI8x16Sub:
			out[i] = a - b
		case wasmir.InstrI8x16AddSatS:
			out[i] = byte(int8(clampInt(int(int8(a))+int(int8(b)), -128, 127)))
		case wasmir.InstrI8x16AddSatU:
			out[i] = byte(clampInt(int(a)+int(b), 0, 255))
		case wasmir.InstrI8x16SubSatS:
			out[i] = byte(int8(clampInt(int(int8(a))-int(int8(b)), -128, 127)))
		case wasmir.InstrI8x16SubSatU:
			out[i] = byte(clampInt(int(a)-int(b), 0, 255))
		}
	}
	return out
}

// binaryI16x8 applies one integer binary operation to each i16 lane.
func binaryI16x8(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 2 {
		a := binary.LittleEndian.Uint16(lhs[i : i+2])
		b := binary.LittleEndian.Uint16(rhs[i : i+2])
		var result uint16
		switch kind {
		case wasmir.InstrI16x8Add:
			result = a + b
		case wasmir.InstrI16x8Sub:
			result = a - b
		case wasmir.InstrI16x8Mul:
			result = a * b
		case wasmir.InstrI16x8AddSatS:
			result = uint16(int16(clampInt(int(int16(a))+int(int16(b)), -32768, 32767)))
		case wasmir.InstrI16x8AddSatU:
			result = uint16(clampInt(int(a)+int(b), 0, 65535))
		case wasmir.InstrI16x8SubSatS:
			result = uint16(int16(clampInt(int(int16(a))-int(int16(b)), -32768, 32767)))
		case wasmir.InstrI16x8SubSatU:
			result = uint16(clampInt(int(a)-int(b), 0, 65535))
		}
		binary.LittleEndian.PutUint16(out[i:i+2], result)
	}
	return out
}

// binaryI32x4 applies one wrapping integer binary operation to each i32 lane.
func binaryI32x4(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		a := binary.LittleEndian.Uint32(lhs[i : i+4])
		b := binary.LittleEndian.Uint32(rhs[i : i+4])
		var result uint32
		switch kind {
		case wasmir.InstrI32x4Add:
			result = a + b
		case wasmir.InstrI32x4Sub:
			result = a - b
		case wasmir.InstrI32x4Mul:
			result = a * b
		}
		binary.LittleEndian.PutUint32(out[i:i+4], result)
	}
	return out
}

// binaryI64x2 applies one wrapping integer binary operation to each i64 lane.
func binaryI64x2(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 8 {
		a := binary.LittleEndian.Uint64(lhs[i : i+8])
		b := binary.LittleEndian.Uint64(rhs[i : i+8])
		var result uint64
		switch kind {
		case wasmir.InstrI64x2Add:
			result = a + b
		case wasmir.InstrI64x2Sub:
			result = a - b
		case wasmir.InstrI64x2Mul:
			result = a * b
		}
		binary.LittleEndian.PutUint64(out[i:i+8], result)
	}
	return out
}

// binaryF32x4 applies one f32 binary operation to each lane.
func binaryF32x4(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		a := math.Float32frombits(binary.LittleEndian.Uint32(lhs[i : i+4]))
		b := math.Float32frombits(binary.LittleEndian.Uint32(rhs[i : i+4]))
		result := binaryF32(kind, a, b)
		binary.LittleEndian.PutUint32(out[i:i+4], result)
	}
	return out
}

// binaryF64x2 applies one f64 binary operation to each lane.
func binaryF64x2(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 8 {
		a := math.Float64frombits(binary.LittleEndian.Uint64(lhs[i : i+8]))
		b := math.Float64frombits(binary.LittleEndian.Uint64(rhs[i : i+8]))
		result := binaryF64(kind, a, b)
		binary.LittleEndian.PutUint64(out[i:i+8], result)
	}
	return out
}

// binaryF32 applies one scalar f32 operation and returns the result bits.
func binaryF32(kind wasmir.InstrKind, a float32, b float32) uint32 {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return canonicalF32NaNBits
	}

	var result float32
	switch kind {
	case wasmir.InstrF32Min, wasmir.InstrF32x4Min:
		if a == 0 && b == 0 && (math.Signbit(float64(a)) || math.Signbit(float64(b))) {
			return 0x80000000
		}
		if a < b {
			result = a
		} else {
			result = b
		}
	case wasmir.InstrF32Max, wasmir.InstrF32x4Max:
		if a == 0 && b == 0 && (!math.Signbit(float64(a)) || !math.Signbit(float64(b))) {
			return 0
		}
		if a > b {
			result = a
		} else {
			result = b
		}
	case wasmir.InstrF32Add, wasmir.InstrF32x4Add:
		result = a + b
	case wasmir.InstrF32Sub, wasmir.InstrF32x4Sub:
		result = a - b
	case wasmir.InstrF32Div, wasmir.InstrF32x4Div:
		result = a / b
	case wasmir.InstrF32Mul, wasmir.InstrF32x4Mul:
		result = a * b
	}
	if math.IsNaN(float64(result)) {
		return canonicalF32NaNBits
	}
	return math.Float32bits(result)
}

// binaryF64 applies one scalar f64 operation and returns the result bits.
func binaryF64(kind wasmir.InstrKind, a float64, b float64) uint64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return canonicalF64NaNBits
	}

	var result float64
	switch kind {
	case wasmir.InstrF64Min, wasmir.InstrF64x2Min:
		if a == 0 && b == 0 && (math.Signbit(a) || math.Signbit(b)) {
			return 0x8000000000000000
		}
		if a < b {
			result = a
		} else {
			result = b
		}
	case wasmir.InstrF64Max, wasmir.InstrF64x2Max:
		if a == 0 && b == 0 && (!math.Signbit(a) || !math.Signbit(b)) {
			return 0
		}
		if a > b {
			result = a
		} else {
			result = b
		}
	case wasmir.InstrF64Add, wasmir.InstrF64x2Add:
		result = a + b
	case wasmir.InstrF64Sub, wasmir.InstrF64x2Sub:
		result = a - b
	case wasmir.InstrF64Div, wasmir.InstrF64x2Div:
		result = a / b
	case wasmir.InstrF64Mul, wasmir.InstrF64x2Mul:
		result = a * b
	}
	if math.IsNaN(result) {
		return canonicalF64NaNBits
	}
	return math.Float64bits(result)
}

// clampInt clamps v into the inclusive [lo, hi] range.
func clampInt(v int, lo int, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// compareI8x16 compares each i8 lane and writes an i8 lane mask result.
func compareI8x16(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := range out {
		a := lhs[i]
		b := rhs[i]
		var ok bool
		switch kind {
		case wasmir.InstrI8x16Eq:
			ok = a == b
		case wasmir.InstrI8x16Ne:
			ok = a != b
		case wasmir.InstrI8x16LtS:
			ok = int8(a) < int8(b)
		case wasmir.InstrI8x16LtU:
			ok = a < b
		case wasmir.InstrI8x16GtS:
			ok = int8(a) > int8(b)
		case wasmir.InstrI8x16GtU:
			ok = a > b
		case wasmir.InstrI8x16LeS:
			ok = int8(a) <= int8(b)
		case wasmir.InstrI8x16LeU:
			ok = a <= b
		case wasmir.InstrI8x16GeS:
			ok = int8(a) >= int8(b)
		case wasmir.InstrI8x16GeU:
			ok = a >= b
		}
		out[i] = boolMask8(ok)
	}
	return out
}

// compareI16x8 compares each i16 lane and writes an i16 lane mask result.
func compareI16x8(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 2 {
		a := binary.LittleEndian.Uint16(lhs[i : i+2])
		b := binary.LittleEndian.Uint16(rhs[i : i+2])
		var ok bool
		switch kind {
		case wasmir.InstrI16x8Eq:
			ok = a == b
		case wasmir.InstrI16x8Ne:
			ok = a != b
		case wasmir.InstrI16x8LtS:
			ok = int16(a) < int16(b)
		case wasmir.InstrI16x8LtU:
			ok = a < b
		case wasmir.InstrI16x8GtS:
			ok = int16(a) > int16(b)
		case wasmir.InstrI16x8GtU:
			ok = a > b
		case wasmir.InstrI16x8LeS:
			ok = int16(a) <= int16(b)
		case wasmir.InstrI16x8LeU:
			ok = a <= b
		case wasmir.InstrI16x8GeS:
			ok = int16(a) >= int16(b)
		case wasmir.InstrI16x8GeU:
			ok = a >= b
		}
		binary.LittleEndian.PutUint16(out[i:i+2], boolMask16(ok))
	}
	return out
}

// compareI32x4 compares each i32 lane and writes an i32 lane mask result.
func compareI32x4(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		a := binary.LittleEndian.Uint32(lhs[i : i+4])
		b := binary.LittleEndian.Uint32(rhs[i : i+4])
		var ok bool
		switch kind {
		case wasmir.InstrI32x4Eq:
			ok = a == b
		case wasmir.InstrI32x4Ne:
			ok = a != b
		case wasmir.InstrI32x4LtS:
			ok = int32(a) < int32(b)
		case wasmir.InstrI32x4LtU:
			ok = a < b
		case wasmir.InstrI32x4GtS:
			ok = int32(a) > int32(b)
		case wasmir.InstrI32x4GtU:
			ok = a > b
		case wasmir.InstrI32x4LeS:
			ok = int32(a) <= int32(b)
		case wasmir.InstrI32x4LeU:
			ok = a <= b
		case wasmir.InstrI32x4GeS:
			ok = int32(a) >= int32(b)
		case wasmir.InstrI32x4GeU:
			ok = a >= b
		}
		binary.LittleEndian.PutUint32(out[i:i+4], boolMask32(ok))
	}
	return out
}

// compareI64x2 compares each i64 lane and writes an i64 lane mask result.
func compareI64x2(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 8 {
		a := binary.LittleEndian.Uint64(lhs[i : i+8])
		b := binary.LittleEndian.Uint64(rhs[i : i+8])
		var ok bool
		switch kind {
		case wasmir.InstrI64x2Eq:
			ok = a == b
		case wasmir.InstrI64x2Ne:
			ok = a != b
		case wasmir.InstrI64x2LtS:
			ok = int64(a) < int64(b)
		case wasmir.InstrI64x2GtS:
			ok = int64(a) > int64(b)
		case wasmir.InstrI64x2LeS:
			ok = int64(a) <= int64(b)
		case wasmir.InstrI64x2GeS:
			ok = int64(a) >= int64(b)
		}
		binary.LittleEndian.PutUint64(out[i:i+8], boolMask64(ok))
	}
	return out
}

// compareF32x4 compares each f32 lane and writes an i32 lane mask result.
func compareF32x4(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 4 {
		a := math.Float32frombits(binary.LittleEndian.Uint32(lhs[i : i+4]))
		b := math.Float32frombits(binary.LittleEndian.Uint32(rhs[i : i+4]))
		var ok bool
		switch kind {
		case wasmir.InstrF32x4Eq:
			ok = a == b
		case wasmir.InstrF32x4Ne:
			ok = a != b
		case wasmir.InstrF32x4Lt:
			ok = a < b
		case wasmir.InstrF32x4Gt:
			ok = a > b
		case wasmir.InstrF32x4Le:
			ok = a <= b
		case wasmir.InstrF32x4Ge:
			ok = a >= b
		}
		binary.LittleEndian.PutUint32(out[i:i+4], boolMask32(ok))
	}
	return out
}

// compareF64x2 compares each f64 lane and writes an i64 lane mask result.
func compareF64x2(kind wasmir.InstrKind, lhs [16]byte, rhs [16]byte) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += 8 {
		a := math.Float64frombits(binary.LittleEndian.Uint64(lhs[i : i+8]))
		b := math.Float64frombits(binary.LittleEndian.Uint64(rhs[i : i+8]))
		var ok bool
		switch kind {
		case wasmir.InstrF64x2Eq:
			ok = a == b
		case wasmir.InstrF64x2Ne:
			ok = a != b
		case wasmir.InstrF64x2Lt:
			ok = a < b
		case wasmir.InstrF64x2Gt:
			ok = a > b
		case wasmir.InstrF64x2Le:
			ok = a <= b
		case wasmir.InstrF64x2Ge:
			ok = a >= b
		}
		binary.LittleEndian.PutUint64(out[i:i+8], boolMask64(ok))
	}
	return out
}

// boolMask8 returns a one-byte lane mask for a SIMD comparison result.
func boolMask8(ok bool) byte {
	if ok {
		return 0xff
	}
	return 0
}

// boolMask16 returns a two-byte lane mask for a SIMD comparison result.
func boolMask16(ok bool) uint16 {
	if ok {
		return ^uint16(0)
	}
	return 0
}

// boolMask32 returns a four-byte lane mask for a SIMD comparison result.
func boolMask32(ok bool) uint32 {
	if ok {
		return ^uint32(0)
	}
	return 0
}

// boolMask64 returns an eight-byte lane mask for a SIMD comparison result.
func boolMask64(ok bool) uint64 {
	if ok {
		return ^uint64(0)
	}
	return 0
}

// v128LaneByteOffset validates a SIMD lane index and returns its byte offset.
func v128LaneByteOffset(kind wasmir.InstrKind, lane uint32, lanes uint32, width uint32) (uint32, error) {
	if lane >= lanes {
		return 0, fmt.Errorf("%s lane %d out of range for %d lanes", instrName(kind), lane, lanes)
	}
	return lane * width, nil
}

// v128LoadSplatWidth returns the byte width read by a SIMD load-splat
// instruction.
func v128LoadSplatWidth(kind wasmir.InstrKind) uint32 {
	switch kind {
	case wasmir.InstrV128Load8Splat:
		return 1
	case wasmir.InstrV128Load16Splat:
		return 2
	case wasmir.InstrV128Load32Splat:
		return 4
	case wasmir.InstrV128Load64Splat:
		return 8
	default:
		return 0
	}
}

// splatV128 fills a v128 value by repeating raw's low width bytes.
func splatV128(width uint32, raw uint64) [16]byte {
	var out [16]byte
	for i := 0; i < len(out); i += int(width) {
		switch width {
		case 1:
			out[i] = byte(raw)
		case 2:
			binary.LittleEndian.PutUint16(out[i:i+2], uint16(raw))
		case 4:
			binary.LittleEndian.PutUint32(out[i:i+4], uint32(raw))
		case 8:
			binary.LittleEndian.PutUint64(out[i:i+8], raw)
		}
	}
	return out
}

// popMemoryIndexOperand pops an operand whose type follows the indexed
// memory's address type.
func (e *executor) popMemoryIndexOperand(memoryIndex uint32) (uint64, error) {
	addressType, err := e.inst.memoryAddressType(memoryIndex)
	if err != nil {
		return 0, err
	}
	switch addressType {
	case wasmir.ValueTypeI32:
		base, err := e.popI32()
		if err != nil {
			return 0, err
		}
		return uint64(uint32(base)), nil
	case wasmir.ValueTypeI64:
		base, err := e.popI64()
		if err != nil {
			return 0, err
		}
		return uint64(base), nil
	default:
		return 0, fmt.Errorf("memory %d has unsupported address type %s", memoryIndex, addressType)
	}
}

// popMemoryAddress pops a dynamic address operand and applies the static
// memory offset immediate.
func (e *executor) popMemoryAddress(memoryIndex uint32, offset uint64) (uint64, error) {
	base, err := e.popMemoryIndexOperand(memoryIndex)
	if err != nil {
		return 0, err
	}
	return memoryAddress(base, offset)
}

// pushMemoryIndexResult pushes a memory.size or memory.grow result in the
// indexed memory's address type.
func (e *executor) pushMemoryIndexResult(memoryIndex uint32, value uint64) error {
	addressType, err := e.inst.memoryAddressType(memoryIndex)
	if err != nil {
		return err
	}
	switch addressType {
	case wasmir.ValueTypeI32:
		e.push(Value{Type: wasmir.ValueTypeI32, I32: int32(uint32(value))})
		return nil
	case wasmir.ValueTypeI64:
		e.push(Value{Type: wasmir.ValueTypeI64, I64: int64(value)})
		return nil
	default:
		return fmt.Errorf("memory %d has unsupported address type %s", memoryIndex, addressType)
	}
}

// memoryAddress computes an effective address from the dynamic base operand
// and the static memory offset immediate.
func memoryAddress(base uint64, offset uint64) (uint64, error) {
	if base > ^uint64(0)-offset {
		return 0, fmt.Errorf("memory address overflow")
	}
	return base + offset, nil
}

// memoryAccessSize returns the byte width used by a supported memory
// load/store instruction.
func memoryAccessSize(kind wasmir.InstrKind) uint32 {
	switch kind {
	case wasmir.InstrI32Load8S, wasmir.InstrI32Load8U, wasmir.InstrI32Store8,
		wasmir.InstrI64Load8S, wasmir.InstrI64Load8U, wasmir.InstrI64Store8:
		return 1
	case wasmir.InstrI32Load16S, wasmir.InstrI32Load16U, wasmir.InstrI32Store16,
		wasmir.InstrI64Load16S, wasmir.InstrI64Load16U, wasmir.InstrI64Store16:
		return 2
	case wasmir.InstrI32Load, wasmir.InstrI32Store,
		wasmir.InstrI64Load32S, wasmir.InstrI64Load32U, wasmir.InstrI64Store32,
		wasmir.InstrF32Load, wasmir.InstrF32Store:
		return 4
	default:
		return 8
	}
}

// extendI32Load applies the sign-extension or zero-extension behavior required
// by kind to the raw little-endian memory value.
func extendI32Load(kind wasmir.InstrKind, raw uint64) int32 {
	switch kind {
	case wasmir.InstrI32Load8S:
		return int32(int8(raw))
	case wasmir.InstrI32Load8U:
		return int32(uint8(raw))
	case wasmir.InstrI32Load16S:
		return int32(int16(raw))
	case wasmir.InstrI32Load16U:
		return int32(uint16(raw))
	default:
		return int32(uint32(raw))
	}
}

// extendI64Load applies the sign-extension or zero-extension behavior required
// by kind to the raw little-endian memory value.
func extendI64Load(kind wasmir.InstrKind, raw uint64) int64 {
	switch kind {
	case wasmir.InstrI64Load8S:
		return int64(int8(raw))
	case wasmir.InstrI64Load8U:
		return int64(uint8(raw))
	case wasmir.InstrI64Load16S:
		return int64(int16(raw))
	case wasmir.InstrI64Load16U:
		return int64(uint16(raw))
	case wasmir.InstrI64Load32S:
		return int64(int32(raw))
	case wasmir.InstrI64Load32U:
		return int64(uint32(raw))
	default:
		return int64(raw)
	}
}

// popWant pops the top operand and verifies it has the expected value type.
func (e *executor) popWant(want wasmir.ValueType) (Value, error) {
	v, err := e.pop()
	if err != nil {
		return Value{}, err
	}
	if v.Type != want {
		return Value{}, fmt.Errorf("got %s operand, want %s", v.Type, want)
	}
	return v, nil
}

// popI32 pops the top operand and returns its i32 payload.
func (e *executor) popI32() (int32, error) {
	v, err := e.popWant(wasmir.ValueTypeI32)
	return v.I32, err
}

// popI64 pops the top operand and returns its i64 payload.
func (e *executor) popI64() (int64, error) {
	v, err := e.popWant(wasmir.ValueTypeI64)
	return v.I64, err
}

// popF32 pops the top operand and returns its f32 payload.
func (e *executor) popF32() (float32, error) {
	v, err := e.popWant(wasmir.ValueTypeF32)
	return v.F32, err
}

// popF64 pops the top operand and returns its f64 payload.
func (e *executor) popF64() (float64, error) {
	v, err := e.popWant(wasmir.ValueTypeF64)
	return v.F64, err
}

// popV128 pops the top operand and returns its v128 payload.
func (e *executor) popV128() ([16]byte, error) {
	v, err := e.popWant(wasmir.ValueTypeV128)
	return v.V128, err
}

// popArgs removes a call's arguments from the operand stack and returns them in
// parameter order.
func (e *executor) popArgs(params []wasmir.ValueType) ([]Value, error) {
	// Wasm evaluates arguments left-to-right and leaves them on the operand
	// stack in parameter order, so the call argument list is the top
	// len(params) values without reversing.
	if len(e.stack) < len(params) {
		return nil, fmt.Errorf("operand stack underflow")
	}
	base := len(e.stack) - len(params)
	args := e.stack[base:]
	e.stack = e.stack[:base]
	if err := checkArgs(params, args); err != nil {
		return nil, err
	}
	return args, nil
}

// popResults removes function result values from the operand stack and returns
// them in result order.
func (e *executor) popResults(results []wasmir.ValueType) ([]Value, error) {
	if len(e.stack) < len(results) {
		return nil, fmt.Errorf("operand stack underflow")
	}
	base := len(e.stack) - len(results)
	out := e.stack[base:]
	e.stack = e.stack[:base]
	if err := checkResults(results, out); err != nil {
		return nil, err
	}
	return out, nil
}
