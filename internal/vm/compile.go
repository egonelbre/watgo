package vm

import (
	"fmt"
	"slices"

	"github.com/eliben/watgo/internal/instrdef"
	"github.com/eliben/watgo/wasmir"
)

// function is the VM's execution form for a module-defined function.
type function struct {
	// locals contains the non-parameter locals declared by the function. At
	// call time executeFunction builds its local array as args followed by
	// these zero-initialized locals.
	locals []wasmir.ValueType

	// code is the linear instruction stream consumed by executeFunction. It has
	// the same instruction order as wasmir.Function.Body, but immediate fields
	// have been normalized for execution.
	code []instr

	// labels stores execution metadata for structured-control openers, keyed by
	// the opener instruction pc. The executor uses it to maintain the active
	// label stack and to normalize operand-stack height on branches and end.
	labels map[int]controlLabel

	// branchTables stores the variable-length target lists used by br_table
	// instructions. A br_table instruction keeps its fixed-size instr small by
	// storing the index of its target list in instr.index.
	//
	// Each target list contains resolved targets in the same order as the
	// source instruction's BranchTable depths. The default target is appended
	// as the final element, so execution can use selector values below
	// len(table)-1 as direct table indices and use len(table)-1 for the default
	// case.
	branchTables [][]branchTarget

	// refTypes stores reference type immediates used by ref.null instructions.
	// A ref.null instruction keeps its fixed-size instr small by storing the
	// index of its type immediate in instr.index.
	refTypes []wasmir.ValueType

	// selectTypes stores explicit result type immediates used by typed select.
	// A typed select instruction stores its selectTypes index in instr.bits;
	// untyped select stores -1 there.
	selectTypes []wasmir.ValueType

	// v128Consts stores SIMD constant immediates used by v128.const. The
	// instruction stores its v128Consts index in instr.index.
	v128Consts [][16]byte

	// shuffleLanes stores SIMD shuffle immediates used by i8x16.shuffle. The
	// instruction stores its shuffleLanes index in instr.index.
	shuffleLanes [][16]byte

	// laneMemories stores the memory index, offset, and lane immediate used by
	// SIMD lane-load and lane-store instructions. The instruction stores its
	// laneMemories index in instr.index.
	laneMemories []laneMemoryImmediate
}

// catchTarget is one compiled try_table catch clause.
type catchTarget struct {
	kind     wasmir.TryTableCatchKind
	tagIndex uint32
	target   branchTarget
}

// laneMemoryImmediate is the immediate payload for a SIMD lane memory
// instruction.
type laneMemoryImmediate struct {
	memoryIndex uint32
	offset      uint64
	lane        uint32
}

// instr is one instruction in the VM's execution form.
type instr struct {
	// kind is the semantic instruction kind executed by the interpreter.
	kind wasmir.InstrKind

	// target is the resolved program counter for fixed-target control-flow
	// instructions. It is used by if, else, br, br_if, br_on_null,
	// br_on_non_null, and loop-branch targets; other instructions leave it at
	// -1. The interpreter assigns pc = target, then its loop increment moves
	// execution to the following instruction.
	target int

	// index is the resolved index immediate for local.get/set/tee,
	// global.get/set, call, call_ref, memory and table instructions,
	// data.drop/elem.drop, br_table's branchTables entry, and ref.null's
	// refTypes entry. v128.const, i8x16.shuffle, and SIMD lane memory
	// instructions use it as a pool entry; SIMD extract/replace instructions
	// use it as the lane immediate.
	index uint32

	// bits is the raw immediate payload for constant instructions.
	//
	// kind determines how to interpret it: i32.const uses int32(bits),
	// i64.const uses bits, f32.const uses uint32(bits), and f64.const uses
	// uint64(bits). For memory load/store instructions, bits stores the raw
	// uint64 bit pattern of the static offset immediate. For memory.copy and
	// memory.init, bits stores the secondary index immediate. Indirect calls
	// use bits for the call type index, and table bulk instructions use it for
	// the source table or element segment index.
	bits int64
}

// branchTarget is one resolved branch destination in the VM's execution form.
type branchTarget struct {
	// pc is the program counter assigned to the executor before the executor's
	// loop increment selects the next instruction to run.
	pc int

	// depth is the original label-depth immediate. The executor uses it to find
	// the active runtime label whose stack height and branch arity must be
	// applied.
	depth uint32
}

// compileFunction compiles a semantic function body into the VM's execution form.
func compileFunction(m *wasmir.Module, fn *wasmir.Function) (*function, error) {
	ctrl, err := analyzeControl(fn.Body)
	if err != nil {
		return nil, err
	}

	out := &function{
		locals: slices.Clone(fn.Locals),
		code:   make([]instr, len(fn.Body)),
		labels: ctrl.labels,
	}
	labelStack := make([]int, 0)
	finalEnd := len(fn.Body) - 1

	for pc, ins := range fn.Body {
		op := instr{kind: ins.Kind, target: -1, bits: -1}
		switch ins.Kind {
		case wasmir.InstrBlock, wasmir.InstrLoop, wasmir.InstrTryTable:
			label, ok := ctrl.labels[pc]
			if !ok {
				return nil, fmt.Errorf("%s at %d has no matching end", instrName(ins.Kind), pc)
			}
			if err := setControlLabelSignature(m, ins, label, ctrl.labels, pc); err != nil {
				return nil, err
			}
			if ins.Kind == wasmir.InstrTryTable {
				if err := setControlLabelCatches(ins, label, labelStack, ctrl, finalEnd, pc); err != nil {
					return nil, err
				}
			}
			labelStack = append(labelStack, pc)
		case wasmir.InstrIf:
			label, ok := ctrl.labels[pc]
			if !ok {
				return nil, fmt.Errorf("if at %d has no matching end", pc)
			}
			if err := setControlLabelSignature(m, ins, label, ctrl.labels, pc); err != nil {
				return nil, err
			}
			if label.elseIndex >= 0 {
				op.target = label.elseIndex
			} else {
				op.target = label.endIndex
			}
			labelStack = append(labelStack, pc)
		case wasmir.InstrElse:
			if len(labelStack) == 0 {
				return nil, fmt.Errorf("else at %d without active label", pc)
			}
			start := labelStack[len(labelStack)-1]
			label := ctrl.labels[start]
			if label.elseIndex != pc {
				return nil, fmt.Errorf("else at %d does not match active label", pc)
			}
			op.target = label.endIndex
		case wasmir.InstrLocalGet, wasmir.InstrLocalSet, wasmir.InstrLocalTee:
			op.index = ins.LocalIndex
		case wasmir.InstrCall, wasmir.InstrReturnCall:
			op.index = ins.FuncIndex
		case wasmir.InstrCallRef, wasmir.InstrReturnCallRef:
			op.index = ins.CallTypeIndex
		case wasmir.InstrCallIndirect, wasmir.InstrReturnCallIndirect:
			op.index = ins.TableIndex
			op.bits = int64(ins.CallTypeIndex)
		case wasmir.InstrThrow:
			op.index = ins.TagIndex
		case wasmir.InstrRefNull:
			op.index = uint32(len(out.refTypes))
			out.refTypes = append(out.refTypes, ins.RefType)
		case wasmir.InstrRefFunc:
			op.index = ins.FuncIndex
		case wasmir.InstrGlobalGet, wasmir.InstrGlobalSet:
			op.index = ins.GlobalIndex
		case wasmir.InstrMemorySize, wasmir.InstrMemoryGrow, wasmir.InstrMemoryFill:
			op.index = ins.MemoryIndex
		case wasmir.InstrTableGet, wasmir.InstrTableSet, wasmir.InstrTableSize,
			wasmir.InstrTableGrow, wasmir.InstrTableFill:
			op.index = ins.TableIndex
		case wasmir.InstrTableCopy:
			op.index = ins.TableIndex
			op.bits = int64(ins.SourceTableIndex)
		case wasmir.InstrTableInit:
			op.index = ins.TableIndex
			op.bits = int64(ins.ElemIndex)
		case wasmir.InstrElemDrop:
			op.index = ins.ElemIndex
		case wasmir.InstrMemoryCopy:
			op.index = ins.MemoryIndex
			op.bits = int64(ins.SourceMemoryIndex)
		case wasmir.InstrMemoryInit:
			op.index = ins.MemoryIndex
			op.bits = int64(ins.DataIndex)
		case wasmir.InstrDataDrop:
			op.index = ins.DataIndex
		case wasmir.InstrSelect:
			if ins.SelectType != nil {
				op.bits = int64(len(out.selectTypes))
				out.selectTypes = append(out.selectTypes, *ins.SelectType)
			}
		case wasmir.InstrI32Load, wasmir.InstrI32Store,
			wasmir.InstrI32Load8S, wasmir.InstrI32Load8U,
			wasmir.InstrI32Load16S, wasmir.InstrI32Load16U,
			wasmir.InstrI32Store8, wasmir.InstrI32Store16,
			wasmir.InstrI64Load, wasmir.InstrI64Store,
			wasmir.InstrI64Load8S, wasmir.InstrI64Load8U,
			wasmir.InstrI64Load16S, wasmir.InstrI64Load16U,
			wasmir.InstrI64Load32S, wasmir.InstrI64Load32U,
			wasmir.InstrI64Store8, wasmir.InstrI64Store16, wasmir.InstrI64Store32,
			wasmir.InstrF32Load, wasmir.InstrF32Store,
			wasmir.InstrF64Load, wasmir.InstrF64Store,
			wasmir.InstrV128Load, wasmir.InstrV128Store,
			wasmir.InstrV128Load8x8S, wasmir.InstrV128Load8x8U,
			wasmir.InstrV128Load16x4S, wasmir.InstrV128Load16x4U,
			wasmir.InstrV128Load32x2S, wasmir.InstrV128Load32x2U,
			wasmir.InstrV128Load8Splat, wasmir.InstrV128Load16Splat,
			wasmir.InstrV128Load32Splat, wasmir.InstrV128Load64Splat,
			wasmir.InstrV128Load32Zero, wasmir.InstrV128Load64Zero:
			op.index = ins.MemoryIndex
			op.bits = int64(ins.MemoryOffset)
		case wasmir.InstrV128Load8Lane, wasmir.InstrV128Load16Lane,
			wasmir.InstrV128Load32Lane, wasmir.InstrV128Load64Lane,
			wasmir.InstrV128Store8Lane, wasmir.InstrV128Store16Lane,
			wasmir.InstrV128Store32Lane, wasmir.InstrV128Store64Lane:
			op.index = uint32(len(out.laneMemories))
			out.laneMemories = append(out.laneMemories, laneMemoryImmediate{
				memoryIndex: ins.MemoryIndex,
				offset:      ins.MemoryOffset,
				lane:        ins.LaneIndex,
			})
		case wasmir.InstrBr, wasmir.InstrBrIf, wasmir.InstrBrOnNull, wasmir.InstrBrOnNonNull:
			target, err := compileBranchTarget(ins.BranchDepth, labelStack, ctrl, finalEnd)
			if err != nil {
				return nil, fmt.Errorf("%s at %d: %w", instrName(ins.Kind), pc, err)
			}
			op.target = target.pc
			op.bits = int64(target.depth)
		case wasmir.InstrBrTable:
			targets := make([]branchTarget, 0, len(ins.BranchTable)+1)
			for i, depth := range ins.BranchTable {
				target, err := compileBranchTarget(depth, labelStack, ctrl, finalEnd)
				if err != nil {
					return nil, fmt.Errorf("br_table at %d target %d: %w", pc, i, err)
				}
				targets = append(targets, target)
			}
			target, err := compileBranchTarget(ins.BranchDefault, labelStack, ctrl, finalEnd)
			if err != nil {
				return nil, fmt.Errorf("br_table at %d default target: %w", pc, err)
			}
			targets = append(targets, target)
			op.index = uint32(len(out.branchTables))
			out.branchTables = append(out.branchTables, targets)
		case wasmir.InstrI32Const:
			op.bits = int64(ins.I32Const)
		case wasmir.InstrI64Const:
			op.bits = ins.I64Const
		case wasmir.InstrF32Const:
			op.bits = int64(ins.F32Const)
		case wasmir.InstrF64Const:
			op.bits = int64(ins.F64Const)
		case wasmir.InstrV128Const:
			op.index = uint32(len(out.v128Consts))
			out.v128Consts = append(out.v128Consts, ins.V128Const)
		case wasmir.InstrI8x16Shuffle:
			op.index = uint32(len(out.shuffleLanes))
			out.shuffleLanes = append(out.shuffleLanes, ins.ShuffleLanes)
		case wasmir.InstrI8x16ExtractLaneS, wasmir.InstrI8x16ExtractLaneU, wasmir.InstrI8x16ReplaceLane,
			wasmir.InstrI16x8ExtractLaneS, wasmir.InstrI16x8ExtractLaneU, wasmir.InstrI16x8ReplaceLane,
			wasmir.InstrI32x4ExtractLane, wasmir.InstrI32x4ReplaceLane,
			wasmir.InstrI64x2ExtractLane, wasmir.InstrI64x2ReplaceLane,
			wasmir.InstrF32x4ExtractLane, wasmir.InstrF32x4ReplaceLane,
			wasmir.InstrF64x2ExtractLane, wasmir.InstrF64x2ReplaceLane:
			op.index = ins.LaneIndex
		case wasmir.InstrI32Add, wasmir.InstrI32Sub, wasmir.InstrI32Mul,
			wasmir.InstrI32DivS, wasmir.InstrI32DivU, wasmir.InstrI32RemS, wasmir.InstrI32RemU,
			wasmir.InstrI32And, wasmir.InstrI32Or, wasmir.InstrI32Xor,
			wasmir.InstrI32Shl, wasmir.InstrI32ShrS, wasmir.InstrI32ShrU,
			wasmir.InstrI32Rotl, wasmir.InstrI32Rotr,
			wasmir.InstrI32Clz, wasmir.InstrI32Ctz, wasmir.InstrI32Popcnt,
			wasmir.InstrI32Extend8S, wasmir.InstrI32Extend16S,
			wasmir.InstrI32Eq, wasmir.InstrI32Ne,
			wasmir.InstrI32LtS, wasmir.InstrI32LtU, wasmir.InstrI32LeS, wasmir.InstrI32LeU,
			wasmir.InstrI32GtS, wasmir.InstrI32GtU, wasmir.InstrI32GeS, wasmir.InstrI32GeU,
			wasmir.InstrI32Eqz,
			wasmir.InstrI64Add, wasmir.InstrI64Sub, wasmir.InstrI64Mul,
			wasmir.InstrI64DivS, wasmir.InstrI64DivU, wasmir.InstrI64RemS, wasmir.InstrI64RemU,
			wasmir.InstrI64And, wasmir.InstrI64Or, wasmir.InstrI64Xor,
			wasmir.InstrI64Shl, wasmir.InstrI64ShrS, wasmir.InstrI64ShrU,
			wasmir.InstrI64Rotl, wasmir.InstrI64Rotr,
			wasmir.InstrI64Clz, wasmir.InstrI64Ctz, wasmir.InstrI64Popcnt,
			wasmir.InstrI64Extend8S, wasmir.InstrI64Extend16S, wasmir.InstrI64Extend32S,
			wasmir.InstrI32WrapI64,
			wasmir.InstrI32TruncF32S, wasmir.InstrI32TruncF32U,
			wasmir.InstrI32TruncF64S, wasmir.InstrI32TruncF64U,
			wasmir.InstrI32TruncSatF32S, wasmir.InstrI32TruncSatF32U,
			wasmir.InstrI32TruncSatF64S, wasmir.InstrI32TruncSatF64U,
			wasmir.InstrI64ExtendI32S, wasmir.InstrI64ExtendI32U,
			wasmir.InstrI64TruncF32S, wasmir.InstrI64TruncF32U,
			wasmir.InstrI64TruncF64S, wasmir.InstrI64TruncF64U,
			wasmir.InstrI64TruncSatF32S, wasmir.InstrI64TruncSatF32U,
			wasmir.InstrI64TruncSatF64S, wasmir.InstrI64TruncSatF64U,
			wasmir.InstrI64Eq, wasmir.InstrI64Ne,
			wasmir.InstrI64LtS, wasmir.InstrI64LtU, wasmir.InstrI64LeS, wasmir.InstrI64LeU,
			wasmir.InstrI64GtS, wasmir.InstrI64GtU, wasmir.InstrI64GeS, wasmir.InstrI64GeU,
			wasmir.InstrI64Eqz,
			wasmir.InstrF32ConvertI32S, wasmir.InstrF32ConvertI32U,
			wasmir.InstrF32ConvertI64S, wasmir.InstrF32ConvertI64U,
			wasmir.InstrF32DemoteF64,
			wasmir.InstrF32Abs, wasmir.InstrF32Neg, wasmir.InstrF32Sqrt,
			wasmir.InstrF32Ceil, wasmir.InstrF32Floor, wasmir.InstrF32Trunc, wasmir.InstrF32Nearest,
			wasmir.InstrF32Add, wasmir.InstrF32Sub, wasmir.InstrF32Mul, wasmir.InstrF32Div,
			wasmir.InstrF32Min, wasmir.InstrF32Max, wasmir.InstrF32Copysign,
			wasmir.InstrF32Eq, wasmir.InstrF32Ne,
			wasmir.InstrF32Lt, wasmir.InstrF32Le, wasmir.InstrF32Gt, wasmir.InstrF32Ge,
			wasmir.InstrF64ConvertI32S, wasmir.InstrF64ConvertI32U,
			wasmir.InstrF64ConvertI64S, wasmir.InstrF64ConvertI64U,
			wasmir.InstrF64PromoteF32,
			wasmir.InstrF64Abs, wasmir.InstrF64Neg, wasmir.InstrF64Sqrt,
			wasmir.InstrF64Ceil, wasmir.InstrF64Floor, wasmir.InstrF64Trunc, wasmir.InstrF64Nearest,
			wasmir.InstrF64Add, wasmir.InstrF64Sub, wasmir.InstrF64Mul, wasmir.InstrF64Div,
			wasmir.InstrF64Min, wasmir.InstrF64Max, wasmir.InstrF64Copysign,
			wasmir.InstrF64Eq, wasmir.InstrF64Ne,
			wasmir.InstrF64Lt, wasmir.InstrF64Le, wasmir.InstrF64Gt, wasmir.InstrF64Ge,
			wasmir.InstrI32ReinterpretF32, wasmir.InstrI64ReinterpretF64,
			wasmir.InstrF32ReinterpretI32, wasmir.InstrF64ReinterpretI64,
			wasmir.InstrDrop, wasmir.InstrNop, wasmir.InstrUnreachable,
			wasmir.InstrThrowRef,
			wasmir.InstrRefIsNull, wasmir.InstrRefAsNonNull,
			wasmir.InstrRefEq, wasmir.InstrExternConvertAny, wasmir.InstrAnyConvertExtern,
			wasmir.InstrI8x16Splat, wasmir.InstrI16x8Splat, wasmir.InstrI32x4Splat,
			wasmir.InstrI64x2Splat, wasmir.InstrF32x4Splat, wasmir.InstrF64x2Splat,
			wasmir.InstrV128AnyTrue, wasmir.InstrV128Not,
			wasmir.InstrI8x16Abs, wasmir.InstrI8x16Popcnt, wasmir.InstrI8x16Neg,
			wasmir.InstrI16x8ExtaddPairwiseI8x16S, wasmir.InstrI16x8ExtaddPairwiseI8x16U,
			wasmir.InstrI16x8Abs, wasmir.InstrI16x8Neg,
			wasmir.InstrI16x8ExtendLowI8x16S, wasmir.InstrI16x8ExtendLowI8x16U,
			wasmir.InstrI16x8ExtendHighI8x16S, wasmir.InstrI16x8ExtendHighI8x16U,
			wasmir.InstrI32x4ExtaddPairwiseI16x8S, wasmir.InstrI32x4ExtaddPairwiseI16x8U,
			wasmir.InstrI32x4Abs, wasmir.InstrI32x4Neg,
			wasmir.InstrI32x4ExtendLowI16x8S, wasmir.InstrI32x4ExtendLowI16x8U,
			wasmir.InstrI32x4ExtendHighI16x8S, wasmir.InstrI32x4ExtendHighI16x8U,
			wasmir.InstrI64x2Abs, wasmir.InstrI64x2Neg,
			wasmir.InstrI64x2ExtendLowI32x4S, wasmir.InstrI64x2ExtendLowI32x4U,
			wasmir.InstrI64x2ExtendHighI32x4S, wasmir.InstrI64x2ExtendHighI32x4U,
			wasmir.InstrI8x16AllTrue, wasmir.InstrI16x8AllTrue, wasmir.InstrI32x4AllTrue, wasmir.InstrI64x2AllTrue,
			wasmir.InstrI8x16Bitmask, wasmir.InstrI16x8Bitmask, wasmir.InstrI32x4Bitmask, wasmir.InstrI64x2Bitmask,
			wasmir.InstrF32x4Abs, wasmir.InstrF32x4Neg, wasmir.InstrF32x4Sqrt,
			wasmir.InstrF32x4Ceil, wasmir.InstrF32x4Floor, wasmir.InstrF32x4Trunc, wasmir.InstrF32x4Nearest,
			wasmir.InstrF64x2Abs, wasmir.InstrF64x2Neg, wasmir.InstrF64x2Sqrt,
			wasmir.InstrF64x2Ceil, wasmir.InstrF64x2Floor, wasmir.InstrF64x2Trunc, wasmir.InstrF64x2Nearest,
			wasmir.InstrF32x4ConvertI32x4S, wasmir.InstrF32x4ConvertI32x4U,
			wasmir.InstrF64x2ConvertLowI32x4S, wasmir.InstrF64x2ConvertLowI32x4U,
			wasmir.InstrF32x4DemoteF64x2Zero, wasmir.InstrF64x2PromoteLowF32x4,
			wasmir.InstrI32x4TruncSatF32x4S, wasmir.InstrI32x4TruncSatF32x4U,
			wasmir.InstrI32x4TruncSatF64x2SZero, wasmir.InstrI32x4TruncSatF64x2UZero,
			wasmir.InstrI32x4RelaxedTruncF32x4S, wasmir.InstrI32x4RelaxedTruncF32x4U,
			wasmir.InstrI32x4RelaxedTruncF64x2SZero, wasmir.InstrI32x4RelaxedTruncF64x2UZero,
			wasmir.InstrI8x16Swizzle, wasmir.InstrI8x16RelaxedSwizzle,
			wasmir.InstrV128And, wasmir.InstrV128AndNot, wasmir.InstrV128Or, wasmir.InstrV128Xor,
			wasmir.InstrI8x16NarrowI16x8S, wasmir.InstrI8x16NarrowI16x8U,
			wasmir.InstrI8x16Add, wasmir.InstrI8x16AddSatS, wasmir.InstrI8x16AddSatU,
			wasmir.InstrI8x16Sub, wasmir.InstrI8x16SubSatS, wasmir.InstrI8x16SubSatU,
			wasmir.InstrI8x16MinS, wasmir.InstrI8x16MinU, wasmir.InstrI8x16MaxS, wasmir.InstrI8x16MaxU, wasmir.InstrI8x16AvgrU,
			wasmir.InstrI16x8NarrowI32x4S, wasmir.InstrI16x8NarrowI32x4U,
			wasmir.InstrI16x8Add, wasmir.InstrI16x8AddSatS, wasmir.InstrI16x8AddSatU,
			wasmir.InstrI16x8Sub, wasmir.InstrI16x8SubSatS, wasmir.InstrI16x8SubSatU, wasmir.InstrI16x8Mul,
			wasmir.InstrI16x8MinS, wasmir.InstrI16x8MinU, wasmir.InstrI16x8MaxS, wasmir.InstrI16x8MaxU, wasmir.InstrI16x8AvgrU,
			wasmir.InstrI16x8Q15mulrSatS,
			wasmir.InstrI16x8RelaxedQ15mulrS, wasmir.InstrI16x8RelaxedDotI8x16I7x16S,
			wasmir.InstrI16x8ExtmulLowI8x16S, wasmir.InstrI16x8ExtmulHighI8x16S,
			wasmir.InstrI16x8ExtmulLowI8x16U, wasmir.InstrI16x8ExtmulHighI8x16U,
			wasmir.InstrI32x4Add, wasmir.InstrI32x4Sub, wasmir.InstrI32x4Mul,
			wasmir.InstrI32x4MinS, wasmir.InstrI32x4MinU, wasmir.InstrI32x4MaxS, wasmir.InstrI32x4MaxU,
			wasmir.InstrI32x4DotI16x8S,
			wasmir.InstrI32x4ExtmulLowI16x8S, wasmir.InstrI32x4ExtmulHighI16x8S,
			wasmir.InstrI32x4ExtmulLowI16x8U, wasmir.InstrI32x4ExtmulHighI16x8U,
			wasmir.InstrI64x2Add, wasmir.InstrI64x2Sub, wasmir.InstrI64x2Mul,
			wasmir.InstrI64x2ExtmulLowI32x4S, wasmir.InstrI64x2ExtmulHighI32x4S,
			wasmir.InstrI64x2ExtmulLowI32x4U, wasmir.InstrI64x2ExtmulHighI32x4U,
			wasmir.InstrF32x4Min, wasmir.InstrF32x4Max, wasmir.InstrF32x4Pmin, wasmir.InstrF32x4Pmax,
			wasmir.InstrF32x4RelaxedMin, wasmir.InstrF32x4RelaxedMax,
			wasmir.InstrF32x4Add, wasmir.InstrF32x4Sub, wasmir.InstrF32x4Div, wasmir.InstrF32x4Mul,
			wasmir.InstrF64x2Min, wasmir.InstrF64x2Max, wasmir.InstrF64x2Pmin, wasmir.InstrF64x2Pmax,
			wasmir.InstrF64x2RelaxedMin, wasmir.InstrF64x2RelaxedMax,
			wasmir.InstrF64x2Add, wasmir.InstrF64x2Sub, wasmir.InstrF64x2Div, wasmir.InstrF64x2Mul,
			wasmir.InstrI8x16Shl, wasmir.InstrI8x16ShrS, wasmir.InstrI8x16ShrU,
			wasmir.InstrI16x8Shl, wasmir.InstrI16x8ShrS, wasmir.InstrI16x8ShrU,
			wasmir.InstrI32x4Shl, wasmir.InstrI32x4ShrS, wasmir.InstrI32x4ShrU,
			wasmir.InstrI64x2Shl, wasmir.InstrI64x2ShrS, wasmir.InstrI64x2ShrU,
			wasmir.InstrI8x16Eq, wasmir.InstrI8x16Ne, wasmir.InstrI8x16LtS, wasmir.InstrI8x16LtU,
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
			wasmir.InstrF64x2Gt, wasmir.InstrF64x2Le, wasmir.InstrF64x2Ge,
			wasmir.InstrV128Bitselect,
			wasmir.InstrI8x16RelaxedLaneselect, wasmir.InstrI16x8RelaxedLaneselect,
			wasmir.InstrI32x4RelaxedLaneselect, wasmir.InstrI64x2RelaxedLaneselect,
			wasmir.InstrF32x4RelaxedMadd, wasmir.InstrF32x4RelaxedNmadd,
			wasmir.InstrF64x2RelaxedMadd, wasmir.InstrF64x2RelaxedNmadd,
			wasmir.InstrI32x4RelaxedDotI8x16I7x16AddS,
			wasmir.InstrReturn:
		case wasmir.InstrEnd:
			if len(labelStack) == 0 {
				if pc != len(fn.Body)-1 {
					return nil, fmt.Errorf("end without active label")
				}
			} else {
				start := labelStack[len(labelStack)-1]
				label := ctrl.labels[start]
				if label.endIndex == pc {
					labelStack = labelStack[:len(labelStack)-1]
				}
			}
		default:
			return nil, fmt.Errorf("unsupported instruction %s", instrName(ins.Kind))
		}
		out.code[pc] = op
	}

	if len(labelStack) != 0 {
		start := labelStack[len(labelStack)-1]
		return nil, fmt.Errorf("%s at %d without matching end", instrName(fn.Body[start].Kind), start)
	}
	return out, nil
}

// setControlLabelSignature records the runtime stack contract for one
// structured-control label.
func setControlLabelSignature(m *wasmir.Module, ins wasmir.Instruction, label controlLabel, labels map[int]controlLabel, pc int) error {
	params, results, err := controlSignature(m, ins)
	if err != nil {
		return fmt.Errorf("%s at %d: %w", instrName(ins.Kind), pc, err)
	}
	label.paramArity = len(params)
	label.resultArity = len(results)
	label.branchArity = len(results)
	if ins.Kind == wasmir.InstrLoop {
		label.isLoop = true
		label.branchArity = len(params)
	}
	labels[pc] = label
	return nil
}

// setControlLabelCatches records compiled catch targets for a try_table label.
func setControlLabelCatches(ins wasmir.Instruction, label controlLabel, labelStack []int, ctrl controlInfo, finalEnd int, pc int) error {
	label = ctrl.labels[pc]
	label.catches = make([]catchTarget, 0, len(ins.TryTableCatches))
	for i, catch := range ins.TryTableCatches {
		target, err := compileBranchTarget(catch.LabelDepth, labelStack, ctrl, finalEnd)
		if err != nil {
			return fmt.Errorf("try_table at %d catch %d: %w", pc, i, err)
		}
		label.catches = append(label.catches, catchTarget{
			kind:     catch.Kind,
			tagIndex: catch.TagIndex,
			target:   target,
		})
	}
	ctrl.labels[pc] = label
	return nil
}

// controlSignature resolves a structured-control instruction's block type.
func controlSignature(m *wasmir.Module, ins wasmir.Instruction) ([]wasmir.ValueType, []wasmir.ValueType, error) {
	if ins.BlockTypeUsesIndex {
		if int(ins.BlockTypeIndex) >= len(m.Types) {
			return nil, nil, fmt.Errorf("block type index %d out of range", ins.BlockTypeIndex)
		}
		ft := m.Types[ins.BlockTypeIndex]
		if ft.Kind != wasmir.TypeDefKindFunc {
			return nil, nil, fmt.Errorf("block type index %d is not a function type", ins.BlockTypeIndex)
		}
		return ft.Params, ft.Results, nil
	}
	if ins.BlockType != nil {
		return nil, []wasmir.ValueType{*ins.BlockType}, nil
	}
	return nil, nil, nil
}

// compileBranchTarget resolves a label-depth immediate to a branch target.
func compileBranchTarget(depth uint32, labelStack []int, ctrl controlInfo, finalEnd int) (branchTarget, error) {
	if int(depth) > len(labelStack) {
		return branchTarget{}, fmt.Errorf("branch depth %d out of range", depth)
	}
	if int(depth) == len(labelStack) {
		return branchTarget{pc: finalEnd - 1, depth: depth}, nil
	}
	start := labelStack[len(labelStack)-1-int(depth)]
	label, ok := ctrl.labels[start]
	if !ok {
		return branchTarget{}, fmt.Errorf("branch target at %d has no matching end", start)
	}
	return branchTarget{pc: label.branchTarget, depth: depth}, nil
}

// controlLabel describes one structured-control label in the flattened
// instruction stream.
//
// endIndex is the matching end instruction. branchTarget is the instruction pc
// assigned to br/br_if/br_table for this label; it is endIndex for block and if,
// but the loop instruction pc for loop, so the interpreter's pc increment
// re-enters the loop body. elseIndex is the matching else instruction for if
// labels, or -1 when the label has no else arm.
type controlLabel struct {
	endIndex     int
	branchTarget int
	elseIndex    int
	paramArity   int
	resultArity  int
	branchArity  int
	isLoop       bool
	catches      []catchTarget
}

// controlInfo stores precomputed control-boundary metadata by opening
// instruction index. Only block, loop, and if instructions have entries.
type controlInfo struct {
	labels map[int]controlLabel
}

// analyzeControl records matching structured-control boundaries in body.
//
// The VM assumes modules were validated before instantiation, but it still
// receives a plain wasmir.Module and should not rely on nested source syntax.
// This pass treats block/loop/if as openers, else as metadata on the current
// if, and end as the closer for the innermost opener. End instructions with no
// opener are accepted here because the final function end is represented the
// same way as a structured-control end.
func analyzeControl(body []wasmir.Instruction) (controlInfo, error) {
	ctrl := controlInfo{labels: make(map[int]controlLabel)}
	stack := make([]int, 0)
	elseIndex := make(map[int]int)

	for pc, ins := range body {
		switch ins.Kind {
		case wasmir.InstrBlock, wasmir.InstrLoop, wasmir.InstrIf, wasmir.InstrTryTable:
			stack = append(stack, pc)
		case wasmir.InstrElse:
			if len(stack) == 0 {
				return controlInfo{}, fmt.Errorf("else at %d without matching if", pc)
			}
			start := stack[len(stack)-1]
			if body[start].Kind != wasmir.InstrIf {
				return controlInfo{}, fmt.Errorf("else at %d matched non-if instruction %s", pc, instrName(body[start].Kind))
			}
			elseIndex[start] = pc
		case wasmir.InstrEnd:
			if len(stack) == 0 {
				continue
			}
			start := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			label := controlLabel{
				endIndex:     pc,
				branchTarget: pc,
				elseIndex:    -1,
			}
			if body[start].Kind == wasmir.InstrLoop {
				label.branchTarget = start
			}
			if elsePC, ok := elseIndex[start]; ok {
				label.elseIndex = elsePC
			}
			ctrl.labels[start] = label
		}
	}
	if len(stack) != 0 {
		return controlInfo{}, fmt.Errorf("%s at %d without matching end", instrName(body[stack[len(stack)-1]].Kind), stack[len(stack)-1])
	}
	return ctrl, nil
}

// instrName formats instruction kinds for VM errors.
func instrName(kind wasmir.InstrKind) string {
	if def, ok := instrdef.LookupInstructionByKind(kind); ok {
		return def.TextName
	}
	return fmt.Sprintf("instruction(%d)", kind)
}
