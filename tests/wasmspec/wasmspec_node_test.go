package tests

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// wasmSpecNodeBackend returns the Node.js backend used for broad wasmspec
// execution coverage.
func wasmSpecNodeBackend() wasmSpecBackend {
	return wasmSpecBackend{
		name:                "node",
		requiresIntegration: true,
		run:                 runWasmSpecNodeScript,
	}
}

// runWasmSpecNodeScript executes parsed commands through the Node.js
// WebAssembly engine bridge.
func runWasmSpecNodeScript(t *testing.T, scriptPath string, commands []scriptCommand, opts runOptions) []commandResult {
	t.Helper()

	runner, err := newScriptRunner(context.Background())
	if err != nil {
		t.Fatalf("spec runner bootstrap failed: %v", err)
	}
	defer func() {
		if wasmSpecDebugEnabled() {
			fmt.Fprintf(os.Stderr, "[wasmspec node %s] closing node runner\n", filepath.ToSlash(scriptPath))
		}
		if closeErr := runner.closeWithLogf(opts.logf); closeErr != nil {
			t.Fatalf("spec runner close failed: %v", closeErr)
		}
		if wasmSpecDebugEnabled() {
			fmt.Fprintf(os.Stderr, "[wasmspec node %s] closed node runner\n", filepath.ToSlash(scriptPath))
		}
	}()

	results := runner.run(commands, opts)
	if wasmSpecDebugEnabled() {
		fmt.Fprintf(os.Stderr, "[wasmspec node %s] finished runner.run\n", filepath.ToSlash(scriptPath))
	}
	return results
}
