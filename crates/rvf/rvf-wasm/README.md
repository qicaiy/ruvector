# rvf-wasm

WASM microkernel for running RuVector Format operations in the browser and at the edge.

## Overview

`rvf-wasm` compiles the core RVF runtime to WebAssembly for use in browsers, Cloudflare Workers, and other WASM environments:

- **Compact binary** -- optimized with `opt-level = "z"` and LTO
- **No-std compatible** -- runs without a system allocator where needed
- **Browser-ready** -- works with `wasm-bindgen` or standalone instantiation

## Build

```bash
cargo build --target wasm32-unknown-unknown --release -p rvf-wasm
```

## License

MIT OR Apache-2.0
