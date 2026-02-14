# rvf-node

Node.js N-API bindings for native RuVector Format operations.

## Overview

`rvf-node` exposes the RVF runtime to Node.js via N-API for high-performance vector operations without leaving JavaScript:

- **Async API** -- non-blocking vector store operations
- **Native speed** -- Rust-compiled N-API addon, no serialization overhead
- **Cross-platform** -- builds for Linux, macOS, and Windows

## Build

```bash
cd crates/rvf/rvf-node
npm run build
```

## License

MIT OR Apache-2.0
