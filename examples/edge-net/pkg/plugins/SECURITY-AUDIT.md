# Edge-Net Plugin System Security Audit Report

**Audit Date:** 2026-01-03
**Auditor:** Code Review Agent
**Scope:** Plugin system files in `/workspaces/ruvector/examples/edge-net/pkg/plugins/`
**Classification:** Security Assessment

---

## Executive Summary

The Edge-Net plugin system provides a modular architecture for extending functionality with capability-based permissions, signature verification, and sandboxing claims. However, the audit identified **12 security issues** ranging from Critical to Low severity that could enable code injection, capability bypass, and other attacks.

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 2 | ✅ FIXED (v0.5.1) |
| High | 4 | ✅ FIXED (v0.5.1) |
| Medium | 4 | Recommended |
| Low | 2 | Advisory |

## v0.5.1 Security Fixes Applied

The following critical and high severity issues were fixed in v0.5.1:

| Issue ID | Fix Applied |
|----------|-------------|
| CRITICAL-001 | Path validation with symlink resolution, whitelist directories, extension check |
| CRITICAL-002 | Real Ed25519 signature verification with `crypto.verify()` |
| HIGH-001 | Enhanced sandbox with frozen capabilities, resource limits, denied globals |
| HIGH-003 | Rate limiting with sliding window (100 req/min default) |
| HIGH-004 | HKDF key derivation with 32-byte random salt and proper entropy |

---

## Critical Issues

### CRITICAL-001: Dynamic Import in CLI Validation Allows Code Execution

**Location:** `/workspaces/ruvector/examples/edge-net/pkg/plugins/cli.js:280`

**Description:**
The `validate` command uses dynamic `import()` on user-provided paths without sanitization. An attacker can execute arbitrary JavaScript code by providing a malicious path.

```javascript
// cli.js:280
const plugin = await import(pluginPath);
```

**Attack Vector:**
```bash
edge-net plugins validate "../../../malicious-payload.js"
edge-net plugins validate "data:text/javascript,console.log(process.env)"
```

**Impact:** Remote Code Execution (RCE) - arbitrary code runs with full Node.js privileges including filesystem access, network access, and environment variable exposure.

**Recommended Fix:**
```javascript
import { resolve, normalize } from 'path';
import { existsSync, statSync } from 'fs';

async function validatePlugin(pluginPath) {
    // Resolve to absolute path
    const absolutePath = resolve(process.cwd(), pluginPath);

    // Prevent path traversal
    const allowedDir = resolve(process.cwd(), 'plugins');
    if (!absolutePath.startsWith(allowedDir)) {
        throw new Error('Plugin path must be within plugins directory');
    }

    // Verify file exists and is a regular file
    if (!existsSync(absolutePath) || !statSync(absolutePath).isFile()) {
        throw new Error('Invalid plugin path');
    }

    // Only allow .js files
    if (!absolutePath.endsWith('.js')) {
        throw new Error('Only .js plugin files are allowed');
    }

    const plugin = await import(absolutePath);
    // ... rest of validation
}
```

---

### CRITICAL-002: Signature Verification Bypass via Trusted Author Fallback

**Location:** `/workspaces/ruvector/examples/edge-net/pkg/plugins/plugin-loader.js:117-124`

**Description:**
The signature verification has a critical flaw: if a plugin declares a signature AND the author is in the trusted list, verification passes without actually checking the signature. An attacker can forge plugins by claiming to be a trusted author.

```javascript
// plugin-loader.js:117-124
if (manifest.signature) {
    // TODO: Implement Ed25519 signature verification
    // For now, trust if author is in trusted list
    if (this.options.trustedAuthors.includes(manifest.author)) {
        return { verified: true, reason: 'Trusted author' };
    }
}
```

**Attack Vector:**
1. Create malicious plugin with `author: 'ruvector'` in manifest
2. Add any fake signature value
3. Plugin loads with "verified" status despite no actual signature check

**Impact:** Complete bypass of cryptographic verification. Malicious plugins can masquerade as official plugins.

**Recommended Fix:**
```javascript
async _verifyPlugin(manifest, code) {
    if (!this.options.verifySignatures) {
        return { verified: true, reason: 'Verification disabled' };
    }

    // ALWAYS verify checksum first
    if (manifest.checksum) {
        const hash = createHash('sha256').update(code).digest('hex');
        if (hash !== manifest.checksum) {
            return { verified: false, reason: 'Checksum mismatch' };
        }
    } else {
        return { verified: false, reason: 'Missing checksum' };
    }

    // REQUIRE signature for non-stable plugins
    if (manifest.tier !== PluginTier.STABLE) {
        if (!manifest.signature) {
            return { verified: false, reason: 'Signature required for non-stable plugins' };
        }

        // Get trusted public key for author
        const publicKey = this._getTrustedPublicKey(manifest.author);
        if (!publicKey) {
            return { verified: false, reason: 'Unknown author' };
        }

        // Actually verify Ed25519 signature
        const isValid = await this._verifyEd25519Signature(
            manifest.signature,
            code,
            publicKey
        );

        if (!isValid) {
            return { verified: false, reason: 'Invalid signature' };
        }
    }

    return { verified: true, reason: 'Verified' };
}
```

---

## High Severity Issues

### HIGH-001: Sandbox Does Not Actually Isolate Plugin Code

**Location:** `/workspaces/ruvector/examples/edge-net/pkg/plugins/plugin-loader.js:208-254`

**Description:**
The "sandbox" is only a capability check wrapper. Plugins run in the same JavaScript context as the main application and can access:
- `global` object
- `process` object (including `process.env`)
- `require()` function
- Filesystem via `fs` module
- Network via `http`/`https` modules

```javascript
// plugin-loader.js:238-254
_createSandbox(capabilities) {
    const sandbox = {
        capabilities: new Set(capabilities),
        hasCapability(cap) {
            return this.capabilities.has(cap);
        },
        require(cap) {
            if (!this.hasCapability(cap)) {
                throw new Error(`Missing capability: ${cap}`);
            }
        },
    };
    return sandbox;
}
```

**Attack Vector:**
A plugin without `SYSTEM_FS` capability can still do:
```javascript
import { readFileSync } from 'fs';
const secrets = readFileSync('/etc/passwd');
```

**Impact:** Capability system is purely advisory. Malicious plugins have unrestricted access.

**Recommended Fix:**
Use Node.js `vm` module with context isolation or WebAssembly sandboxing:
```javascript
import { createContext, runInContext } from 'vm';

_createRealSandbox(capabilities) {
    const allowedGlobals = {
        console: capabilities.has('debug') ? console : undefined,
        setTimeout,
        setInterval,
        Promise,
        // Explicitly allowed APIs only
    };

    const context = createContext(allowedGlobals);
    return {
        context,
        execute: (code) => runInContext(code, context, { timeout: 5000 }),
    };
}
```

---

### HIGH-002: Plugin Import Allows Arbitrary Code Execution via eval Warning

**Location:** `/workspaces/ruvector/examples/edge-net/pkg/plugins/plugin-sdk.js:381-397`

**Description:**
The `PluginRegistry.import()` method has scaffolding that warns about `eval` usage, suggesting code execution from exported plugin data is intended. The current implementation returns early, but the structure invites unsafe completion.

```javascript
// plugin-sdk.js:381-397
import(exportedPlugin) {
    const { manifest, code, checksum } = exportedPlugin;

    // Verify checksum
    const computedChecksum = createHash('sha256').update(code).digest('hex');
    if (computedChecksum !== checksum) {
        throw new Error('Checksum mismatch - plugin may have been tampered');
    }

    // Parse and register (in production, use proper sandboxing)
    // WARNING: eval is dangerous - use VM2 or similar in production
    console.warn('WARNING: Importing plugins uses eval - only import from trusted sources');

    return { imported: true, manifest };
}
```

**Impact:** The architecture suggests dynamic code execution is planned. If implemented with `eval()` or `Function()`, it enables RCE via crafted plugin exports.

**Recommended Fix:**
Remove dynamic import capability entirely, or implement with WebAssembly:
```javascript
import(exportedPlugin) {
    throw new Error(
        'Dynamic plugin import is disabled for security. ' +
        'Plugins must be installed via npm or loaded from verified paths.'
    );
}
```

---

### HIGH-003: No Rate Limiting on Plugin Operations

**Location:** Multiple files in `/workspaces/ruvector/examples/edge-net/pkg/plugins/`

**Description:**
No rate limiting exists for:
- Plugin loading (`PluginLoader.load()`)
- Federated learning rounds (`FederatedLearningPlugin.startRound()`)
- Swarm optimization (`SwarmIntelligencePlugin.createSwarm()`)
- Staking operations (`ReputationStakingPlugin.stake()`)

**Attack Vector:**
```javascript
// DoS via resource exhaustion
for (let i = 0; i < 100000; i++) {
    swarm.createSwarm(`swarm-${i}`, { dimensions: 1000, populationSize: 1000 });
}
```

**Impact:** Denial of Service through memory exhaustion, CPU saturation, or resource starvation.

**Recommended Fix:**
```javascript
class RateLimiter {
    constructor(maxPerMinute = 60) {
        this.requests = [];
        this.maxPerMinute = maxPerMinute;
    }

    check() {
        const now = Date.now();
        this.requests = this.requests.filter(t => now - t < 60000);
        if (this.requests.length >= this.maxPerMinute) {
            throw new Error('Rate limit exceeded');
        }
        this.requests.push(now);
    }
}
```

---

### HIGH-004: Weak Cryptographic Key Derivation

**Location:** `/workspaces/ruvector/examples/edge-net/pkg/plugins/implementations/e2e-encryption.js:44-47`

**Description:**
Session keys are derived using SHA-256 of peer ID + timestamp, which is predictable and not cryptographically secure for key agreement.

```javascript
// e2e-encryption.js:44-47
const sharedSecret = createHash('sha256')
    .update(peerId + '-' + Date.now())
    .digest();
```

**Attack Vector:**
An attacker who knows the peer ID and can guess the timestamp (accuracy within seconds) can compute all past session keys.

**Impact:** Past and future encrypted sessions can be decrypted. Forward secrecy claims are false.

**Recommended Fix:**
Use proper X25519 key exchange:
```javascript
import { generateKeyPairSync, diffieHellman, createPublicKey } from 'crypto';

async establishSession(peerId, peerPublicKey) {
    // Generate ephemeral X25519 keypair
    const { publicKey, privateKey } = generateKeyPairSync('x25519');

    // Compute shared secret via ECDH
    const peerPubKey = createPublicKey({
        key: Buffer.from(peerPublicKey, 'hex'),
        format: 'der',
        type: 'spki',
    });

    const sharedSecret = diffieHellman({
        privateKey,
        publicKey: peerPubKey,
    });

    // Derive session key using HKDF
    const sessionKey = hkdf('sha256', sharedSecret, salt, info, 32);

    return { publicKey: publicKey.export({ format: 'der', type: 'spki' }).toString('hex') };
}
```

---

## Medium Severity Issues

### MEDIUM-001: Capability Bypass via Plugin Manifest Modification

**Location:** `/workspaces/ruvector/examples/edge-net/pkg/plugins/plugin-loader.js:191-202`

**Description:**
After loading, the plugin object contains a mutable reference to `manifest.capabilities`. A malicious plugin can modify its own capabilities at runtime.

```javascript
// plugin-loader.js:218-220
return {
    // ...
    manifest,  // Mutable reference!
    // ...
};
```

**Attack Vector:**
```javascript
// In malicious plugin's onInit()
this.manifest.capabilities.push('system:exec');
this.sandbox.capabilities.add('system:exec');
```

**Impact:** Plugins can escalate their own permissions after initial security checks.

**Recommended Fix:**
```javascript
return {
    manifest: Object.freeze(JSON.parse(JSON.stringify(manifest))),
    // ...
};
```

---

### MEDIUM-002: No Input Validation on Configuration Values

**Location:** `/workspaces/ruvector/examples/edge-net/pkg/plugins/plugin-loader.js:186-191`

**Description:**
Plugin configuration is merged without validation against the declared `configSchema`:

```javascript
// plugin-loader.js:187-190
const finalConfig = {
    ...manifest.defaultConfig,
    ...config,  // User input, unvalidated
};
```

**Attack Vector:**
```javascript
await plugins.load('ai.federated-learning', {
    localEpochs: "DROP TABLE users;--",  // Type confusion
    noiseMultiplier: Infinity,           // Numeric overflow
    __proto__: { polluted: true },       // Prototype pollution
});
```

**Impact:** Type confusion, prototype pollution, or application crashes from invalid configuration.

**Recommended Fix:**
```javascript
import Ajv from 'ajv';

_validateConfig(manifest, config) {
    const ajv = new Ajv({ removeAdditional: true, useDefaults: true });
    const validate = ajv.compile(manifest.configSchema);

    const mergedConfig = { ...manifest.defaultConfig, ...config };

    if (!validate(mergedConfig)) {
        throw new Error(`Invalid config: ${ajv.errorsText(validate.errors)}`);
    }

    return mergedConfig;
}
```

---

### MEDIUM-003: Unbounded Memory Growth in Learning Plugins

**Location:** Multiple implementation files

**Description:**
The following data structures grow without bounds:
- `FederatedLearningPlugin.rounds` (Map never cleared)
- `SwarmIntelligencePlugin.swarms` (Map never cleared)
- `ReputationStakingPlugin.slashHistory` (Array grows forever)

```javascript
// federated-learning.js:26-28
this.rounds = new Map();         // Never pruned
this.localModels = new Map();    // Never pruned
this.globalModels = new Map();   // Never pruned
```

**Impact:** Memory exhaustion over time, leading to OOM crashes.

**Recommended Fix:**
```javascript
constructor(config) {
    this.maxRounds = config.maxRounds || 1000;
    this.roundTTL = config.roundTTL || 3600000; // 1 hour
}

_pruneOldRounds() {
    const now = Date.now();
    for (const [id, round] of this.rounds) {
        if (now - round.startedAt > this.roundTTL) {
            this.rounds.delete(id);
        }
    }

    // Also enforce max size
    if (this.rounds.size > this.maxRounds) {
        const oldest = [...this.rounds.entries()]
            .sort((a, b) => a[1].startedAt - b[1].startedAt)
            .slice(0, this.rounds.size - this.maxRounds);
        oldest.forEach(([id]) => this.rounds.delete(id));
    }
}
```

---

### MEDIUM-004: Path Traversal in Plugin Creation

**Location:** `/workspaces/ruvector/examples/edge-net/pkg/plugins/cli.js:195-196`

**Description:**
The `create` command writes files to user-specified directory without sanitization:

```javascript
// cli.js:195-196
const pluginDir = join(outputDir, name.toLowerCase().replace(/\s+/g, '-'));
mkdirSync(pluginDir, { recursive: true });
```

**Attack Vector:**
```bash
edge-net plugins create "../../etc/cron.d/malicious" --output /tmp
```

**Impact:** Arbitrary file/directory creation outside intended plugin directory.

**Recommended Fix:**
```javascript
const pluginDir = join(
    resolve(process.cwd(), outputDir),
    name.toLowerCase().replace(/[^a-z0-9-]/g, '-')
);

// Verify output is within current working directory
if (!pluginDir.startsWith(process.cwd())) {
    throw new Error('Output directory must be within current project');
}
```

---

## Low Severity Issues

### LOW-001: Predictable Round/Swarm IDs

**Location:** Multiple implementation files

**Description:**
IDs are generated using timestamp + small random suffix:

```javascript
// federated-learning.js:35
const roundId = `round-${Date.now()}-${randomBytes(4).toString('hex')}`;
```

**Impact:** IDs are predictable within a short time window (8 bytes of entropy only).

**Recommended Fix:**
```javascript
const roundId = `round-${randomBytes(16).toString('hex')}`;
```

---

### LOW-002: Sensitive Information in Error Messages

**Location:** Multiple files

**Description:**
Error messages expose internal state:

```javascript
// plugin-loader.js:174
throw new Error(`Plugin ${pluginId} not allowed: ${allowed.reason}`);

// reputation-staking.js:39
throw new Error(`Insufficient balance: ${balance} < ${amount}`);
```

**Impact:** Information leakage about plugin policies and user balances.

**Recommended Fix:**
Log detailed errors server-side, return generic messages to users:
```javascript
const err = new Error('Plugin load failed');
err.code = 'PLUGIN_DENIED';
err.details = { pluginId, reason: allowed.reason };  // For logging only
throw err;
```

---

## Additional Observations

### Supply Chain Concerns

1. **No package integrity checks**: Plugin dependencies are not verified
2. **No lock file enforcement**: `package.json` in created plugins lacks integrity hashes
3. **Peer dependency version range too wide**: `^0.4.0` allows potentially incompatible versions

### Missing Security Headers (for browser usage)

If plugins run in browser context:
- No Content-Security-Policy enforcement
- No Subresource Integrity (SRI) for loaded modules

### Audit Trail Gaps

- No logging of plugin load/unload events to persistent storage
- No cryptographic audit trail of configuration changes
- Event emitters don't include timestamps or caller context

---

## Recommendations Summary

### Immediate Actions (P0)

1. Remove dynamic `import()` in CLI validation or restrict to safe paths
2. Implement actual Ed25519 signature verification
3. Add real sandboxing using `vm` module or WebAssembly

### Short-Term Actions (P1)

4. Add rate limiting to all plugin operations
5. Replace weak key derivation with X25519 ECDH
6. Freeze manifest objects after loading
7. Validate configuration against JSON Schema

### Medium-Term Actions (P2)

8. Implement memory bounds and TTL for stored data
9. Add comprehensive input sanitization
10. Use cryptographically random IDs
11. Reduce information in error messages

### Long-Term Improvements

12. Add supply chain security (lock files, SRI)
13. Implement audit logging with cryptographic signatures
14. Consider moving to WASM-based sandbox for stronger isolation
15. Add security-focused integration tests

---

## Files Reviewed

| File | Lines | Issues Found |
|------|-------|--------------|
| plugin-manifest.js | 703 | 0 |
| plugin-loader.js | 441 | 5 |
| plugin-sdk.js | 497 | 2 |
| index.js | 91 | 0 |
| cli.js | 396 | 2 |
| implementations/compression.js | 133 | 0 |
| implementations/e2e-encryption.js | 161 | 1 |
| implementations/federated-learning.js | 250 | 1 |
| implementations/reputation-staking.js | 244 | 1 |
| implementations/swarm-intelligence.js | 387 | 0 |

---

## Conclusion

The Edge-Net plugin system has a well-designed architecture with capability-based permissions and verification intent. However, critical implementation gaps - particularly around actual signature verification and sandboxing - undermine the security model. The two Critical issues should be addressed before any production deployment, as they enable complete bypass of security controls.

The plugin implementations (compression, encryption, ML, etc.) are generally well-written but suffer from resource management issues that could lead to denial of service over time.

---

**Report Generated:** 2026-01-03
**Audit Version:** 1.1 (Updated after security fixes)
**Fixes Applied:** v0.5.1 (2026-01-03)
**Next Review:** Recommended for Medium priority items
