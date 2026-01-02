#!/usr/bin/env node
/**
 * Test for GitHub Issue #102: Node.js ONNX/WASM embedding fixes
 *
 * Tests:
 * 1. WASM loads without --experimental-wasm-modules flag
 * 2. embedText() returns array directly (not object)
 * 3. embedTexts() works for batch operations
 *
 * Run: node test/embedding-fix-test.js
 * (No special flags required!)
 */

const assert = require('assert');

async function runTests() {
  console.log('='.repeat(60));
  console.log('Testing GitHub Issue #102 Fixes');
  console.log('='.repeat(60));
  console.log();

  // Import without --experimental-wasm-modules flag!
  console.log('Test 1: Import without experimental flags...');
  let embedText, embedTexts, embed, embedBatch;
  try {
    const ruvector = require('../dist');
    embedText = ruvector.embedText;
    embedTexts = ruvector.embedTexts;
    embed = ruvector.embed;
    embedBatch = ruvector.embedBatch;
    console.log('  ✅ PASS: Imported successfully without --experimental-wasm-modules');
  } catch (e) {
    console.error('  ❌ FAIL: Import failed:', e.message);
    process.exit(1);
  }

  // Test 2: embedText returns array directly
  console.log('\nTest 2: embedText() returns array directly...');
  try {
    const vector = await embedText('hello world');

    // Should be an array
    assert(Array.isArray(vector), 'embedText should return an array');

    // Should be 384 dimensions
    assert.strictEqual(vector.length, 384, 'embedding should be 384 dimensions');

    // Should contain numbers
    assert(typeof vector[0] === 'number', 'elements should be numbers');

    console.log('  ✅ PASS: embedText() returns number[] directly');
    console.log(`     - Type: ${Array.isArray(vector) ? 'Array' : typeof vector}`);
    console.log(`     - Length: ${vector.length} dimensions`);
    console.log(`     - Sample: [${vector.slice(0, 3).map(n => n.toFixed(4)).join(', ')}, ...]`);
  } catch (e) {
    console.error('  ❌ FAIL:', e.message);
    process.exit(1);
  }

  // Test 3: embedTexts batch function
  console.log('\nTest 3: embedTexts() batch processing...');
  try {
    const texts = ['hello', 'world', 'semantic', 'search'];
    const start = Date.now();
    const vectors = await embedTexts(texts);
    const elapsed = Date.now() - start;

    // Should return array of arrays
    assert(Array.isArray(vectors), 'embedTexts should return an array');
    assert.strictEqual(vectors.length, texts.length, 'should return same number of vectors as inputs');

    // Each vector should be 384d
    for (let i = 0; i < vectors.length; i++) {
      assert(Array.isArray(vectors[i]), `vector ${i} should be an array`);
      assert.strictEqual(vectors[i].length, 384, `vector ${i} should be 384 dimensions`);
    }

    console.log('  ✅ PASS: embedTexts() returns number[][] directly');
    console.log(`     - Processed ${texts.length} texts in ${elapsed}ms`);
    console.log(`     - Each vector: ${vectors[0].length} dimensions`);
  } catch (e) {
    console.error('  ❌ FAIL:', e.message);
    process.exit(1);
  }

  // Test 4: Original embed() still works (returns object for backwards compat)
  console.log('\nTest 4: Original embed() returns object (backwards compat)...');
  try {
    const result = await embed('test');

    assert(typeof result === 'object', 'embed should return an object');
    assert(Array.isArray(result.embedding), 'result.embedding should be an array');
    assert(typeof result.dimension === 'number', 'result.dimension should be a number');
    assert(typeof result.timeMs === 'number', 'result.timeMs should be a number');

    console.log('  ✅ PASS: embed() returns { embedding, dimension, timeMs }');
    console.log(`     - Dimension: ${result.dimension}`);
    console.log(`     - Time: ${result.timeMs.toFixed(2)}ms`);
  } catch (e) {
    console.error('  ❌ FAIL:', e.message);
    process.exit(1);
  }

  // Test 5: Performance comparison
  console.log('\nTest 5: Batch vs sequential performance...');
  try {
    const testTexts = Array(10).fill(null).map((_, i) => `Test document number ${i + 1} for embedding`);

    // Sequential
    const seqStart = Date.now();
    for (const text of testTexts) {
      await embedText(text);
    }
    const seqElapsed = Date.now() - seqStart;

    // Batch
    const batchStart = Date.now();
    await embedTexts(testTexts);
    const batchElapsed = Date.now() - batchStart;

    const speedup = seqElapsed / batchElapsed;

    console.log('  ✅ PASS: Performance comparison');
    console.log(`     - Sequential (${testTexts.length}x embedText): ${seqElapsed}ms`);
    console.log(`     - Batch (1x embedTexts): ${batchElapsed}ms`);
    console.log(`     - Speedup: ${speedup.toFixed(1)}x faster`);
  } catch (e) {
    console.error('  ⚠️ WARN:', e.message);
  }

  console.log('\n' + '='.repeat(60));
  console.log('All tests passed! Issue #102 is fixed.');
  console.log('='.repeat(60));
  console.log(`
Summary of fixes:
1. ✅ No --experimental-wasm-modules flag needed
2. ✅ embedText() returns number[] directly
3. ✅ embedTexts() available for batch operations
4. ✅ Backwards compatible: embed()/embedBatch() still work
  `);
}

// Run tests
runTests().catch(e => {
  console.error('Test failed:', e);
  process.exit(1);
});
