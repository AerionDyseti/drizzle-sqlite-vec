import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  serializeVector,
  deserializeVector,
  vec0Table,
  vecFloat,
} from '../src/index.js';
import { createTestContext, closeTestContext, type TestContext } from './setup.js';

describe('Serialization Round-trips', () => {
  let ctx: TestContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  afterEach(() => {
    closeTestContext(ctx);
  });

  it('should preserve number array values through Float32 round-trip', () => {
    const original = [0.1, 0.2, 0.3, 0.4, -0.5, 1.0, -1.0, 0.0];
    const buffer = serializeVector(original);

    expect(buffer).toBeInstanceOf(Buffer);
    expect(buffer.byteLength).toBe(original.length * 4);

    const restored = deserializeVector(buffer);
    expect(restored).toHaveLength(original.length);
    restored.forEach((val, i) => {
      expect(val).toBeCloseTo(original[i], 5);
    });
  });

  it('should convert regular JS number arrays correctly', () => {
    const jsArray = [1, 2, 3, 4, 5];
    const buffer = serializeVector(jsArray);
    const restored = deserializeVector(buffer);

    expect(restored).toHaveLength(5);
    restored.forEach((val, i) => {
      expect(val).toBeCloseTo(jsArray[i], 5);
    });
  });

  it('should handle empty vectors predictably', () => {
    const empty: number[] = [];
    const buffer = serializeVector(empty);
    expect(buffer.byteLength).toBe(0);

    const restored = deserializeVector(buffer);
    expect(restored).toEqual([]);
  });

  it('should handle very large vectors (1536 dimensions - OpenAI)', () => {
    const large = Array.from({ length: 1536 }, (_, i) => (i - 768) * 0.001);
    const buffer = serializeVector(large);

    expect(buffer.byteLength).toBe(1536 * 4);

    const restored = deserializeVector(buffer);
    expect(restored).toHaveLength(1536);
    expect(restored[0]).toBeCloseTo(large[0], 5);
    expect(restored[768]).toBeCloseTo(large[768], 5);
    expect(restored[1535]).toBeCloseTo(large[1535], 5);
  });

  it('should handle very large vectors (4096 dimensions)', () => {
    const veryLarge = Array.from({ length: 4096 }, (_, i) => Math.sin(i * 0.01));
    const buffer = serializeVector(veryLarge);

    expect(buffer.byteLength).toBe(4096 * 4);

    const restored = deserializeVector(buffer);
    expect(restored).toHaveLength(4096);
    expect(restored[0]).toBeCloseTo(veryLarge[0], 5);
    expect(restored[2048]).toBeCloseTo(veryLarge[2048], 5);
    expect(restored[4095]).toBeCloseTo(veryLarge[4095], 5);
  });

  it('should handle Float32 edge cases', () => {
    // Float32 max is ~3.4e38, min positive is ~1.2e-38
    const FLOAT32_MAX = 3.4028235e38;
    const FLOAT32_MIN_POSITIVE = 1.17549435e-38;
    const FLOAT32_SUBNORMAL = 1e-45; // smallest subnormal

    const edgeCases = [
      0,
      -0,
      FLOAT32_MAX,
      -FLOAT32_MAX,
      FLOAT32_MIN_POSITIVE,
      FLOAT32_SUBNORMAL,
    ];
    const buffer = serializeVector(edgeCases);
    const restored = deserializeVector(buffer);

    expect(restored).toHaveLength(6);
    expect(restored[0]).toBe(0);
    // -0 may or may not be preserved depending on Float32Array behavior
    expect(Object.is(restored[1], 0) || Object.is(restored[1], -0)).toBe(true);
    expect(restored[2]).toBeCloseTo(FLOAT32_MAX, -35); // very large, check magnitude
    expect(restored[3]).toBeCloseTo(-FLOAT32_MAX, -35);
    expect(restored[4]).toBeCloseTo(FLOAT32_MIN_POSITIVE, 43);
    // Subnormal may lose precision
    expect(restored[5]).toBeGreaterThanOrEqual(0);
    expect(restored[5]).toBeLessThan(FLOAT32_MIN_POSITIVE);
  });

  it('should handle values that overflow Float32', () => {
    // Values beyond Float32 range become Infinity
    const overflow = [1e39, -1e39];
    const buffer = serializeVector(overflow);
    const restored = deserializeVector(buffer);

    expect(restored[0]).toBe(Infinity);
    expect(restored[1]).toBe(-Infinity);
  });

  it('should handle precision loss in Float32', () => {
    // 0.1 cannot be exactly represented in Float32
    const imprecise = [0.1, 0.2, 0.3];
    const buffer = serializeVector(imprecise);
    const restored = deserializeVector(buffer);

    // Values are close but not exact due to Float32 precision
    restored.forEach((val, i) => {
      expect(val).toBeCloseTo(imprecise[i], 6); // ~7 significant digits in Float32
    });
  });

  it('should round to Float32 precision (~7 significant digits)', () => {
    // Float32 has ~7 decimal digits of precision
    // Values with more precision will be rounded
    const original = [0.123456789012345]; // 15 digits - more than Float32 can hold
    const restored = deserializeVector(serializeVector(original));

    // Should NOT be exactly equal due to Float32 precision loss
    expect(restored[0]).not.toBe(original[0]);
    // But should be close (~7 significant digits)
    expect(restored[0]).toBeCloseTo(original[0], 6);

    // Demonstrate the actual rounded value
    const float32Value = new Float32Array([original[0]])[0];
    expect(restored[0]).toBe(float32Value);
  });

  it('should use platform endianness (little-endian on most systems)', () => {
    // Float32Array uses platform endianness. This test documents the expected
    // byte layout for portability awareness. Most modern systems are little-endian.
    //
    // NOTE: If vectors are serialized on one machine and queried on another with
    // different endianness, results will be incorrect. This is acceptable for
    // in-memory SQLite but matters for file-based DBs shared across platforms.

    // 1.0 in IEEE 754 float32 is 0x3F800000
    // Little-endian: [0x00, 0x00, 0x80, 0x3F]
    // Big-endian:    [0x3F, 0x80, 0x00, 0x00]
    const buffer = serializeVector([1.0]);

    // Detect platform endianness
    const isLittleEndian = new Uint8Array(new Float32Array([1.0]).buffer)[0] === 0x00;

    if (isLittleEndian) {
      expect(buffer[0]).toBe(0x00);
      expect(buffer[1]).toBe(0x00);
      expect(buffer[2]).toBe(0x80);
      expect(buffer[3]).toBe(0x3f);
    } else {
      // Big-endian (rare but possible: some ARM, PowerPC)
      expect(buffer[0]).toBe(0x3f);
      expect(buffer[1]).toBe(0x80);
      expect(buffer[2]).toBe(0x00);
      expect(buffer[3]).toBe(0x00);
    }

    // Verify round-trip works regardless of endianness
    const restored = deserializeVector(buffer);
    expect(restored[0]).toBe(1.0);
  });

  it('should preserve values through SQLite storage', () => {
    const itemsVec = vec0Table('storage_test', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    const original = [0.123456, -0.789012, 0.0, 1.0];
    ctx.sqlite.prepare('INSERT INTO storage_test(embedding) VALUES (?)').run(serializeVector(original));

    const result = ctx.sqlite.prepare('SELECT vec_to_json(embedding) as vec FROM storage_test').get() as {
      vec: string;
    };
    const parsed = JSON.parse(result.vec);

    parsed.forEach((val: number, i: number) => {
      expect(val).toBeCloseTo(original[i], 5);
    });
  });
});
