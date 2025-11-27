import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import {
  serializeVector,
  deserializeVector,
  vec0Table,
  vecFloat,
  validateVector,
  typedVector,
  l2Distance,
  cosineSimilarity,
  cosineDistance,
  dotProduct,
} from '../src/index.js';
import { createTestContext, closeTestContext, type TestContext } from './setup.js';

describe('Edge Cases & Error Handling', () => {
  let ctx: TestContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  afterEach(() => {
    closeTestContext(ctx);
  });

  it('should error clearly when querying with wrong dimensions', () => {
    const itemsVec = vec0Table('dim_check', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    ctx.sqlite.prepare('INSERT INTO dim_check(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));

    expect(() => {
      ctx.sqlite
        .prepare('SELECT rowid, distance FROM dim_check WHERE embedding MATCH ? AND k = 1')
        .all(serializeVector([1, 0, 0]));
    }).toThrow(/dimension/i);
  });

  it('should handle null vector values in nullable columns gracefully', () => {
    ctx.sqlite.exec(`
      CREATE TABLE nullable_docs (
        id INTEGER PRIMARY KEY,
        content TEXT,
        embedding BLOB
      )
    `);

    ctx.sqlite.prepare('INSERT INTO nullable_docs (id, content, embedding) VALUES (?, ?, ?)').run(1, 'test', null);

    const result = ctx.sqlite.prepare('SELECT * FROM nullable_docs WHERE id = 1').get() as any;
    expect(result.embedding).toBeNull();
  });

  it('should fail gracefully when sqlite-vec extension is not loaded', () => {
    const plainSqlite = new Database(':memory:');

    expect(() => {
      plainSqlite.exec('CREATE VIRTUAL TABLE test USING vec0(embedding float[4])');
    }).toThrow(/no such module: vec0/i);

    plainSqlite.close();
  });

  it('should validate vectors with validateVector function', () => {
    expect(validateVector([1, 2, 3])).toBe(true);
    expect(validateVector([1, 2, 3], 3)).toBe(true);
    expect(validateVector([])).toBe(true);

    expect(() => validateVector('not an array')).toThrow('must be an array');
    expect(() => validateVector([1, 2, NaN])).toThrow('valid numbers');
    expect(() => validateVector([1, 2], 3)).toThrow('dimension mismatch');
    expect(() => validateVector([1, 'two', 3])).toThrow('valid numbers');
    expect(() => validateVector(null)).toThrow('must be an array');
    expect(() => validateVector(undefined)).toThrow('must be an array');
  });

  it('should handle Infinity and -Infinity in vectors', () => {
    const withInfinity = [1, Infinity, -Infinity, 0];
    const buffer = serializeVector(withInfinity);
    const restored = deserializeVector(buffer);

    expect(restored[1]).toBe(Infinity);
    expect(restored[2]).toBe(-Infinity);
  });

  it('should handle NaN in vectors (bypassing validation)', () => {
    // NaN round-trips through serialization
    const withNaN = [1, NaN, 0, 0];
    const buffer = serializeVector(withNaN);
    const restored = deserializeVector(buffer);

    expect(Number.isNaN(restored[1])).toBe(true);
  });

  it('should handle NaN in distance calculations and queries', () => {
    const itemsVec = vec0Table('nan_query_test', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    // Insert normal vectors
    ctx.sqlite.prepare('INSERT INTO nan_query_test(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
    ctx.sqlite.prepare('INSERT INTO nan_query_test(embedding) VALUES (?)').run(serializeVector([0, 1, 0, 0]));

    // Query with NaN returns null distances (not NaN)
    const nanQueryResults = ctx.sqlite
      .prepare('SELECT rowid, distance FROM nan_query_test WHERE embedding MATCH ? AND k = 2')
      .all(serializeVector([NaN, 0, 0, 0])) as Array<{ rowid: number; distance: number | null }>;

    expect(nanQueryResults).toHaveLength(2);
    expect(nanQueryResults[0].distance).toBeNull();
    expect(nanQueryResults[1].distance).toBeNull();

    // L2 distance with NaN returns null
    const l2Result = ctx.sqlite
      .prepare('SELECT vec_distance_L2(?, ?) as distance')
      .get(serializeVector([NaN, 0, 0, 0]), serializeVector([1, 0, 0, 0])) as { distance: number | null };
    expect(l2Result.distance).toBeNull();

    // Cosine distance with NaN returns null
    const cosineResult = ctx.sqlite
      .prepare('SELECT vec_distance_cosine(?, ?) as distance')
      .get(serializeVector([NaN, 0, 0, 0]), serializeVector([1, 0, 0, 0])) as { distance: number | null };
    expect(cosineResult.distance).toBeNull();

    // Storing NaN in vec0 is allowed but produces null distances
    ctx.sqlite.prepare('INSERT INTO nan_query_test(embedding) VALUES (?)').run(serializeVector([NaN, 0, 0, 0]));

    const resultsWithNaN = ctx.sqlite
      .prepare('SELECT rowid, distance FROM nan_query_test WHERE embedding MATCH ? AND k = 3')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number | null }>;

    expect(resultsWithNaN).toHaveLength(3);

    // Verify by rowid to avoid depending on sort order assumptions
    const nanVectorResult = resultsWithNaN.find((r) => r.rowid === 3);
    const exactMatchResult = resultsWithNaN.find((r) => r.rowid === 1);
    const otherResult = resultsWithNaN.find((r) => r.rowid === 2);

    expect(exactMatchResult?.distance).toBe(0);
    expect(Number.isFinite(otherResult?.distance)).toBe(true);
    expect(nanVectorResult?.distance).toBeNull();
  });

  it('should handle Infinity in distance calculations and queries', () => {
    const itemsVec = vec0Table('inf_query_test', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    // Insert normal vectors
    ctx.sqlite.prepare('INSERT INTO inf_query_test(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
    ctx.sqlite.prepare('INSERT INTO inf_query_test(embedding) VALUES (?)').run(serializeVector([0, 1, 0, 0]));

    // Query with Infinity returns Infinity distances
    const infQueryResults = ctx.sqlite
      .prepare('SELECT rowid, distance FROM inf_query_test WHERE embedding MATCH ? AND k = 2')
      .all(serializeVector([Infinity, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;

    expect(infQueryResults).toHaveLength(2);
    expect(infQueryResults[0].distance).toBe(Infinity);
    expect(infQueryResults[1].distance).toBe(Infinity);

    // L2 distance with Infinity returns Infinity
    const l2Result = ctx.sqlite
      .prepare('SELECT vec_distance_L2(?, ?) as distance')
      .get(serializeVector([Infinity, 0, 0, 0]), serializeVector([1, 0, 0, 0])) as { distance: number };
    expect(l2Result.distance).toBe(Infinity);

    // Cosine distance with Infinity returns null (normalization fails)
    const cosineResult = ctx.sqlite
      .prepare('SELECT vec_distance_cosine(?, ?) as distance')
      .get(serializeVector([Infinity, 0, 0, 0]), serializeVector([1, 0, 0, 0])) as { distance: number | null };
    expect(cosineResult.distance).toBeNull();

    // Storing Infinity in vec0 is allowed but produces Infinity distances
    ctx.sqlite.prepare('INSERT INTO inf_query_test(embedding) VALUES (?)').run(serializeVector([Infinity, 0, 0, 0]));

    const resultsWithInf = ctx.sqlite
      .prepare('SELECT rowid, distance FROM inf_query_test WHERE embedding MATCH ? AND k = 3')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;

    expect(resultsWithInf).toHaveLength(3);

    // Verify by rowid to avoid depending on sort order assumptions
    const infVectorResult = resultsWithInf.find((r) => r.rowid === 3);
    const exactMatchResult = resultsWithInf.find((r) => r.rowid === 1);
    const otherResult = resultsWithInf.find((r) => r.rowid === 2);

    expect(exactMatchResult?.distance).toBe(0);
    expect(Number.isFinite(otherResult?.distance)).toBe(true);
    expect(infVectorResult?.distance).toBe(Infinity);
  });

  it('should handle typedVector dimension mismatch', () => {
    expect(() => typedVector(4, [1, 2, 3])).toThrow('Vector dimension mismatch: expected 4, got 3');
    expect(() => typedVector(2, [1, 2, 3, 4, 5])).toThrow('Vector dimension mismatch: expected 2, got 5');
  });

  it('should handle math function dimension mismatches', () => {
    expect(() => l2Distance([1, 2], [1, 2, 3])).toThrow('same dimensions');
    expect(() => cosineSimilarity([1], [1, 2])).toThrow('same dimensions');
    expect(() => cosineDistance([1, 2, 3], [1])).toThrow('same dimensions');
    expect(() => dotProduct([1, 2], [1, 2, 3])).toThrow('same dimensions');
  });

  describe('Invalid dimension validation at schema time', () => {
    it('should reject zero dimensions when creating vec0 table', () => {
      const invalidTable = vec0Table('zero_dim', {
        embedding: vecFloat('embedding', 0),
      });

      // sqlite-vec should reject this at table creation time
      expect(() => {
        ctx.sqlite.exec(invalidTable.createSQL());
      }).toThrow();
    });

    it('should reject negative dimensions when creating vec0 table', () => {
      const invalidTable = vec0Table('negative_dim', {
        embedding: vecFloat('embedding', -4),
      });

      // sqlite-vec should reject this at table creation time
      expect(() => {
        ctx.sqlite.exec(invalidTable.createSQL());
      }).toThrow();
    });

    it('should handle zero/negative dimensions in schema builder', () => {
      // With dimension 0, the code treats it as falsy and omits the type specification
      const zeroDimTable = vec0Table('zero_test', {
        embedding: vecFloat('embedding', 0),
      });
      // 0 is falsy, so "float[0]" is NOT generated - just bare "embedding"
      expect(zeroDimTable.createSQL()).not.toContain('float[0]');

      // Negative dimensions are passed through and will fail at DB level
      const negativeDimTable = vec0Table('neg_test', {
        embedding: vecFloat('embedding', -4),
      });
      expect(negativeDimTable.createSQL()).toContain('float[-4]');
    });
  });

  describe('Index/table on non-existent column references', () => {
    it('should fail when inserting into non-existent column', () => {
      const itemsVec = vec0Table('col_test', {
        embedding: vecFloat('embedding', 4),
      });
      ctx.sqlite.exec(itemsVec.createSQL());

      // Try to insert into a column that doesn't exist
      expect(() => {
        ctx.sqlite.prepare('INSERT INTO col_test(nonexistent_col) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
      }).toThrow();
    });

    it('should fail when querying non-existent vector column', () => {
      const itemsVec = vec0Table('query_col_test', {
        embedding: vecFloat('embedding', 4),
      });
      ctx.sqlite.exec(itemsVec.createSQL());

      expect(() => {
        ctx.sqlite.prepare('SELECT rowid FROM query_col_test WHERE wrong_column MATCH ? AND k = 1').all(serializeVector([1, 0, 0, 0]));
      }).toThrow();
    });
  });
});
