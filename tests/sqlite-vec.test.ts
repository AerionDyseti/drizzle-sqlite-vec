import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  serializeVector,
  vec0Table,
  vecFloat,
} from '../src/index.js';
import { createTestContext, closeTestContext, type TestContext } from './setup.js';

describe('sqlite-vec Integration', () => {
  let ctx: TestContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  afterEach(() => {
    closeTestContext(ctx);
  });

  it('should perform end-to-end insert and KNN search correctly', () => {
    const itemsVec = vec0Table('e2e_vec', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    const vectors = [
      [1.0, 0.0, 0.0, 0.0],
      [0.9, 0.1, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ];

    vectors.forEach((vec) => {
      ctx.sqlite.prepare('INSERT INTO e2e_vec(embedding) VALUES (?)').run(serializeVector(vec));
    });

    const results = ctx.sqlite
      .prepare('SELECT rowid, distance FROM e2e_vec WHERE embedding MATCH ? AND k = 3')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;

    expect(results).toHaveLength(3);
    expect(results[0].rowid).toBe(1);
    expect(results[0].distance).toBeCloseTo(0, 5);
    expect(results[1].rowid).toBe(2);
    expect([3, 4, 5]).toContain(results[2].rowid);
  });

  it('should return results ordered by distance (closest first)', () => {
    const itemsVec = vec0Table('order_vec', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    ctx.sqlite.prepare('INSERT INTO order_vec(embedding) VALUES (?)').run(serializeVector([0.5, 0.5, 0.5, 0.5]));
    ctx.sqlite.prepare('INSERT INTO order_vec(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
    ctx.sqlite.prepare('INSERT INTO order_vec(embedding) VALUES (?)').run(serializeVector([0, 0, 0, 1]));

    const results = ctx.sqlite
      .prepare('SELECT rowid, distance FROM order_vec WHERE embedding MATCH ? AND k = 10')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;

    expect(results).toHaveLength(3);

    for (let i = 1; i < results.length; i++) {
      expect(results[i].distance).toBeGreaterThanOrEqual(results[i - 1].distance);
    }

    expect(results[0].distance).toBeCloseTo(0, 5);
  });

  it('should support KNN queries on multiple vector columns independently', () => {
    const multiVec = vec0Table('multi_knn_vec', {
      embedding1: vecFloat('embedding1', 4),
      embedding2: vecFloat('embedding2', 4),
    });
    ctx.sqlite.exec(multiVec.createSQL());

    // Insert vectors where embedding1 and embedding2 have different "closest" patterns
    // Row 1: embedding1 close to [1,0,0,0], embedding2 close to [0,0,0,1]
    ctx.sqlite.prepare('INSERT INTO multi_knn_vec(embedding1, embedding2) VALUES (?, ?)')
      .run(serializeVector([1, 0, 0, 0]), serializeVector([0, 0, 0, 1]));
    // Row 2: embedding1 close to [0,1,0,0], embedding2 close to [1,0,0,0]
    ctx.sqlite.prepare('INSERT INTO multi_knn_vec(embedding1, embedding2) VALUES (?, ?)')
      .run(serializeVector([0, 1, 0, 0]), serializeVector([1, 0, 0, 0]));

    // Query embedding1 for [1,0,0,0] - row 1 should be closest
    const results1 = ctx.sqlite
      .prepare('SELECT rowid, distance FROM multi_knn_vec WHERE embedding1 MATCH ? AND k = 2')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;

    expect(results1[0].rowid).toBe(1);
    expect(results1[0].distance).toBeCloseTo(0, 5);

    // Query embedding2 for [1,0,0,0] - row 2 should be closest
    const results2 = ctx.sqlite
      .prepare('SELECT rowid, distance FROM multi_knn_vec WHERE embedding2 MATCH ? AND k = 2')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;

    expect(results2[0].rowid).toBe(2);
    expect(results2[0].distance).toBeCloseTo(0, 5);
  });

  it('should return mathematically plausible distance values', () => {
    const itemsVec = vec0Table('math_vec', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    ctx.sqlite.prepare('INSERT INTO math_vec(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
    ctx.sqlite.prepare('INSERT INTO math_vec(embedding) VALUES (?)').run(serializeVector([0, 1, 0, 0]));

    const l2Result = ctx.sqlite
      .prepare('SELECT vec_distance_L2(?, ?) as distance')
      .get(serializeVector([1, 0, 0, 0]), serializeVector([0, 1, 0, 0])) as { distance: number };

    expect(l2Result.distance).toBeCloseTo(Math.sqrt(2), 4);

    const cosineResult = ctx.sqlite
      .prepare('SELECT vec_distance_cosine(?, ?) as distance')
      .get(serializeVector([1, 0, 0, 0]), serializeVector([0, 1, 0, 0])) as { distance: number };

    expect(cosineResult.distance).toBeCloseTo(1, 4);

    const sameCosine = ctx.sqlite
      .prepare('SELECT vec_distance_cosine(?, ?) as distance')
      .get(serializeVector([1, 0, 0, 0]), serializeVector([1, 0, 0, 0])) as { distance: number };

    expect(sameCosine.distance).toBeCloseTo(0, 5);
  });

  it('should handle vec_length correctly', () => {
    const vec = [3, 4, 0, 0];
    const result = ctx.sqlite.prepare('SELECT vec_length(?) as len').get(serializeVector(vec)) as { len: number };

    expect(result.len).toBe(4);
  });

  it('should calculate vector magnitude using distance from zero', () => {
    const vec = [3, 4, 0, 0];
    const zeroVec = [0, 0, 0, 0];
    const result = ctx.sqlite
      .prepare('SELECT vec_distance_L2(?, ?) as magnitude')
      .get(serializeVector(vec), serializeVector(zeroVec)) as { magnitude: number };

    expect(result.magnitude).toBeCloseTo(5, 5);
  });

  it('should handle vec_normalize correctly', () => {
    const vec = [3, 4, 0, 0];
    const result = ctx.sqlite.prepare('SELECT vec_to_json(vec_normalize(?)) as normalized').get(serializeVector(vec)) as {
      normalized: string;
    };

    const normalized = JSON.parse(result.normalized);
    expect(normalized[0]).toBeCloseTo(0.6, 5);
    expect(normalized[1]).toBeCloseTo(0.8, 5);
    expect(normalized[2]).toBeCloseTo(0, 5);
    expect(normalized[3]).toBeCloseTo(0, 5);
  });

  it('should handle vec_add correctly', () => {
    const vec1 = [1, 2, 3, 4];
    const vec2 = [4, 3, 2, 1];
    const result = ctx.sqlite.prepare('SELECT vec_to_json(vec_add(?, ?)) as sum').get(serializeVector(vec1), serializeVector(vec2)) as {
      sum: string;
    };

    const sum = JSON.parse(result.sum);
    expect(sum).toEqual([5, 5, 5, 5]);
  });

  it('should handle vec_sub correctly', () => {
    const vec1 = [5, 5, 5, 5];
    const vec2 = [1, 2, 3, 4];
    const result = ctx.sqlite.prepare('SELECT vec_to_json(vec_sub(?, ?)) as diff').get(serializeVector(vec1), serializeVector(vec2)) as {
      diff: string;
    };

    const diff = JSON.parse(result.diff);
    expect(diff).toEqual([4, 3, 2, 1]);
  });

  it('should handle vec_slice correctly', () => {
    const vec = [1, 2, 3, 4, 5, 6];
    const result = ctx.sqlite.prepare('SELECT vec_to_json(vec_slice(?, 1, 4)) as slice').get(serializeVector(vec)) as {
      slice: string;
    };

    const slice = JSON.parse(result.slice);
    expect(slice).toEqual([2, 3, 4]);
  });

  describe('Distance threshold filtering', () => {
    it('should filter results by distance threshold using subquery', () => {
      const itemsVec = vec0Table('threshold_vec', {
        embedding: vecFloat('embedding', 4),
      });
      ctx.sqlite.exec(itemsVec.createSQL());

      // Insert vectors at various distances from [1,0,0,0]
      ctx.sqlite.prepare('INSERT INTO threshold_vec(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));       // distance = 0
      ctx.sqlite.prepare('INSERT INTO threshold_vec(embedding) VALUES (?)').run(serializeVector([0.9, 0.1, 0, 0]));   // distance ≈ 0.14
      ctx.sqlite.prepare('INSERT INTO threshold_vec(embedding) VALUES (?)').run(serializeVector([0.7, 0.3, 0, 0]));   // distance ≈ 0.42
      ctx.sqlite.prepare('INSERT INTO threshold_vec(embedding) VALUES (?)').run(serializeVector([0, 1, 0, 0]));       // distance = sqrt(2) ≈ 1.41
      ctx.sqlite.prepare('INSERT INTO threshold_vec(embedding) VALUES (?)').run(serializeVector([0, 0, 1, 0]));       // distance = sqrt(2) ≈ 1.41

      const queryVec = serializeVector([1, 0, 0, 0]);

      // Get all results first, then filter by distance < 0.5
      const results = ctx.sqlite.prepare(`
        SELECT rowid, distance
        FROM threshold_vec
        WHERE embedding MATCH ?
          AND k = 10
      `).all(queryVec) as Array<{ rowid: number; distance: number }>;

      // Filter in application code (sqlite-vec KNN doesn't support distance filtering directly in WHERE)
      const filtered = results.filter(r => r.distance < 0.5);

      expect(filtered.length).toBe(3); // rows 1, 2, 3 should be within threshold
      expect(filtered.every(r => r.distance < 0.5)).toBe(true);
    });

    it('should filter results by distance threshold using CTE', () => {
      const itemsVec = vec0Table('threshold_cte_vec', {
        embedding: vecFloat('embedding', 4),
      });
      ctx.sqlite.exec(itemsVec.createSQL());

      ctx.sqlite.prepare('INSERT INTO threshold_cte_vec(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
      ctx.sqlite.prepare('INSERT INTO threshold_cte_vec(embedding) VALUES (?)').run(serializeVector([0.9, 0.1, 0, 0]));
      ctx.sqlite.prepare('INSERT INTO threshold_cte_vec(embedding) VALUES (?)').run(serializeVector([0, 1, 0, 0]));

      const queryVec = serializeVector([1, 0, 0, 0]);

      // Use CTE to first get KNN results, then filter by distance
      const results = ctx.sqlite.prepare(`
        WITH knn_results AS (
          SELECT rowid, distance
          FROM threshold_cte_vec
          WHERE embedding MATCH ?
            AND k = 10
        )
        SELECT * FROM knn_results WHERE distance < 0.5
      `).all(queryVec) as Array<{ rowid: number; distance: number }>;

      // Should only include the exact match and very close vector
      expect(results.length).toBe(2);
      expect(results[0].distance).toBeCloseTo(0, 5);
    });

    it('should support range-based filtering with regular tables using vec_distance', () => {
      // Create a regular table with vectors
      ctx.sqlite.exec(`
        CREATE TABLE range_docs (
          id INTEGER PRIMARY KEY,
          embedding BLOB
        )
      `);

      ctx.sqlite.prepare('INSERT INTO range_docs (id, embedding) VALUES (?, ?)').run(1, serializeVector([1, 0, 0, 0]));
      ctx.sqlite.prepare('INSERT INTO range_docs (id, embedding) VALUES (?, ?)').run(2, serializeVector([0.9, 0.1, 0, 0]));
      ctx.sqlite.prepare('INSERT INTO range_docs (id, embedding) VALUES (?, ?)').run(3, serializeVector([0, 1, 0, 0]));
      ctx.sqlite.prepare('INSERT INTO range_docs (id, embedding) VALUES (?, ?)').run(4, serializeVector([-1, 0, 0, 0]));

      const queryVec = serializeVector([1, 0, 0, 0]);

      // For regular tables, use vec_distance_L2 with WHERE clause
      const results = ctx.sqlite.prepare(`
        SELECT id, vec_distance_L2(embedding, ?) as distance
        FROM range_docs
        WHERE vec_distance_L2(embedding, ?) < 1.0
        ORDER BY distance
      `).all(queryVec, queryVec) as Array<{ id: number; distance: number }>;

      expect(results.length).toBe(2); // Only id 1 and 2 should be within distance 1.0
      expect(results[0].id).toBe(1);
      expect(results[1].id).toBe(2);
    });
  });
});
