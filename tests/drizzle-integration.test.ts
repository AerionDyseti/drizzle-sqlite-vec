import { describe, it, expect, beforeEach, afterEach, expectTypeOf } from 'vitest';
import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';
import {
  vector,
  serializeVector,
  vec0Table,
  vecFloat,
} from '../src/index.js';
import { createTestContext, closeTestContext, type TestContext } from './setup.js';

describe('Integration with Drizzle Patterns', () => {
  let ctx: TestContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  afterEach(() => {
    closeTestContext(ctx);
  });

  it('should work with $inferSelect and $inferInsert', () => {
    const documents = sqliteTable('documents', {
      id: integer('id').primaryKey(),
      content: text('content'),
      embedding: vector('embedding', { dimensions: 4 }),
    });

    type SelectType = typeof documents.$inferSelect;
    type InsertType = typeof documents.$inferInsert;

    // Verify actual types, not just property existence
    expectTypeOf<SelectType['id']>().toEqualTypeOf<number>();
    expectTypeOf<SelectType['content']>().toEqualTypeOf<string | null>();
    expectTypeOf<SelectType['embedding']>().toEqualTypeOf<number[]>();

    // Insert type should also have correct types
    expectTypeOf<InsertType['id']>().toEqualTypeOf<number | undefined>();
    expectTypeOf<InsertType['content']>().toEqualTypeOf<string | null | undefined>();
    expectTypeOf<InsertType['embedding']>().toEqualTypeOf<number[]>();
  });

  it('should work with transactions (commit)', () => {
    const itemsVec = vec0Table('tx_vec', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    const insertMany = ctx.sqlite.transaction((vectors: number[][]) => {
      const stmt = ctx.sqlite.prepare('INSERT INTO tx_vec(embedding) VALUES (?)');
      for (const vec of vectors) {
        stmt.run(serializeVector(vec));
      }
    });

    const vectors = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
    ];

    insertMany(vectors);

    const count = ctx.sqlite.prepare('SELECT COUNT(*) as count FROM tx_vec').get() as { count: number };
    expect(count.count).toBe(3);
  });

  it('should rollback vector inserts on transaction failure', () => {
    const itemsVec = vec0Table('tx_rollback_vec', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    // Insert one vector outside the transaction
    ctx.sqlite.prepare('INSERT INTO tx_rollback_vec(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));

    const countBefore = ctx.sqlite.prepare('SELECT COUNT(*) as count FROM tx_rollback_vec').get() as { count: number };
    expect(countBefore.count).toBe(1);

    // Transaction that will fail partway through
    const insertWithFailure = ctx.sqlite.transaction((vectors: number[][], failAtIndex: number) => {
      const stmt = ctx.sqlite.prepare('INSERT INTO tx_rollback_vec(embedding) VALUES (?)');
      for (let i = 0; i < vectors.length; i++) {
        if (i === failAtIndex) {
          throw new Error('Simulated failure');
        }
        stmt.run(serializeVector(vectors[i]));
      }
    });

    const vectors = [
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];

    // This should throw and rollback all inserts within the transaction
    expect(() => insertWithFailure(vectors, 2)).toThrow('Simulated failure');

    // Count should still be 1 (only the pre-transaction insert)
    const countAfter = ctx.sqlite.prepare('SELECT COUNT(*) as count FROM tx_rollback_vec').get() as { count: number };
    expect(countAfter.count).toBe(1);

    // Verify the original vector is still there
    const results = ctx.sqlite
      .prepare('SELECT rowid, distance FROM tx_rollback_vec WHERE embedding MATCH ? AND k = 10')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;

    expect(results).toHaveLength(1);
    expect(results[0].distance).toBeCloseTo(0, 5);
  });

  it('should work with prepared statements', () => {
    const itemsVec = vec0Table('prepared_vec', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    const insertStmt = ctx.sqlite.prepare('INSERT INTO prepared_vec(embedding) VALUES (?)');
    insertStmt.run(serializeVector([1, 0, 0, 0]));
    insertStmt.run(serializeVector([0, 1, 0, 0]));

    const searchStmt = ctx.sqlite.prepare('SELECT rowid, distance FROM prepared_vec WHERE embedding MATCH ? AND k = ?');
    const results = searchStmt.all(serializeVector([1, 0, 0, 0]), 5) as Array<{
      rowid: number;
      distance: number;
    }>;

    expect(results).toHaveLength(2);
    expect(results[0].distance).toBeCloseTo(0, 5);
  });

  it('should work with subqueries involving vector operations', () => {
    // Create a mapping table to track the relationship between items and their vectors
    // This avoids relying on implicit rowid assignment order in vec0
    ctx.sqlite.exec(`
      CREATE TABLE items (
        id INTEGER PRIMARY KEY,
        name TEXT
      )
    `);
    ctx.sqlite.exec(`
      CREATE TABLE item_vectors (
        item_id INTEGER PRIMARY KEY,
        vec_rowid INTEGER
      )
    `);

    const itemsVec = vec0Table('items_vec', {
      embedding: vecFloat('embedding', 4),
    });
    ctx.sqlite.exec(itemsVec.createSQL());

    // Insert items
    ctx.sqlite.prepare('INSERT INTO items (id, name) VALUES (?, ?)').run(1, 'Item A');
    ctx.sqlite.prepare('INSERT INTO items (id, name) VALUES (?, ?)').run(2, 'Item B');

    // Insert vectors and explicitly track the rowid mapping
    // vec0 auto-assigns rowids; we capture them via last_insert_rowid()
    const insertVec = ctx.sqlite.prepare('INSERT INTO items_vec (embedding) VALUES (?)');
    const insertMapping = ctx.sqlite.prepare('INSERT INTO item_vectors (item_id, vec_rowid) VALUES (?, ?)');

    insertVec.run(serializeVector([1, 0, 0, 0]));
    const rowid1 = ctx.sqlite.prepare('SELECT last_insert_rowid() as rowid').get() as { rowid: number };
    insertMapping.run(1, rowid1.rowid); // Item A -> first vector

    insertVec.run(serializeVector([0, 1, 0, 0]));
    const rowid2 = ctx.sqlite.prepare('SELECT last_insert_rowid() as rowid').get() as { rowid: number };
    insertMapping.run(2, rowid2.rowid); // Item B -> second vector

    // Query using the explicit mapping table instead of assuming rowid == item.id
    const results = ctx.sqlite
      .prepare(
        `
      SELECT i.*, sub.distance
      FROM items i
      INNER JOIN item_vectors iv ON i.id = iv.item_id
      INNER JOIN (
        SELECT rowid, distance
        FROM items_vec
        WHERE embedding MATCH ?
        AND k = 10
      ) sub ON iv.vec_rowid = sub.rowid
      ORDER BY sub.distance
    `
      )
      .all(serializeVector([1, 0, 0, 0])) as Array<{ id: number; name: string; distance: number }>;

    expect(results).toHaveLength(2);
    expect(results[0].name).toBe('Item A'); // Closest to query vector [1,0,0,0]
    expect(results[0].distance).toBeCloseTo(0, 5);
    expect(results[1].name).toBe('Item B');
  });

  it('should work with Drizzle select API on regular tables with vector columns', () => {
    const documents = sqliteTable('drizzle_docs', {
      id: integer('id').primaryKey(),
      content: text('content'),
      embedding: vector('embedding', { dimensions: 4 }),
    });

    ctx.sqlite.exec(`
      CREATE TABLE drizzle_docs (
        id INTEGER PRIMARY KEY,
        content TEXT,
        embedding BLOB
      )
    `);

    const originalEmbedding = [1, 2, 3, 4];
    ctx.sqlite
      .prepare('INSERT INTO drizzle_docs (id, content, embedding) VALUES (?, ?, ?)')
      .run(1, 'test', serializeVector(originalEmbedding));

    const results = ctx.db.select().from(documents).all();
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe(1);
    expect(results[0].content).toBe('test');

    // Verify the embedding round-trips correctly through the vector type
    expect(results[0].embedding).toHaveLength(4);
    results[0].embedding.forEach((val, i) => {
      expect(val).toBeCloseTo(originalEmbedding[i], 5);
    });
  });

  describe('Index Operations', () => {
    it('should create vec0 table which acts as an index', () => {
      const itemsVec = vec0Table('items_idx', {
        embedding: vecFloat('embedding', 4),
      });

      const createSql = itemsVec.createSQL();
      expect(createSql).toContain('USING vec0');

      ctx.sqlite.exec(createSql);
    });

    it('should support cosine distance metric in table definition', () => {
      const itemsVec = vec0Table('cosine_idx', {
        embedding: vecFloat('embedding', 4).distanceMetric('cosine'),
      });

      const createSql = itemsVec.createSQL();
      expect(createSql).toContain('distance_metric=cosine');

      ctx.sqlite.exec(createSql);

      ctx.sqlite.prepare('INSERT INTO cosine_idx(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
      ctx.sqlite.prepare('INSERT INTO cosine_idx(embedding) VALUES (?)').run(serializeVector([0, 1, 0, 0]));

      const results = ctx.sqlite
        .prepare('SELECT rowid, distance FROM cosine_idx WHERE embedding MATCH ? AND k = 2')
        .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;

      expect(results).toHaveLength(2);
      expect(results[0].distance).toBeCloseTo(0, 5);
    });
  });
});
