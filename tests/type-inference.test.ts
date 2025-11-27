import { describe, it, expect, beforeEach, afterEach, expectTypeOf } from 'vitest';
import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';
import type { SQL } from 'drizzle-orm';
import {
  vector,
  vec_distance_L2,
  typedVector,
  serializeVector,
  type Vector,
} from '../src/index.js';
import { createTestContext, closeTestContext, type TestContext } from './setup.js';

describe('Type Inference', () => {
  let ctx: TestContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  afterEach(() => {
    closeTestContext(ctx);
  });

  it('should infer correct return type for vector column', () => {
    const documents = sqliteTable('documents', {
      id: integer('id').primaryKey(),
      embedding: vector('embedding', { dimensions: 4 }),
    });

    type SelectResult = typeof documents.$inferSelect;
    expectTypeOf<SelectResult['embedding']>().toEqualTypeOf<number[]>();
  });

  it('should infer number type for distance function results', () => {
    const documents = sqliteTable('documents', {
      id: integer('id').primaryKey(),
      embedding: vector('embedding', { dimensions: 4 }),
    });

    const queryVec = [0.1, 0.2, 0.3, 0.4];
    const distanceExpr = vec_distance_L2(documents.embedding, queryVec);

    // vec_distance_L2 returns SQL<number>, meaning the result type is number
    expectTypeOf(distanceExpr).toMatchTypeOf<SQL<number>>();
  });

  it('should require vector field in insert type', () => {
    const documents = sqliteTable('documents', {
      id: integer('id').primaryKey(),
      embedding: vector('embedding', { dimensions: 4 }),
    });

    type InsertType = typeof documents.$inferInsert;
    expectTypeOf<InsertType['embedding']>().toEqualTypeOf<number[]>();
  });

  it('should have correct Vector type with dimensions', () => {
    const vec: Vector<4> = typedVector(4, [1, 2, 3, 4]);
    expectTypeOf(vec).toMatchTypeOf<number[]>();

    // Compile-time dimension safety: Vector<4> should not be assignable to Vector<3>
    // @ts-expect-error - dimension mismatch
    const _wrongDim: Vector<3> = typedVector(4, [1, 2, 3, 4]);
  });

  describe('Partial selects excluding vectors', () => {
    it('should exclude vector column from partial select result type', () => {
      const documents = sqliteTable('partial_docs', {
        id: integer('id').primaryKey(),
        title: text('title'),
        content: text('content'),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      // Test actual Drizzle type inference for partial select
      const query = ctx.db
        .select({ id: documents.id, title: documents.title })
        .from(documents);

      type Result = Awaited<ReturnType<typeof query.all>>[number];

      // Verify the inferred type has only id and title
      expectTypeOf<Result>().toEqualTypeOf<{ id: number; title: string | null }>();
      expectTypeOf<Result>().not.toHaveProperty('embedding');
      expectTypeOf<Result>().not.toHaveProperty('content');
    });

    it('should return only selected columns at runtime', () => {
      const documents = sqliteTable('runtime_partial_docs', {
        id: integer('id').primaryKey(),
        title: text('title'),
        content: text('content'),
      });

      ctx.sqlite.exec(`
        CREATE TABLE runtime_partial_docs (
          id INTEGER PRIMARY KEY,
          title TEXT,
          content TEXT,
          embedding BLOB
        )
      `);

      ctx.sqlite.prepare('INSERT INTO runtime_partial_docs (id, title, content, embedding) VALUES (?, ?, ?, ?)')
        .run(1, 'Test', 'Content', serializeVector([1, 2, 3, 4]));

      // Select only id and title
      const results = ctx.db
        .select({ id: documents.id, title: documents.title })
        .from(documents)
        .all();

      expect(results).toHaveLength(1);
      expect(results[0]).toHaveProperty('id', 1);
      expect(results[0]).toHaveProperty('title', 'Test');
      expect(results[0]).not.toHaveProperty('embedding');
      expect(results[0]).not.toHaveProperty('content');
    });

    it('should correctly type partial select with distance calculation', () => {
      const documents = sqliteTable('distance_partial_docs', {
        id: integer('id').primaryKey(),
        title: text('title'),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const queryVec = [1, 0, 0, 0];

      // Partial select with computed distance column
      const query = ctx.db
        .select({
          id: documents.id,
          title: documents.title,
          distance: vec_distance_L2(documents.embedding, queryVec),
        })
        .from(documents);

      // Type inference should give us id, title, and distance (not embedding)
      type QueryResult = Awaited<ReturnType<typeof query.all>>[number];

      // Verify property types, not just existence
      expectTypeOf<QueryResult['id']>().toEqualTypeOf<number>();
      expectTypeOf<QueryResult['title']>().toEqualTypeOf<string | null>();
      expectTypeOf<QueryResult['distance']>().toEqualTypeOf<number>();
      expectTypeOf<QueryResult>().not.toHaveProperty('embedding');
    });
  });
});
