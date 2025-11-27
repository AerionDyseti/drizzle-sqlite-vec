import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { sqliteTable, integer } from 'drizzle-orm/sqlite-core';
import {
  vector,
  serializeVector,
  vec0Table,
  vecFloat,
  vec_distance_L2,
  vec_distance_cosine,
  vec_length,
  vec_normalize,
  vec_add,
  vec_sub,
  vec_slice,
  vec_f32,
  vec_to_json,
  vec_quantize_i8,
  vec_quantize_binary,
  knnWhere,
} from '../src/index.js';
import { createTestContext, closeTestContext, type TestContext } from './setup.js';

describe('SQL Generation', () => {
  let ctx: TestContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  afterEach(() => {
    closeTestContext(ctx);
  });

  describe('Distance functions generate correct SQL', () => {
    it('should generate vec_distance_L2 SQL with parameterized vector', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const queryVec = [0.1, 0.2, 0.3, 0.4];
      const distanceExpr = vec_distance_L2(documents.embedding, queryVec);

      const query = ctx.db
        .select({ distance: distanceExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_distance_L2');
      // Verify the vector is parameterized (column, ?)
      expect(query.sql).toMatch(/vec_distance_L2\([^,]+,\s*\?\)/);
      // Verify params array contains the serialized vector
      expect(query.params).toBeDefined();
      expect(query.params.length).toBeGreaterThan(0);
    });

    it('should generate vec_distance_cosine SQL with parameterized vector', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const queryVec = [0.1, 0.2, 0.3, 0.4];
      const distanceExpr = vec_distance_cosine(documents.embedding, queryVec);

      const query = ctx.db
        .select({ distance: distanceExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_distance_cosine');
      // Verify the vector is parameterized (column, ?)
      expect(query.sql).toMatch(/vec_distance_cosine\([^,]+,\s*\?\)/);
      // Verify params array contains the serialized vector
      expect(query.params).toBeDefined();
      expect(query.params.length).toBeGreaterThan(0);
    });
  });

  describe('Vector operation functions generate correct SQL', () => {
    it('should generate vec_length SQL', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const lengthExpr = vec_length(documents.embedding);
      const query = ctx.db
        .select({ length: lengthExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_length');
    });

    it('should generate vec_normalize SQL', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const normalizedExpr = vec_normalize(documents.embedding);
      const query = ctx.db
        .select({ normalized: normalizedExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_normalize');
    });

    it('should generate vec_add SQL with literal + literal', () => {
      const vec1 = [1, 0, 0, 0];
      const vec2 = [0, 1, 0, 0];
      const addExpr = vec_add(vec1, vec2);

      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
      });

      const query = ctx.db
        .select({ sum: addExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_add');
      expect(query.params.length).toBe(2); // Both vectors parameterized
    });

    it('should generate vec_add SQL with column + literal', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const literal = [1, 0, 0, 0];
      const addExpr = vec_add(documents.embedding, literal);

      const query = ctx.db
        .select({ sum: addExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_add');
      expect(query.sql).toContain('"embedding"'); // Column reference
      expect(query.params.length).toBe(1); // Only literal parameterized
    });

    it('should generate vec_sub SQL with literal + literal', () => {
      const vec1 = [1, 0, 0, 0];
      const vec2 = [0, 1, 0, 0];
      const subExpr = vec_sub(vec1, vec2);

      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
      });

      const query = ctx.db
        .select({ diff: subExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_sub');
      expect(query.params.length).toBe(2); // Both vectors parameterized
    });

    it('should generate vec_sub SQL with column + literal', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const literal = [1, 0, 0, 0];
      const subExpr = vec_sub(documents.embedding, literal);

      const query = ctx.db
        .select({ diff: subExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_sub');
      expect(query.sql).toContain('"embedding"'); // Column reference
      expect(query.params.length).toBe(1); // Only literal parameterized
    });

    it('should generate vec_add SQL with column + column', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding1: vector('embedding1', { dimensions: 4 }),
        embedding2: vector('embedding2', { dimensions: 4 }),
      });

      const addExpr = vec_add(documents.embedding1, documents.embedding2);

      const query = ctx.db
        .select({ sum: addExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_add');
      expect(query.sql).toContain('"embedding1"');
      expect(query.sql).toContain('"embedding2"');
      expect(query.params.length).toBe(0); // No literals, both are column refs
    });

    it('should generate vec_sub SQL with column + column', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding1: vector('embedding1', { dimensions: 4 }),
        embedding2: vector('embedding2', { dimensions: 4 }),
      });

      const subExpr = vec_sub(documents.embedding1, documents.embedding2);

      const query = ctx.db
        .select({ diff: subExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_sub');
      expect(query.sql).toContain('"embedding1"');
      expect(query.sql).toContain('"embedding2"');
      expect(query.params.length).toBe(0); // No literals, both are column refs
    });

    it('should generate vec_slice SQL', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const sliceExpr = vec_slice(documents.embedding, 0, 2);
      const query = ctx.db
        .select({ slice: sliceExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_slice');
    });

    it('should generate vec_f32 SQL', () => {
      const f32Expr = vec_f32('[1.0, 2.0, 3.0]');

      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
      });

      const query = ctx.db
        .select({ vec: f32Expr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_f32');
    });

    it('should generate vec_to_json SQL', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const jsonExpr = vec_to_json(documents.embedding);
      const query = ctx.db
        .select({ json: jsonExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_to_json');
    });

    it('should generate vec_quantize_i8 SQL', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const quantizedExpr = vec_quantize_i8(documents.embedding);
      const query = ctx.db
        .select({ quantized: quantizedExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_quantize_i8');
    });

    it('should generate vec_quantize_binary SQL', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const binaryExpr = vec_quantize_binary(documents.embedding);
      const query = ctx.db
        .select({ binary: binaryExpr })
        .from(documents)
        .toSQL();

      expect(query.sql).toContain('vec_quantize_binary');
    });
  });

  describe('KNN query helpers', () => {
    it('should build vector search query that executes correctly', () => {
      const itemsVec = vec0Table('search_items_vec', {
        embedding: vecFloat('embedding', 4),
      });
      ctx.sqlite.exec(itemsVec.createSQL());
      ctx.sqlite.prepare('INSERT INTO search_items_vec(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
      ctx.sqlite.prepare('INSERT INTO search_items_vec(embedding) VALUES (?)').run(serializeVector([0, 1, 0, 0]));

      const queryVec = [1, 0, 0, 0];

      const results = ctx.sqlite.prepare(
        'SELECT rowid, distance FROM search_items_vec WHERE embedding MATCH ? AND k = ?'
      ).all(serializeVector(queryVec), 10) as Array<{ rowid: number; distance: number }>;

      expect(results).toHaveLength(2);
      expect(results[0].distance).toBeCloseTo(0, 5);
    });

    it('should build vector search with join that produces correct results', () => {
      ctx.sqlite.exec(`
        CREATE TABLE search_documents (
          id INTEGER PRIMARY KEY,
          title TEXT
        )
      `);
      const docsVec = vec0Table('search_documents_vec', {
        embedding: vecFloat('embedding', 4),
      });
      ctx.sqlite.exec(docsVec.createSQL());

      ctx.sqlite.prepare('INSERT INTO search_documents (id, title) VALUES (?, ?)').run(1, 'Doc A');
      ctx.sqlite.prepare('INSERT INTO search_documents (id, title) VALUES (?, ?)').run(2, 'Doc B');
      ctx.sqlite.prepare('INSERT INTO search_documents_vec(embedding) VALUES (?)').run(serializeVector([1, 0, 0, 0]));
      ctx.sqlite.prepare('INSERT INTO search_documents_vec(embedding) VALUES (?)').run(serializeVector([0, 1, 0, 0]));

      const queryVec = [1, 0, 0, 0];

      const results = ctx.sqlite.prepare(`
        SELECT d.*, v.distance
        FROM search_documents d
        INNER JOIN (
          SELECT rowid, distance
          FROM search_documents_vec
          WHERE embedding MATCH ?
          AND k = 5
        ) v ON d.id = v.rowid
        ORDER BY v.distance
      `).all(serializeVector(queryVec)) as Array<{ id: number; title: string; distance: number }>;

      expect(results).toHaveLength(2);
      expect(results[0].title).toBe('Doc A');
      expect(results[0].distance).toBeCloseTo(0, 5);
    });

    it('should generate knnWhere clause correctly', () => {
      const queryVec = [1, 0, 0, 0];
      const whereClause = knnWhere('embedding', queryVec, 10);

      const documents = sqliteTable('test_docs', {
        id: integer('id').primaryKey(),
      });

      const query = ctx.db.select().from(documents).where(whereClause).toSQL();
      // Verify MATCH and k clauses are present
      expect(query.sql.toLowerCase()).toContain('match');
      expect(query.sql.toLowerCase()).toMatch(/k\s*=\s*\?/); // k is parameterized
      // Verify params: vector buffer and k value
      expect(query.params.length).toBe(2);
      expect(query.params[1]).toBe(10); // k value
    });
  });

  describe('orderBy with distance function', () => {
    it('should produce valid ORDER BY clause', () => {
      const documents = sqliteTable('documents', {
        id: integer('id').primaryKey(),
        embedding: vector('embedding', { dimensions: 4 }),
      });

      const queryVec = [0.1, 0.2, 0.3, 0.4];
      const query = ctx.db
        .select()
        .from(documents)
        .orderBy(vec_distance_L2(documents.embedding, queryVec))
        .toSQL();

      expect(query.sql.toLowerCase()).toContain('order by');
      expect(query.sql).toContain('vec_distance_L2');
    });
  });
});
