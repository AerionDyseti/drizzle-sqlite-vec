import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';
import {
  vector,
  serializeVector,
  deserializeVector,
  vec0Table,
  vecFloat,
  vecInt8,
  vecBit,
  vecInteger,
  vecText,
  vecBlob,
  shadowVec0Table,
  randomVector,
  zeroVector,
} from '../src/index.js';
import { createTestContext, closeTestContext, type TestContext } from './setup.js';

describe('Schema & Column Definition', () => {
  let ctx: TestContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  afterEach(() => {
    closeTestContext(ctx);
  });

  it('should accept valid dimension count and store it correctly', () => {
    const itemsVec = vec0Table('valid_dims', {
      embedding: vecFloat('embedding', 384),
    });

    // The generated SQL proves dimensions are stored correctly
    const createSql = itemsVec.createSQL();
    expect(createSql).toContain('float[384]');
  });

  it('should compile to valid CREATE TABLE SQL with correct sqlite-vec syntax', () => {
    const itemsVec = vec0Table('items_vec', {
      id: vecInteger('id').primaryKey(),
      embedding: vecFloat('embedding', 128),
    });

    // Exact match is intentional - this is a contract test to catch any SQL format changes
    const createSql = itemsVec.createSQL();
    expect(createSql).toBe(
      'CREATE VIRTUAL TABLE IF NOT EXISTS items_vec USING vec0(id integer primary key, embedding float[128])'
    );
  });

  it('should support multiple vector columns on the same table', () => {
    const multiVec = vec0Table('multi_vec', {
      id: vecInteger('id').primaryKey(),
      embedding1: vecFloat('embedding1', 384),
      embedding2: vecFloat('embedding2', 768),
    });

    const createSql = multiVec.createSQL();
    expect(createSql).toContain('embedding1 float[384]');
    expect(createSql).toContain('embedding2 float[768]');

    ctx.sqlite.exec(createSql);
    ctx.sqlite
      .prepare('INSERT INTO multi_vec(embedding1, embedding2) VALUES (?, ?)')
      .run(serializeVector(zeroVector(384)), serializeVector(zeroVector(768)));

    const count = ctx.sqlite.prepare('SELECT COUNT(*) as count FROM multi_vec').get() as { count: number };
    expect(count.count).toBe(1);
  });

  it('should work with vector columns alongside regular Drizzle columns', () => {
    const documents = sqliteTable('documents', {
      id: integer('id').primaryKey(),
      title: text('title'),
      content: text('content'),
      embedding: vector('embedding', { dimensions: 384 }),
      metadata: text('metadata'),
    });

    // Note: Manual table creation is required because this library doesn't yet
    // support Drizzle's migrate()/push(). The vector column maps to BLOB storage.
    ctx.sqlite.exec(`
      CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        title TEXT,
        content TEXT,
        embedding BLOB,
        metadata TEXT
      )
    `);

    const embedding = randomVector(384);
    ctx.sqlite
      .prepare('INSERT INTO documents (id, title, content, embedding, metadata) VALUES (?, ?, ?, ?, ?)')
      .run(1, 'Test Title', 'Test Content', serializeVector(embedding), '{"key": "value"}');

    const result = ctx.sqlite.prepare('SELECT * FROM documents WHERE id = 1').get() as any;
    expect(result.id).toBe(1);
    expect(result.title).toBe('Test Title');
    expect(result.content).toBe('Test Content');
    expect(result.metadata).toBe('{"key": "value"}');

    // Verify embedding round-trips with correct values, not just length
    const restored = deserializeVector(result.embedding);
    expect(restored).toHaveLength(384);
    restored.forEach((val, i) => expect(val).toBeCloseTo(embedding[i], 5));
  });

  it('should support all vec0 column types', () => {
    // Test float vector with auxiliary columns
    const floatTable = vec0Table('float_types', {
      floatVec: vecFloat('float_vec', 4),
      textCol: vecText('text_col'),
    });
    expect(floatTable.createSQL()).toContain('float_vec float[4]');
    expect(floatTable.createSQL()).toContain('text_col text');
    ctx.sqlite.exec(floatTable.createSQL());

    ctx.sqlite
      .prepare('INSERT INTO float_types(float_vec, text_col) VALUES (?, ?)')
      .run(serializeVector([1, 0, 0, 0]), 'test metadata');

    const floatResults = ctx.sqlite
      .prepare('SELECT rowid, distance FROM float_types WHERE float_vec MATCH ? AND k = 1')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ rowid: number; distance: number }>;
    expect(floatResults).toHaveLength(1);
    expect(floatResults[0].distance).toBeCloseTo(0, 5);

    // Test int8 vector (requires int8 serialization format)
    const int8Table = vec0Table('int8_types', {
      int8Vec: vecInt8('int8_vec', 4),
    });
    expect(int8Table.createSQL()).toContain('int8_vec int8[4]');
    ctx.sqlite.exec(int8Table.createSQL());

    // Test bit vector
    const bitTable = vec0Table('bit_types', {
      bitVec: vecBit('bit_vec', 8),
    });
    expect(bitTable.createSQL()).toContain('bit_vec bit[8]');
    ctx.sqlite.exec(bitTable.createSQL());

    // Test integer primary key
    const idTable = vec0Table('id_types', {
      id: vecInteger('id').primaryKey(),
      embedding: vecFloat('embedding', 4),
    });
    expect(idTable.createSQL()).toContain('id integer primary key');
    ctx.sqlite.exec(idTable.createSQL());
  });

  it('should support distance metric configuration', () => {
    // sqlite-vec supports: L2, cosine, L1
    const cosineTable = vec0Table('cosine_vec', {
      embedding: vecFloat('embedding', 4).distanceMetric('cosine'),
    });
    expect(cosineTable.createSQL()).toContain('distance_metric=cosine');
    ctx.sqlite.exec(cosineTable.createSQL());

    const l2Table = vec0Table('l2_vec', {
      embedding: vecFloat('embedding', 4).distanceMetric('L2'),
    });
    expect(l2Table.createSQL()).toContain('distance_metric=L2');
    ctx.sqlite.exec(l2Table.createSQL());

    const l1Table = vec0Table('l1_vec', {
      embedding: vecFloat('embedding', 4).distanceMetric('L1'),
    });
    expect(l1Table.createSQL()).toContain('distance_metric=L1');
    ctx.sqlite.exec(l1Table.createSQL());
  });

  it('should generate valid DROP TABLE SQL', () => {
    const itemsVec = vec0Table('items_vec', {
      embedding: vecFloat('embedding', 4),
    });

    expect(itemsVec.dropSQL()).toBe('DROP TABLE IF EXISTS items_vec');
  });

  it('should create shadow tables for regular tables', () => {
    const shadow = shadowVec0Table({
      tableName: 'documents_vec',
      vectorColumn: 'embedding',
      dimensions: 4,
      idColumn: 'doc_id',
    });

    expect(shadow.name).toBe('documents_vec');
    const createSql = shadow.createSQL();
    expect(createSql).toContain('doc_id integer primary key');
    expect(createSql).toContain('embedding float[4]');

    // Verify the generated SQL actually executes
    ctx.sqlite.exec(createSql);

    // Verify the table is usable (vec0 auto-assigns primary keys)
    ctx.sqlite
      .prepare('INSERT INTO documents_vec(embedding) VALUES (?)')
      .run(serializeVector([1, 0, 0, 0]));

    const results = ctx.sqlite
      .prepare('SELECT doc_id, distance FROM documents_vec WHERE embedding MATCH ? AND k = 1')
      .all(serializeVector([1, 0, 0, 0])) as Array<{ doc_id: number; distance: number }>;

    expect(results).toHaveLength(1);
    expect(results[0].doc_id).toBe(1); // auto-assigned
    expect(results[0].distance).toBeCloseTo(0, 5);
  });
});
