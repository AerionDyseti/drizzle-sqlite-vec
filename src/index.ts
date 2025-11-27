/**
 * drizzle-sqlite-vec
 *
 * SQLite vector search extension for Drizzle ORM using sqlite-vec.
 *
 * @packageDocumentation
 *
 * @example Basic Usage
 * ```ts
 * import Database from 'better-sqlite3';
 * import * as sqlite_vec from 'sqlite-vec';
 * import { drizzle } from 'drizzle-orm/better-sqlite3';
 * import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';
 * import { sql } from 'drizzle-orm';
 * import {
 *   vector,
 *   vec0Table,
 *   vecFloat,
 *   vecInteger,
 *   serializeVector,
 *   buildVectorSearchQuery,
 * } from 'drizzle-sqlite-vec';
 *
 * // Initialize database with sqlite-vec
 * const sqlite = new Database(':memory:');
 * sqlite_vec.load(sqlite);
 * const db = drizzle(sqlite);
 *
 * // Regular table with vector column (stored as BLOB)
 * const documents = sqliteTable('documents', {
 *   id: integer('id').primaryKey(),
 *   content: text('content'),
 *   embedding: vector('embedding', { dimensions: 384 }),
 * });
 *
 * // Or use a vec0 virtual table for KNN search
 * const docsVec = vec0Table('docs_vec', {
 *   docId: vecInteger('doc_id').primaryKey(),
 *   embedding: vecFloat('embedding', 384),
 * });
 *
 * // Create the virtual table
 * sqlite.exec(docsVec.createSQL().toQuery().sql);
 *
 * // Insert a vector
 * const embedding = [0.1, 0.2, ...]; // 384-dimensional vector
 * sqlite.prepare('INSERT INTO docs_vec(doc_id, embedding) VALUES (?, ?)')
 *   .run(1, serializeVector(embedding));
 *
 * // Search for similar vectors
 * const results = sqlite.prepare(`
 *   SELECT doc_id, distance
 *   FROM docs_vec
 *   WHERE embedding MATCH ?
 *     AND k = 10
 * `).all(serializeVector(queryEmbedding));
 * ```
 */

// Vector column type for regular tables
export {
  vector,
  serializeVector,
  deserializeVector,
  vectorType,
  vectorToSql,
  type VectorConfig,
} from './vector.js';

// Virtual table support for vec0
export {
  vec0Table,
  vecFloat,
  vecInt8,
  vecBit,
  vecInteger,
  vecText,
  vecBlob,
  shadowVec0Table,
  Vec0ColumnBuilder,
  type Vec0Table,
  type Vec0ColumnConfig,
  type ShadowTableOptions,
} from './virtual-table.js';

// Query helpers and vector functions
export {
  // KNN search
  vectorMatch,
  knnWhere,
  buildVectorSearchQuery,
  buildVectorSearchWithJoin,
  // Distance functions
  vec_distance_L2,
  vec_distance_cosine,
  // Vector operations
  vec_length,
  vec_normalize,
  vec_add,
  vec_sub,
  vec_slice,
  // Conversion functions
  vec_f32,
  vec_to_json,
  // Quantization
  vec_quantize_i8,
  vec_quantize_binary,
  type VectorSearchOptions,
} from './query.js';

// Type utilities
export {
  // Types
  type Tuple,
  type Vector,
  // Vector creation
  typedVector,
  createVector,
  zeroVector,
  randomVector,
  // Validation
  validateVector,
  // Math utilities
  normalizeVector,
  l2Distance,
  cosineSimilarity,
  cosineDistance,
  dotProduct,
  // Constants
  EmbeddingDimensions,
  type EmbeddingDimensionName,
} from './types.js';
