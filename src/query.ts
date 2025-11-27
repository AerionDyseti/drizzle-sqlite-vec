import { sql, SQL, and, gt, lt, gte, lte, eq } from 'drizzle-orm';
import type { SQLiteColumn } from 'drizzle-orm/sqlite-core';
import { serializeVector } from './vector.js';

/**
 * Create a KNN search condition using sqlite-vec's MATCH syntax.
 *
 * This is used with vec0 virtual tables for efficient K-nearest neighbor search.
 *
 * @param column - The vector column to search
 * @param queryVector - The query vector to match against
 * @param k - Number of nearest neighbors to return
 * @returns SQL condition for use in WHERE clause
 *
 * @example
 * ```ts
 * // Basic KNN search
 * const results = db.all(sql`
 *   SELECT rowid, distance
 *   FROM items_vec
 *   WHERE ${vectorMatch('embedding', queryEmbedding, 10)}
 * `);
 * ```
 */
export function vectorMatch(
  column: string | SQLiteColumn,
  queryVector: number[],
  k: number
): SQL {
  const columnName = typeof column === 'string' ? column : column.name;
  const vectorBuffer = serializeVector(queryVector);

  return sql.raw(`${columnName} MATCH ?`)
    .append(sql` AND k = ${k}`)
    .mapWith(() => vectorBuffer);
}

/**
 * Create a WHERE clause for vec0 KNN search.
 *
 * sqlite-vec uses a special syntax: WHERE column MATCH vector AND k = N
 *
 * @param column - Column name to match against
 * @param queryVector - The query vector
 * @param k - Number of results to return
 * @returns SQL for the WHERE clause
 *
 * @example
 * ```ts
 * db.all(sql`
 *   SELECT rowid, distance
 *   FROM items_vec
 *   WHERE ${knnWhere('embedding', queryVector, 10)}
 * `);
 * ```
 */
export function knnWhere(
  column: string,
  queryVector: number[],
  k: number
): SQL {
  const vectorBuffer = serializeVector(queryVector);
  return sql`${sql.identifier(column)} MATCH ${vectorBuffer} AND k = ${k}`;
}

/**
 * Calculate L2 (Euclidean) distance between two vectors.
 *
 * @param vec1 - First vector (column reference or array)
 * @param vec2 - Second vector (column reference or array)
 * @returns SQL expression for L2 distance
 *
 * @example
 * ```ts
 * // Compare column to a query vector
 * const distance = vec_distance_L2(items.embedding, queryVector);
 *
 * // Use in ORDER BY
 * db.select().from(items).orderBy(distance);
 * ```
 */
export function vec_distance_L2(
  vec1: SQLiteColumn | number[] | SQL,
  vec2: SQLiteColumn | number[] | SQL
): SQL<number> {
  const v1 = toVectorSql(vec1);
  const v2 = toVectorSql(vec2);
  return sql<number>`vec_distance_L2(${v1}, ${v2})`;
}

/**
 * Calculate cosine distance between two vectors.
 * Cosine distance = 1 - cosine_similarity
 *
 * @param vec1 - First vector (column reference or array)
 * @param vec2 - Second vector (column reference or array)
 * @returns SQL expression for cosine distance
 *
 * @example
 * ```ts
 * const distance = vec_distance_cosine(items.embedding, queryVector);
 * ```
 */
export function vec_distance_cosine(
  vec1: SQLiteColumn | number[] | SQL,
  vec2: SQLiteColumn | number[] | SQL
): SQL<number> {
  const v1 = toVectorSql(vec1);
  const v2 = toVectorSql(vec2);
  return sql<number>`vec_distance_cosine(${v1}, ${v2})`;
}

/**
 * Calculate the length (L2 norm) of a vector.
 *
 * @param vec - Vector (column reference or array)
 * @returns SQL expression for vector length
 */
export function vec_length(vec: SQLiteColumn | number[] | SQL): SQL<number> {
  const v = toVectorSql(vec);
  return sql<number>`vec_length(${v})`;
}

/**
 * Normalize a vector to unit length.
 *
 * @param vec - Vector to normalize
 * @returns SQL expression for normalized vector
 */
export function vec_normalize(vec: SQLiteColumn | number[] | SQL): SQL {
  const v = toVectorSql(vec);
  return sql`vec_normalize(${v})`;
}

/**
 * Add two vectors element-wise.
 *
 * @param vec1 - First vector
 * @param vec2 - Second vector
 * @returns SQL expression for vector sum
 */
export function vec_add(
  vec1: SQLiteColumn | number[] | SQL,
  vec2: SQLiteColumn | number[] | SQL
): SQL {
  const v1 = toVectorSql(vec1);
  const v2 = toVectorSql(vec2);
  return sql`vec_add(${v1}, ${v2})`;
}

/**
 * Subtract two vectors element-wise.
 *
 * @param vec1 - First vector
 * @param vec2 - Second vector
 * @returns SQL expression for vector difference
 */
export function vec_sub(
  vec1: SQLiteColumn | number[] | SQL,
  vec2: SQLiteColumn | number[] | SQL
): SQL {
  const v1 = toVectorSql(vec1);
  const v2 = toVectorSql(vec2);
  return sql`vec_sub(${v1}, ${v2})`;
}

/**
 * Slice a vector to get a subset of its elements.
 *
 * @param vec - Vector to slice
 * @param start - Start index (0-based)
 * @param end - End index (exclusive)
 * @returns SQL expression for vector slice
 */
export function vec_slice(
  vec: SQLiteColumn | number[] | SQL,
  start: number,
  end: number
): SQL {
  const v = toVectorSql(vec);
  return sql`vec_slice(${v}, ${start}, ${end})`;
}

/**
 * Convert a JSON array to a vector.
 *
 * @param jsonArray - JSON array string or SQL expression
 * @returns SQL expression for the vector
 */
export function vec_f32(jsonArray: string | SQL): SQL {
  return sql`vec_f32(${jsonArray})`;
}

/**
 * Convert a vector to a JSON array string.
 *
 * @param vec - Vector to convert
 * @returns SQL expression for JSON representation
 */
export function vec_to_json(vec: SQLiteColumn | number[] | SQL): SQL<string> {
  const v = toVectorSql(vec);
  return sql<string>`vec_to_json(${v})`;
}

/**
 * Quantize a float32 vector to int8 for storage efficiency.
 *
 * @param vec - Vector to quantize
 * @returns SQL expression for quantized vector
 */
export function vec_quantize_i8(vec: SQLiteColumn | number[] | SQL): SQL {
  const v = toVectorSql(vec);
  return sql`vec_quantize_i8(${v})`;
}

/**
 * Convert a vector to binary (bit) representation.
 *
 * @param vec - Vector to convert
 * @returns SQL expression for binary vector
 */
export function vec_quantize_binary(vec: SQLiteColumn | number[] | SQL): SQL {
  const v = toVectorSql(vec);
  return sql`vec_quantize_binary(${v})`;
}

/**
 * Convert a value to a SQL vector expression
 * @internal
 */
function toVectorSql(value: SQLiteColumn | number[] | SQL): SQL {
  if (Array.isArray(value)) {
    return sql`${serializeVector(value)}`;
  }
  if (value instanceof SQL) {
    return value;
  }
  // Column reference
  return sql`${value}`;
}

/**
 * Options for vector search queries
 */
export interface VectorSearchOptions {
  /** Number of results to return */
  limit?: number;
  /** Minimum similarity threshold (for filtering) */
  minSimilarity?: number;
  /** Maximum distance threshold (for filtering) */
  maxDistance?: number;
  /** Distance metric to use */
  metric?: 'L2' | 'cosine';
}

/**
 * Build a raw SQL query for KNN search on a vec0 virtual table.
 *
 * @param tableName - Name of the vec0 virtual table
 * @param vectorColumn - Name of the vector column
 * @param queryVector - Query vector for similarity search
 * @param options - Search options
 * @returns SQL query for vector search
 *
 * @example
 * ```ts
 * const query = buildVectorSearchQuery(
 *   'items_vec',
 *   'embedding',
 *   queryEmbedding,
 *   { limit: 10 }
 * );
 *
 * const results = db.all(query);
 * ```
 */
export function buildVectorSearchQuery(
  tableName: string,
  vectorColumn: string,
  queryVector: number[],
  options: VectorSearchOptions = {}
): SQL {
  const { limit = 10 } = options;
  const vectorBuffer = serializeVector(queryVector);

  return sql`
    SELECT rowid, distance
    FROM ${sql.identifier(tableName)}
    WHERE ${sql.identifier(vectorColumn)} MATCH ${vectorBuffer}
      AND k = ${limit}
    ORDER BY distance
  `;
}

/**
 * Build a SQL query that joins vec0 search results with a regular table.
 *
 * @param vecTableName - Name of the vec0 virtual table
 * @param dataTableName - Name of the regular data table
 * @param vectorColumn - Name of the vector column in vec0 table
 * @param joinColumn - Column to join on (typically ID)
 * @param queryVector - Query vector for similarity search
 * @param options - Search options
 * @returns SQL query for vector search with join
 *
 * @example
 * ```ts
 * const query = buildVectorSearchWithJoin(
 *   'documents_vec',
 *   'documents',
 *   'embedding',
 *   'document_id',
 *   queryEmbedding,
 *   { limit: 10 }
 * );
 *
 * const results = db.all(query);
 * // Returns: [{ id, title, content, distance }, ...]
 * ```
 */
export function buildVectorSearchWithJoin(
  vecTableName: string,
  dataTableName: string,
  vectorColumn: string,
  joinColumn: string,
  queryVector: number[],
  options: VectorSearchOptions = {}
): SQL {
  const { limit = 10 } = options;
  const vectorBuffer = serializeVector(queryVector);

  return sql`
    SELECT ${sql.identifier(dataTableName)}.*, vec_results.distance
    FROM ${sql.identifier(dataTableName)}
    INNER JOIN (
      SELECT rowid, distance
      FROM ${sql.identifier(vecTableName)}
      WHERE ${sql.identifier(vectorColumn)} MATCH ${vectorBuffer}
        AND k = ${limit}
    ) AS vec_results ON ${sql.identifier(dataTableName)}.${sql.identifier(joinColumn.replace(/_id$/, '') === joinColumn ? 'id' : joinColumn.replace(/_id$/, ''))} = vec_results.rowid
    ORDER BY vec_results.distance
  `;
}
