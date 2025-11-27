import { customType } from 'drizzle-orm/sqlite-core';
import { sql, SQL } from 'drizzle-orm';

/**
 * Configuration for vector columns
 */
export interface VectorConfig {
  dimensions: number;
}

/**
 * Serialize a number array to Float32Array buffer for sqlite-vec
 */
export function serializeVector(vector: number[]): Buffer {
  const float32 = new Float32Array(vector);
  return Buffer.from(float32.buffer);
}

/**
 * Deserialize a Buffer from sqlite-vec to number array
 */
export function deserializeVector(buffer: Buffer): number[] {
  const float32 = new Float32Array(
    buffer.buffer,
    buffer.byteOffset,
    buffer.byteLength / Float32Array.BYTES_PER_ELEMENT
  );
  return Array.from(float32);
}

/**
 * Create a vector column type for use in regular SQLite tables.
 * The vector is stored as a BLOB (Float32Array binary format).
 *
 * @param config - Vector configuration with dimensions
 * @returns A custom column type for vectors
 *
 * @example
 * ```ts
 * import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';
 * import { vector } from 'drizzle-sqlite-vec';
 *
 * const documents = sqliteTable('documents', {
 *   id: integer('id').primaryKey(),
 *   content: text('content'),
 *   embedding: vector('embedding', { dimensions: 384 }),
 * });
 * ```
 */
export const vector = customType<{
  data: number[];
  driverData: Buffer;
  config: VectorConfig;
  configRequired: true;
}>({
  dataType(config) {
    // In regular tables, vectors are stored as BLOBs
    // The dimensions are for documentation/type safety
    return `BLOB`;
  },
  toDriver(value: number[]): Buffer {
    return serializeVector(value);
  },
  fromDriver(value: Buffer): number[] {
    return deserializeVector(value);
  },
});

/**
 * Create a vector type string for use in virtual tables.
 * sqlite-vec uses the float[N] syntax for virtual tables.
 *
 * @param dimensions - Number of dimensions in the vector
 * @returns SQL type string like "float[384]"
 */
export function vectorType(dimensions: number): string {
  return `float[${dimensions}]`;
}

/**
 * Convert a number array to a SQL literal for vector operations.
 * This creates the proper binary format that sqlite-vec expects.
 *
 * @param vector - Array of numbers representing the vector
 * @returns SQL expression with the serialized vector
 */
export function vectorToSql(vector: number[]): SQL {
  return sql`${serializeVector(vector)}`;
}
