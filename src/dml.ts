import { sql, SQL } from 'drizzle-orm';
import { serializeVector } from './vector.js';
import type { Vec0Table, Vec0ColumnConfig } from './virtual-table.js';

/**
 * Map Vec0 column types to their TypeScript equivalents for insert/update operations
 */
type Vec0ColumnValueType<T extends Vec0ColumnConfig> =
  T['type'] extends 'float' | 'int8' | 'bit' ? number[] :
  T['type'] extends 'integer' ? number :
  T['type'] extends 'text' ? string :
  T['type'] extends 'blob' ? Buffer | Uint8Array :
  never;

/**
 * Infer the insert type from a Vec0Table's columns
 * Primary key columns are optional (sqlite-vec auto-assigns rowid)
 */
export type Vec0InsertValues<T extends Vec0Table> = {
  [K in T['columns'][number] as K['name']]: K['primaryKey'] extends true
    ? Vec0ColumnValueType<K> | undefined
    : Vec0ColumnValueType<K>;
};

/**
 * Infer the update type from a Vec0Table's columns (all optional)
 */
export type Vec0UpdateValues<T extends Vec0Table> = {
  [K in T['columns'][number] as K['name']]?: Vec0ColumnValueType<K>;
};

/**
 * Check if a column is a vector type
 */
function isVectorColumn(config: Vec0ColumnConfig): boolean {
  return config.type === 'float' || config.type === 'int8' || config.type === 'bit';
}

/**
 * Serialize a value for SQL insertion based on column type
 */
function serializeValue(value: unknown, config: Vec0ColumnConfig): unknown {
  if (value === null || value === undefined) {
    return null;
  }
  if (isVectorColumn(config) && Array.isArray(value)) {
    return serializeVector(value as number[]);
  }
  return value;
}

/**
 * Insert a row into a vec0 virtual table.
 *
 * @param table - The vec0 table definition
 * @param values - The values to insert (vector columns accept number[])
 * @returns SQL INSERT statement
 *
 * @example
 * ```ts
 * const itemsVec = vec0Table('items_vec', {
 *   id: vecInteger('id').primaryKey(),
 *   embedding: vecFloat('embedding', 384),
 * });
 *
 * const query = insertVec0(itemsVec, {
 *   id: 1,
 *   embedding: [0.1, 0.2, ...], // 384-dimensional vector
 * });
 *
 * // Execute with your driver
 * db.run(query);
 * ```
 */
export function insertVec0<T extends Vec0Table>(
  table: T,
  values: Record<string, unknown>
): SQL {
  const columns: string[] = [];
  const sqlValues: SQL[] = [];

  for (const col of table.columns) {
    const value = values[col.name];
    if (value !== undefined) {
      columns.push(col.name);
      const serialized = serializeValue(value, col);
      sqlValues.push(sql`${serialized}`);
    }
  }

  if (columns.length === 0) {
    throw new Error('insertVec0: No values provided for insert');
  }

  const columnList = sql.join(
    columns.map((name) => sql.identifier(name)),
    sql`, `
  );
  const valueList = sql.join(sqlValues, sql`, `);

  return sql`INSERT INTO ${sql.identifier(table.name)} (${columnList}) VALUES (${valueList})`;
}

/**
 * Insert multiple rows into a vec0 virtual table.
 *
 * @param table - The vec0 table definition
 * @param rows - Array of row values to insert
 * @returns SQL INSERT statement with multiple value sets
 *
 * @example
 * ```ts
 * const query = insertManyVec0(itemsVec, [
 *   { id: 1, embedding: [...] },
 *   { id: 2, embedding: [...] },
 * ]);
 *
 * db.run(query);
 * ```
 */
export function insertManyVec0<T extends Vec0Table>(
  table: T,
  rows: Record<string, unknown>[]
): SQL {
  if (rows.length === 0) {
    throw new Error('insertManyVec0: No rows provided for insert');
  }

  // Use the columns from the first row to determine column order
  const firstRow = rows[0];
  const columns: string[] = [];
  const columnConfigs: Vec0ColumnConfig[] = [];

  for (const col of table.columns) {
    if (firstRow[col.name] !== undefined) {
      columns.push(col.name);
      columnConfigs.push(col);
    }
  }

  if (columns.length === 0) {
    throw new Error('insertManyVec0: No values provided for insert');
  }

  const columnList = sql.join(
    columns.map((name) => sql.identifier(name)),
    sql`, `
  );

  const valueSets = rows.map((row) => {
    const sqlValues = columnConfigs.map((col) => {
      const value = row[col.name];
      const serialized = serializeValue(value, col);
      return sql`${serialized}`;
    });
    return sql`(${sql.join(sqlValues, sql`, `)})`;
  });

  return sql`INSERT INTO ${sql.identifier(table.name)} (${columnList}) VALUES ${sql.join(valueSets, sql`, `)}`;
}

/**
 * Where condition for vec0 DML operations.
 * Can be a rowid number, or a SQL expression for more complex conditions.
 */
export type Vec0WhereCondition = number | SQL;

/**
 * Delete rows from a vec0 virtual table.
 *
 * @param table - The vec0 table definition
 * @param where - Row ID or SQL WHERE condition
 * @returns SQL DELETE statement
 *
 * @example
 * ```ts
 * // Delete by rowid
 * const query = deleteVec0(itemsVec, 1);
 *
 * // Delete with SQL condition
 * const query = deleteVec0(itemsVec, sql`id = ${someId}`);
 *
 * db.run(query);
 * ```
 */
export function deleteVec0<T extends Vec0Table>(
  table: T,
  where: Vec0WhereCondition
): SQL {
  const whereClause = typeof where === 'number' ? sql`rowid = ${where}` : where;

  return sql`DELETE FROM ${sql.identifier(table.name)} WHERE ${whereClause}`;
}

/**
 * Update rows in a vec0 virtual table.
 *
 * @param table - The vec0 table definition
 * @param values - The values to update (vector columns accept number[])
 * @param where - Row ID or SQL WHERE condition
 * @returns SQL UPDATE statement
 *
 * @example
 * ```ts
 * // Update by rowid
 * const query = updateVec0(itemsVec, { embedding: [...] }, 1);
 *
 * // Update with SQL condition
 * const query = updateVec0(itemsVec, { embedding: [...] }, sql`id = ${someId}`);
 *
 * db.run(query);
 * ```
 */
export function updateVec0<T extends Vec0Table>(
  table: T,
  values: Record<string, unknown>,
  where: Vec0WhereCondition
): SQL {
  const setClauses: SQL[] = [];

  for (const col of table.columns) {
    const value = values[col.name];
    if (value !== undefined) {
      const serialized = serializeValue(value, col);
      setClauses.push(sql`${sql.identifier(col.name)} = ${serialized}`);
    }
  }

  if (setClauses.length === 0) {
    throw new Error('updateVec0: No values provided for update');
  }

  const setClause = sql.join(setClauses, sql`, `);
  const whereClause = typeof where === 'number' ? sql`rowid = ${where}` : where;

  return sql`UPDATE ${sql.identifier(table.name)} SET ${setClause} WHERE ${whereClause}`;
}
