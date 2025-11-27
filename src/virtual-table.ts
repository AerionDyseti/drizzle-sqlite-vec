import { serializeVector, deserializeVector } from './vector.js';

/**
 * Configuration for a vec0 virtual table column
 */
export interface Vec0ColumnConfig {
  /** Column name */
  name: string;
  /** Column type: 'float' for vectors, 'integer', 'text', etc. for auxiliary columns */
  type: 'float' | 'int8' | 'bit' | 'integer' | 'text' | 'blob';
  /** Dimensions (required for vector types) */
  dimensions?: number;
  /** Whether this is a primary key (for auxiliary columns) */
  primaryKey?: boolean;
  /** Distance metric for vector columns */
  distanceMetric?: 'L2' | 'cosine';
}

/**
 * Represents a vec0 virtual table definition
 */
export interface Vec0Table<TName extends string = string> {
  /** Table name */
  readonly name: TName;
  /** Column configurations */
  readonly columns: Vec0ColumnConfig[];
  /** Generate CREATE VIRTUAL TABLE statement as a raw SQL string */
  createSQL(): string;
  /** Generate DROP TABLE statement as a raw SQL string */
  dropSQL(): string;
}

/**
 * Builder for vec0 virtual table columns
 */
export class Vec0ColumnBuilder {
  private config: Vec0ColumnConfig;

  constructor(name: string, type: Vec0ColumnConfig['type'], dimensions?: number) {
    this.config = { name, type, dimensions };
  }

  /** Mark this column as a primary key */
  primaryKey(): this {
    this.config.primaryKey = true;
    return this;
  }

  /** Set the distance metric for vector columns (L2 or cosine) */
  distanceMetric(metric: 'L2' | 'cosine'): this {
    this.config.distanceMetric = metric;
    return this;
  }

  /** @internal */
  build(): Vec0ColumnConfig {
    return this.config;
  }
}

/**
 * Create a vector column for vec0 virtual tables
 *
 * @param name - Column name
 * @param dimensions - Number of dimensions in the vector
 *
 * @example
 * ```ts
 * const myVecTable = vec0Table('my_vectors', {
 *   embedding: vecFloat('embedding', 384),
 * });
 * ```
 */
export function vecFloat(name: string, dimensions: number): Vec0ColumnBuilder {
  return new Vec0ColumnBuilder(name, 'float', dimensions);
}

/**
 * Create an int8 quantized vector column for vec0 virtual tables
 *
 * @param name - Column name
 * @param dimensions - Number of dimensions in the vector
 */
export function vecInt8(name: string, dimensions: number): Vec0ColumnBuilder {
  return new Vec0ColumnBuilder(name, 'int8', dimensions);
}

/**
 * Create a binary/bit vector column for vec0 virtual tables
 *
 * @param name - Column name
 * @param dimensions - Number of bits in the vector
 */
export function vecBit(name: string, dimensions: number): Vec0ColumnBuilder {
  return new Vec0ColumnBuilder(name, 'bit', dimensions);
}

/**
 * Create an integer auxiliary column for vec0 virtual tables
 *
 * @param name - Column name
 */
export function vecInteger(name: string): Vec0ColumnBuilder {
  return new Vec0ColumnBuilder(name, 'integer');
}

/**
 * Create a text auxiliary column for vec0 virtual tables
 *
 * @param name - Column name
 */
export function vecText(name: string): Vec0ColumnBuilder {
  return new Vec0ColumnBuilder(name, 'text');
}

/**
 * Create a blob auxiliary column for vec0 virtual tables
 *
 * @param name - Column name
 */
export function vecBlob(name: string): Vec0ColumnBuilder {
  return new Vec0ColumnBuilder(name, 'blob');
}

/**
 * Create a vec0 virtual table definition
 *
 * @param name - Table name
 * @param columns - Column definitions
 * @returns Vec0 virtual table object
 *
 * @example
 * ```ts
 * const itemsVec = vec0Table('items_vec', {
 *   id: vecInteger('id').primaryKey(),
 *   embedding: vecFloat('embedding', 384),
 * });
 *
 * // Create the table
 * db.run(itemsVec.createSQL());
 *
 * // Insert vectors
 * db.run(sql`INSERT INTO items_vec(id, embedding) VALUES (${1}, ${vectorToSql(myVector)})`);
 * ```
 */
export function vec0Table<
  TName extends string,
  TColumns extends Record<string, Vec0ColumnBuilder>
>(
  name: TName,
  columns: TColumns
): Vec0Table<TName> {
  const columnConfigs = Object.entries(columns).map(([key, builder]) => {
    const config = builder.build();
    // Use the key as the column name if different
    if (!config.name) {
      config.name = key;
    }
    return config;
  });

  return {
    name,
    columns: columnConfigs,
    createSQL(): string {
      const columnDefs = columnConfigs.map((col) => {
        let def = col.name;

        if (col.type === 'float' && col.dimensions) {
          def += ` float[${col.dimensions}]`;
          if (col.distanceMetric) {
            def += ` distance_metric=${col.distanceMetric}`;
          }
        } else if (col.type === 'int8' && col.dimensions) {
          def += ` int8[${col.dimensions}]`;
          if (col.distanceMetric) {
            def += ` distance_metric=${col.distanceMetric}`;
          }
        } else if (col.type === 'bit' && col.dimensions) {
          def += ` bit[${col.dimensions}]`;
        } else if (col.type === 'integer') {
          def += ' integer';
          if (col.primaryKey) {
            def += ' primary key';
          }
        } else if (col.type === 'text') {
          def += ' text';
        } else if (col.type === 'blob') {
          def += ' blob';
        }

        return def;
      });

      return `CREATE VIRTUAL TABLE IF NOT EXISTS ${name} USING vec0(${columnDefs.join(', ')})`;
    },
    dropSQL(): string {
      return `DROP TABLE IF EXISTS ${name}`;
    },
  };
}

/**
 * Options for creating a shadow table
 */
export interface ShadowTableOptions {
  /** Table name */
  tableName: string;
  /** Vector column name */
  vectorColumn: string;
  /** Number of dimensions */
  dimensions: number;
  /** Primary key column name from the source table */
  idColumn: string;
}

/**
 * Create a vec0 virtual table that shadows a regular table
 *
 * This is a common pattern where you have a regular SQLite table
 * with your data, and a vec0 virtual table for vector search.
 *
 * @param options - Configuration options
 * @returns Vec0 virtual table object
 *
 * @example
 * ```ts
 * // Regular table with documents
 * const documents = sqliteTable('documents', {
 *   id: integer('id').primaryKey(),
 *   content: text('content'),
 * });
 *
 * // Shadow vec0 table for embeddings
 * const documentsVec = shadowVec0Table({
 *   tableName: 'documents_vec',
 *   vectorColumn: 'embedding',
 *   dimensions: 384,
 *   idColumn: 'document_id',
 * });
 * ```
 */
export function shadowVec0Table(options: ShadowTableOptions): Vec0Table {
  return vec0Table(options.tableName, {
    [options.idColumn]: vecInteger(options.idColumn).primaryKey(),
    [options.vectorColumn]: vecFloat(options.vectorColumn, options.dimensions),
  });
}
