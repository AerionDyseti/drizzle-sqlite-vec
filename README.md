# drizzle-sqlite-vec

SQLite vector search extension for [Drizzle ORM](https://orm.drizzle.team/) using [sqlite-vec](https://github.com/asg017/sqlite-vec).

## Installation

```bash
npm install drizzle-sqlite-vec
```

### Peer Dependencies

This package requires the following peer dependencies:

```bash
npm install drizzle-orm better-sqlite3 sqlite-vec
```

## Quick Start

```ts
import Database from 'better-sqlite3';
import * as sqlite_vec from 'sqlite-vec';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';
import {
  vector,
  vec0Table,
  vecFloat,
  vecInteger,
  serializeVector,
} from 'drizzle-sqlite-vec';

// 1. Initialize database with sqlite-vec extension
const sqlite = new Database(':memory:');
sqlite_vec.load(sqlite);
const db = drizzle(sqlite);

// 2. Create a vec0 virtual table for vector search
const itemsVec = vec0Table('items_vec', {
  itemId: vecInteger('item_id').primaryKey(),
  embedding: vecFloat('embedding', 384),
});

// Create the table
sqlite.exec(itemsVec.createSQL());

// 3. Insert vectors
const embedding = Array.from({ length: 384 }, () => Math.random());
sqlite
  .prepare('INSERT INTO items_vec(embedding) VALUES (?)')
  .run(serializeVector(embedding));

// 4. Search for similar vectors (KNN search)
const queryEmbedding = Array.from({ length: 384 }, () => Math.random());
const results = sqlite
  .prepare('SELECT rowid, distance FROM items_vec WHERE embedding MATCH ? AND k = 10')
  .all(serializeVector(queryEmbedding));
```

## Usage

### Setting Up the Database

Always load the sqlite-vec extension before using vector operations:

```ts
import Database from 'better-sqlite3';
import * as sqlite_vec from 'sqlite-vec';

const sqlite = new Database('my-database.db');
sqlite_vec.load(sqlite);
```

### Vector Columns in Regular Tables

Use the `vector` column type to store embeddings in regular SQLite tables as BLOBs:

```ts
import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';
import { vector, serializeVector, deserializeVector } from 'drizzle-sqlite-vec';

// Define schema
const documents = sqliteTable('documents', {
  id: integer('id').primaryKey(),
  content: text('content'),
  embedding: vector('embedding', { dimensions: 384 }),
});

// Create table
sqlite.exec(`
  CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding BLOB
  )
`);

// Insert with vector
const embedding = [0.1, 0.2, 0.3, /* ... */];
sqlite
  .prepare('INSERT INTO documents (id, content, embedding) VALUES (?, ?, ?)')
  .run(1, 'Hello world', serializeVector(embedding));

// Read and deserialize
const row = sqlite.prepare('SELECT * FROM documents WHERE id = 1').get();
const vector = deserializeVector(row.embedding);
```

### vec0 Virtual Tables (Recommended for Search)

For efficient K-nearest neighbor (KNN) search, use vec0 virtual tables:

```ts
import { vec0Table, vecFloat, vecInteger, serializeVector } from 'drizzle-sqlite-vec';

// Define a vec0 virtual table
const itemsVec = vec0Table('items_vec', {
  itemId: vecInteger('item_id').primaryKey(),
  embedding: vecFloat('embedding', 384),
});

// Create the table
sqlite.exec(itemsVec.createSQL());
// Generates: CREATE VIRTUAL TABLE IF NOT EXISTS items_vec USING vec0(item_id integer primary key, embedding float[384])

// Drop the table
sqlite.exec(itemsVec.dropSQL());
```

#### Vector Column Types

| Function | Description |
|----------|-------------|
| `vecFloat(name, dimensions)` | Float32 vector (default, most common) |
| `vecInt8(name, dimensions)` | Int8 quantized vector (smaller storage) |
| `vecBit(name, dimensions)` | Binary vector (smallest storage) |

#### Auxiliary Column Types

| Function | Description |
|----------|-------------|
| `vecInteger(name)` | Integer column (use for IDs) |
| `vecText(name)` | Text column |
| `vecBlob(name)` | Blob column |

#### Distance Metrics

```ts
// Set distance metric for a vector column
const itemsVec = vec0Table('items_vec', {
  embedding: vecFloat('embedding', 384).distanceMetric('cosine'), // or 'L2'
});
```

### KNN Search

Perform K-nearest neighbor search using the `MATCH` syntax:

```ts
// Insert vectors
const vectors = [
  [1.0, 0.0, 0.0, 0.0],
  [0.9, 0.1, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0],
];

vectors.forEach((vec) => {
  sqlite
    .prepare('INSERT INTO items_vec(embedding) VALUES (?)')
    .run(serializeVector(vec));
});

// Search for top 5 nearest neighbors
const queryVector = [1.0, 0.0, 0.0, 0.0];
const results = sqlite
  .prepare('SELECT rowid, distance FROM items_vec WHERE embedding MATCH ? AND k = 5')
  .all(serializeVector(queryVector));

// Results are sorted by distance (closest first)
// [{ rowid: 1, distance: 0 }, { rowid: 2, distance: 0.1414... }, ...]
```

### Distance Functions

Calculate distances between vectors using SQL functions:

```ts
import { vec_distance_L2, vec_distance_cosine, serializeVector } from 'drizzle-sqlite-vec';

// L2 (Euclidean) distance
const l2Result = sqlite
  .prepare('SELECT vec_distance_L2(?, ?) as distance')
  .get(serializeVector([1, 0, 0, 0]), serializeVector([0, 1, 0, 0]));
// distance ≈ 1.414 (sqrt(2))

// Cosine distance
const cosineResult = sqlite
  .prepare('SELECT vec_distance_cosine(?, ?) as distance')
  .get(serializeVector([1, 0, 0, 0]), serializeVector([1, 0, 0, 0]));
// distance = 0 (identical vectors)
```

### Vector Operations

Additional sqlite-vec functions available:

```ts
import {
  vec_length,      // Vector magnitude (L2 norm)
  vec_normalize,   // Normalize to unit length
  vec_add,         // Element-wise addition
  vec_sub,         // Element-wise subtraction
  vec_slice,       // Extract subset of dimensions
  vec_f32,         // Convert JSON array to vector
  vec_to_json,     // Convert vector to JSON array
  vec_quantize_i8, // Quantize to int8
  vec_quantize_binary, // Convert to binary
} from 'drizzle-sqlite-vec';
```

### Type Utilities

Helper functions for working with vectors in TypeScript:

```ts
import {
  typedVector,
  createVector,
  zeroVector,
  randomVector,
  normalizeVector,
  validateVector,
  l2Distance,
  cosineSimilarity,
  cosineDistance,
  dotProduct,
} from 'drizzle-sqlite-vec';

// Create typed vectors
const vec = typedVector(4, [1, 2, 3, 4]);

// Generate vectors
const zeros = zeroVector(384);
const random = randomVector(384);
const custom = createVector(384, (i) => i * 0.01);

// Normalize to unit length
const normalized = normalizeVector([3, 4, 0, 0]); // [0.6, 0.8, 0, 0]

// Validate vectors
validateVector([1, 2, 3]);        // true
validateVector([1, 2, 3], 3);     // true (checks dimensions)
validateVector([1, 2, NaN]);      // throws Error

// Calculate distances (JavaScript, not SQL)
l2Distance([0, 0], [3, 4]);       // 5
cosineSimilarity([1, 0], [1, 0]); // 1
cosineDistance([1, 0], [0, 1]);   // 1
dotProduct([1, 2], [3, 4]);       // 11
```

### Common Embedding Dimensions

Reference constants for popular embedding models:

```ts
import { EmbeddingDimensions } from 'drizzle-sqlite-vec';

EmbeddingDimensions.OPENAI_ADA_002;   // 1536
EmbeddingDimensions.OPENAI_3_SMALL;   // 1536
EmbeddingDimensions.OPENAI_3_LARGE;   // 3072
EmbeddingDimensions.COHERE_V3;        // 1024
EmbeddingDimensions.MINILM_L6_V2;     // 384
EmbeddingDimensions.MPNET_BASE_V2;    // 768
EmbeddingDimensions.BGE_SMALL;        // 384
EmbeddingDimensions.BGE_BASE;         // 768
EmbeddingDimensions.BGE_LARGE;        // 1024
```

## Full Example: Document Search

```ts
import Database from 'better-sqlite3';
import * as sqlite_vec from 'sqlite-vec';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';
import {
  vec0Table,
  vecFloat,
  vecInteger,
  serializeVector,
  EmbeddingDimensions,
} from 'drizzle-sqlite-vec';

// Initialize
const sqlite = new Database('documents.db');
sqlite_vec.load(sqlite);
const db = drizzle(sqlite);

// Regular table for document data
sqlite.exec(`
  CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT
  )
`);

// vec0 table for embeddings
const documentsVec = vec0Table('documents_vec', {
  docId: vecInteger('doc_id').primaryKey(),
  embedding: vecFloat('embedding', EmbeddingDimensions.MINILM_L6_V2),
});
sqlite.exec(documentsVec.createSQL());

// Insert a document with its embedding
function insertDocument(id: number, title: string, content: string, embedding: number[]) {
  sqlite.prepare('INSERT INTO documents (id, title, content) VALUES (?, ?, ?)').run(id, title, content);
  sqlite.prepare('INSERT INTO documents_vec (doc_id, embedding) VALUES (?, ?)').run(id, serializeVector(embedding));
}

// Search for similar documents
function searchDocuments(queryEmbedding: number[], limit = 10) {
  const results = sqlite.prepare(`
    SELECT d.*, v.distance
    FROM documents d
    INNER JOIN (
      SELECT doc_id, distance
      FROM documents_vec
      WHERE embedding MATCH ?
        AND k = ?
    ) v ON d.id = v.doc_id
    ORDER BY v.distance
  `).all(serializeVector(queryEmbedding), limit);

  return results;
}

// Usage
const embedding = await getEmbedding('Introduction to machine learning'); // Your embedding function
insertDocument(1, 'ML Basics', 'Introduction to machine learning...', embedding);

const queryEmbed = await getEmbedding('what is AI?');
const similar = searchDocuments(queryEmbed, 5);
```

## API Reference

### Vector Serialization

- `serializeVector(vector: number[]): Buffer` - Convert array to sqlite-vec format
- `deserializeVector(buffer: Buffer): number[]` - Convert buffer back to array

### Virtual Table

- `vec0Table(name, columns)` - Create a vec0 table definition
- `vecFloat(name, dimensions)` - Float32 vector column
- `vecInt8(name, dimensions)` - Int8 vector column
- `vecBit(name, dimensions)` - Binary vector column
- `vecInteger(name)` - Integer auxiliary column
- `vecText(name)` - Text auxiliary column
- `vecBlob(name)` - Blob auxiliary column

### SQL Functions

- `vec_distance_L2(vec1, vec2)` - Euclidean distance
- `vec_distance_cosine(vec1, vec2)` - Cosine distance
- `vec_length(vec)` - Vector magnitude
- `vec_normalize(vec)` - Normalize vector
- `vec_add(vec1, vec2)` - Add vectors
- `vec_sub(vec1, vec2)` - Subtract vectors
- `vec_slice(vec, start, end)` - Slice vector
- `vec_f32(json)` - JSON to vector
- `vec_to_json(vec)` - Vector to JSON
- `vec_quantize_i8(vec)` - Quantize to int8
- `vec_quantize_binary(vec)` - Convert to binary

### Type Utilities

- `typedVector(dimensions, values)` - Create typed vector
- `createVector(dimensions, generator)` - Generate vector
- `zeroVector(dimensions)` - Zero vector
- `randomVector(dimensions)` - Random vector
- `normalizeVector(vector)` - Normalize to unit length
- `validateVector(value, dimensions?)` - Validate vector
- `l2Distance(a, b)` - L2 distance (JS)
- `cosineSimilarity(a, b)` - Cosine similarity (JS)
- `cosineDistance(a, b)` - Cosine distance (JS)
- `dotProduct(a, b)` - Dot product (JS)

## License

MIT
