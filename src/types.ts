/**
 * Type-level utilities for vector dimension safety
 */

/**
 * Create a tuple type of fixed length N
 *
 * @example
 * ```ts
 * type Vec3 = Tuple<number, 3>; // [number, number, number]
 * ```
 */
export type Tuple<T, N extends number> = N extends N
  ? number extends N
    ? T[]
    : _TupleOf<T, N, []>
  : never;

type _TupleOf<T, N extends number, R extends unknown[]> = R['length'] extends N
  ? R
  : _TupleOf<T, N, [T, ...R]>;

/**
 * Type representing a vector with a specific number of dimensions.
 * At runtime it's just a number[], but provides compile-time safety.
 *
 * @example
 * ```ts
 * type Embedding384 = Vector<384>;
 * const embed: Embedding384 = createVector(384, () => Math.random());
 * ```
 */
export type Vector<N extends number = number> = number[] & { readonly __dimensions?: N };

/**
 * Create a typed vector with the specified dimensions.
 * Runtime validation ensures the array has the correct length.
 *
 * @param dimensions - Expected number of dimensions
 * @param values - Array of values (must match dimensions length)
 * @returns Typed vector
 * @throws Error if values length doesn't match dimensions
 *
 * @example
 * ```ts
 * const vec = typedVector(3, [1, 2, 3]);
 * // vec is typed as Vector<3>
 * ```
 */
export function typedVector<N extends number>(
  dimensions: N,
  values: number[]
): Vector<N> {
  if (values.length !== dimensions) {
    throw new Error(
      `Vector dimension mismatch: expected ${dimensions}, got ${values.length}`
    );
  }
  return values as Vector<N>;
}

/**
 * Create a vector by calling a generator function for each dimension.
 *
 * @param dimensions - Number of dimensions
 * @param generator - Function to generate each value (receives index)
 * @returns Typed vector
 *
 * @example
 * ```ts
 * // Create a zero vector
 * const zeros = createVector(384, () => 0);
 *
 * // Create a vector with random values
 * const random = createVector(384, () => Math.random());
 *
 * // Create a vector with index-based values
 * const indexed = createVector(10, (i) => i * 0.1);
 * ```
 */
export function createVector<N extends number>(
  dimensions: N,
  generator: (index: number) => number
): Vector<N> {
  const values: number[] = [];
  for (let i = 0; i < dimensions; i++) {
    values.push(generator(i));
  }
  return values as Vector<N>;
}

/**
 * Create a zero vector with the specified dimensions.
 *
 * @param dimensions - Number of dimensions
 * @returns Zero vector
 */
export function zeroVector<N extends number>(dimensions: N): Vector<N> {
  return createVector(dimensions, () => 0);
}

/**
 * Create a vector filled with random values between 0 and 1.
 *
 * @param dimensions - Number of dimensions
 * @returns Random vector
 */
export function randomVector<N extends number>(dimensions: N): Vector<N> {
  return createVector(dimensions, () => Math.random());
}

/**
 * Validate that a value is a valid vector with the expected dimensions.
 *
 * @param value - Value to validate
 * @param dimensions - Expected number of dimensions
 * @returns True if valid
 * @throws Error if invalid
 */
export function validateVector(
  value: unknown,
  dimensions?: number
): value is number[] {
  if (!Array.isArray(value)) {
    throw new Error('Vector must be an array');
  }

  if (!value.every((v) => typeof v === 'number' && !isNaN(v))) {
    throw new Error('Vector must contain only valid numbers');
  }

  if (dimensions !== undefined && value.length !== dimensions) {
    throw new Error(
      `Vector dimension mismatch: expected ${dimensions}, got ${value.length}`
    );
  }

  return true;
}

/**
 * Normalize a vector to unit length (L2 norm = 1).
 *
 * @param vector - Vector to normalize
 * @returns Normalized vector
 */
export function normalizeVector<N extends number>(vector: Vector<N>): Vector<N> {
  const magnitude = Math.sqrt(
    vector.reduce((sum, val) => sum + val * val, 0)
  );

  if (magnitude === 0) {
    return vector;
  }

  return vector.map((val) => val / magnitude) as Vector<N>;
}

/**
 * Calculate the L2 (Euclidean) distance between two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns L2 distance
 */
export function l2Distance(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimensions');
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Calculate the cosine similarity between two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimensions');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Calculate the cosine distance between two vectors.
 * Cosine distance = 1 - cosine similarity
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)
 */
export function cosineDistance(a: number[], b: number[]): number {
  return 1 - cosineSimilarity(a, b);
}

/**
 * Calculate the dot product of two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Dot product
 */
export function dotProduct(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimensions');
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Common embedding model dimensions for reference
 */
export const EmbeddingDimensions = {
  /** OpenAI text-embedding-ada-002 */
  OPENAI_ADA_002: 1536,
  /** OpenAI text-embedding-3-small */
  OPENAI_3_SMALL: 1536,
  /** OpenAI text-embedding-3-large */
  OPENAI_3_LARGE: 3072,
  /** Cohere embed-english-v3 */
  COHERE_V3: 1024,
  /** Sentence Transformers all-MiniLM-L6-v2 */
  MINILM_L6_V2: 384,
  /** Sentence Transformers all-mpnet-base-v2 */
  MPNET_BASE_V2: 768,
  /** BERT base */
  BERT_BASE: 768,
  /** BGE small */
  BGE_SMALL: 384,
  /** BGE base */
  BGE_BASE: 768,
  /** BGE large */
  BGE_LARGE: 1024,
  /** Jina embeddings v2 */
  JINA_V2: 768,
  /** Voyage AI */
  VOYAGE_2: 1024,
} as const;

export type EmbeddingDimensionName = keyof typeof EmbeddingDimensions;
