import { describe, it, expect } from 'vitest';
import {
  typedVector,
  createVector,
  zeroVector,
  randomVector,
  normalizeVector,
  l2Distance,
  cosineSimilarity,
  cosineDistance,
  dotProduct,
  EmbeddingDimensions,
} from '../src/index.js';

describe('Type utilities', () => {
  it('should create typed vectors', () => {
    const vec = typedVector(4, [1, 2, 3, 4]);
    expect(vec).toHaveLength(4);
    expect(vec).toEqual([1, 2, 3, 4]);
  });

  it('should throw on dimension mismatch', () => {
    expect(() => typedVector(4, [1, 2, 3])).toThrow('Vector dimension mismatch');
  });

  it('should create vectors with generator', () => {
    const zeros = zeroVector(4);
    expect(zeros).toEqual([0, 0, 0, 0]);

    const indexed = createVector(4, (i) => i);
    expect(indexed).toEqual([0, 1, 2, 3]);
  });

  it('should handle empty vectors', () => {
    const emptyZero = zeroVector(0);
    expect(emptyZero).toEqual([]);
    expect(emptyZero).toHaveLength(0);

    const emptyCreate = createVector(0, (i) => i);
    expect(emptyCreate).toEqual([]);
    expect(emptyCreate).toHaveLength(0);

    const emptyRandom = randomVector(0);
    expect(emptyRandom).toEqual([]);
    expect(emptyRandom).toHaveLength(0);
  });

  it('should create random vectors', () => {
    const random = randomVector(100);
    expect(random).toHaveLength(100);
    expect(random.every((v) => v >= 0 && v < 1)).toBe(true);
  });

  it('should normalize vectors', () => {
    const vec = [3, 4, 0, 0]; // magnitude = 5
    const normalized = normalizeVector(vec);

    expect(normalized).toHaveLength(4);
    // Verify unit norm
    const norm = Math.sqrt(normalized.reduce((sum, v) => sum + v * v, 0));
    expect(norm).toBeCloseTo(1, 10);
    // Verify direction is preserved (3/5 = 0.6, 4/5 = 0.8)
    expect(normalized[0]).toBeCloseTo(0.6, 10);
    expect(normalized[1]).toBeCloseTo(0.8, 10);
    expect(normalized[2]).toBeCloseTo(0, 10);
    expect(normalized[3]).toBeCloseTo(0, 10);
  });

  it('should handle zero vector normalization', () => {
    const zeroVec = [0, 0, 0, 0];
    const normalized = normalizeVector(zeroVec);
    expect(normalized).toEqual([0, 0, 0, 0]);
  });
});

describe('Math utilities', () => {
  it('should calculate L2 distance', () => {
    const distance = l2Distance([0, 0], [3, 4]);
    expect(distance).toBe(5);
  });

  it('should calculate L2 distance for higher dimensions', () => {
    const a = [1, 2, 3, 4];
    const b = [5, 6, 7, 8];
    expect(l2Distance(a, b)).toBe(8);
  });

  it('should calculate cosine similarity', () => {
    expect(cosineSimilarity([1, 0], [2, 0])).toBeCloseTo(1, 10);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 10);
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1, 10);
  });

  it('should calculate cosine distance', () => {
    expect(cosineDistance([1, 0], [1, 0])).toBeCloseTo(0, 10);
    expect(cosineDistance([1, 0], [0, 1])).toBeCloseTo(1, 10);
    expect(cosineDistance([1, 0], [-1, 0])).toBeCloseTo(2, 10);
  });

  it('should calculate dot product', () => {
    expect(dotProduct([1, 2], [3, 4])).toBe(11);
    expect(dotProduct([1, 0, 0], [0, 1, 0])).toBe(0);
    expect(dotProduct([2, 3, 4], [5, 6, 7])).toBe(56);
  });

  it('should handle zero vectors in similarity calculations', () => {
    expect(cosineSimilarity([0, 0], [1, 0])).toBe(0);
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
  });
});

describe('Embedding Dimensions Constants', () => {
  it('should have correct OpenAI dimensions', () => {
    expect(EmbeddingDimensions.OPENAI_ADA_002).toBe(1536);
    expect(EmbeddingDimensions.OPENAI_3_SMALL).toBe(1536);
    expect(EmbeddingDimensions.OPENAI_3_LARGE).toBe(3072);
  });

  it('should have correct Cohere dimensions', () => {
    expect(EmbeddingDimensions.COHERE_V3).toBe(1024);
  });

  it('should have correct Sentence Transformers dimensions', () => {
    expect(EmbeddingDimensions.MINILM_L6_V2).toBe(384);
    expect(EmbeddingDimensions.MPNET_BASE_V2).toBe(768);
  });

  it('should have correct BGE dimensions', () => {
    expect(EmbeddingDimensions.BGE_SMALL).toBe(384);
    expect(EmbeddingDimensions.BGE_BASE).toBe(768);
    expect(EmbeddingDimensions.BGE_LARGE).toBe(1024);
  });
});
