import { describe, it, expect } from 'vitest';
import { sql, SQL } from 'drizzle-orm';
import {
  vec0Table,
  vecFloat,
  vecInteger,
  vecText,
  insertVec0,
  insertManyVec0,
  deleteVec0,
  updateVec0,
} from '../src/index.js';

describe('DML Helpers - Unit Tests', () => {
  describe('insertVec0', () => {
    it('should return an SQL object', () => {
      const itemsVec = vec0Table('items_vec', {
        embedding: vecFloat('embedding', 4),
      });

      const result = insertVec0(itemsVec, {
        embedding: [1, 0, 0, 0],
      });

      expect(result).toBeInstanceOf(SQL);
    });

    it('should accept multiple column types', () => {
      const itemsVec = vec0Table('items_vec', {
        id: vecInteger('id').primaryKey(),
        title: vecText('title'),
        embedding: vecFloat('embedding', 4),
      });

      const result = insertVec0(itemsVec, {
        id: 1,
        title: 'Test Item',
        embedding: [1, 0, 0, 0],
      });

      expect(result).toBeInstanceOf(SQL);
    });

    it('should accept partial values (skip undefined)', () => {
      const itemsVec = vec0Table('items_vec', {
        id: vecInteger('id').primaryKey(),
        embedding: vecFloat('embedding', 4),
      });

      // Only provide embedding, not id
      const result = insertVec0(itemsVec, {
        embedding: [1, 0, 0, 0],
      });

      expect(result).toBeInstanceOf(SQL);
    });

    it('should throw error when no values provided', () => {
      const itemsVec = vec0Table('items_vec', {
        embedding: vecFloat('embedding', 4),
      });

      expect(() => insertVec0(itemsVec, {})).toThrow('No values provided for insert');
    });
  });

  describe('insertManyVec0', () => {
    it('should return an SQL object for multiple rows', () => {
      const itemsVec = vec0Table('items_vec', {
        embedding: vecFloat('embedding', 4),
      });

      const result = insertManyVec0(itemsVec, [
        { embedding: [1, 0, 0, 0] },
        { embedding: [0, 1, 0, 0] },
        { embedding: [0, 0, 1, 0] },
      ]);

      expect(result).toBeInstanceOf(SQL);
    });

    it('should accept rows with multiple columns', () => {
      const itemsVec = vec0Table('items_vec', {
        id: vecInteger('id').primaryKey(),
        embedding: vecFloat('embedding', 4),
      });

      const result = insertManyVec0(itemsVec, [
        { id: 1, embedding: [1, 0, 0, 0] },
        { id: 2, embedding: [0, 1, 0, 0] },
      ]);

      expect(result).toBeInstanceOf(SQL);
    });

    it('should throw error when empty array provided', () => {
      const itemsVec = vec0Table('items_vec', {
        embedding: vecFloat('embedding', 4),
      });

      expect(() => insertManyVec0(itemsVec, [])).toThrow('No rows provided for insert');
    });
  });

  describe('deleteVec0', () => {
    it('should return an SQL object with rowid condition', () => {
      const itemsVec = vec0Table('items_vec', {
        embedding: vecFloat('embedding', 4),
      });

      const result = deleteVec0(itemsVec, 1);

      expect(result).toBeInstanceOf(SQL);
    });

    it('should return an SQL object with SQL condition', () => {
      const itemsVec = vec0Table('items_vec', {
        id: vecInteger('id').primaryKey(),
        embedding: vecFloat('embedding', 4),
      });

      const result = deleteVec0(itemsVec, sql`id = ${5}`);

      expect(result).toBeInstanceOf(SQL);
    });
  });

  describe('updateVec0', () => {
    it('should return an SQL object with rowid condition', () => {
      const itemsVec = vec0Table('items_vec', {
        embedding: vecFloat('embedding', 4),
      });

      const result = updateVec0(itemsVec, { embedding: [0, 0, 1, 0] }, 1);

      expect(result).toBeInstanceOf(SQL);
    });

    it('should return an SQL object with SQL condition', () => {
      const itemsVec = vec0Table('items_vec', {
        id: vecInteger('id').primaryKey(),
        embedding: vecFloat('embedding', 4),
      });

      const result = updateVec0(itemsVec, { embedding: [0, 0, 1, 0] }, sql`id = ${5}`);

      expect(result).toBeInstanceOf(SQL);
    });

    it('should accept multiple columns to update', () => {
      const itemsVec = vec0Table('items_vec', {
        id: vecInteger('id').primaryKey(),
        title: vecText('title'),
        embedding: vecFloat('embedding', 4),
      });

      const result = updateVec0(
        itemsVec,
        { title: 'New Title', embedding: [0, 0, 1, 0] },
        sql`id = ${5}`
      );

      expect(result).toBeInstanceOf(SQL);
    });

    it('should throw error when no values provided', () => {
      const itemsVec = vec0Table('items_vec', {
        embedding: vecFloat('embedding', 4),
      });

      expect(() => updateVec0(itemsVec, {}, 1)).toThrow('No values provided for update');
    });
  });
});
