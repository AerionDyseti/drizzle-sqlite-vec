import Database from 'better-sqlite3';
import * as sqlite_vec from 'sqlite-vec';
import { drizzle } from 'drizzle-orm/better-sqlite3';

export interface TestContext {
  sqlite: Database.Database;
  db: ReturnType<typeof drizzle>;
}

export function createTestContext(): TestContext {
  const sqlite = new Database(':memory:');
  sqlite_vec.load(sqlite);
  const db = drizzle(sqlite);
  return { sqlite, db };
}

export function closeTestContext(ctx: TestContext): void {
  ctx.sqlite.close();
}
