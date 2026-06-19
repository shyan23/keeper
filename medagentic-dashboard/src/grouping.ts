import { ApiDocument } from './types';

export interface CategoryGroup { category: string; docs: ApiDocument[]; }
export interface YearGroup { year: string; total: number; categories: CategoryGroup[]; }

// Year of report date (fallback upload date). Undated -> "Undated".
function yearOf(d: ApiDocument): string {
  if (!d.date) return 'Undated';
  const y = new Date(d.date).getFullYear();
  return Number.isNaN(y) ? 'Undated' : String(y);
}

function categoryOf(d: ApiDocument): string {
  return (d.category && d.category.trim())
    || (d.type && d.type.trim())
    || 'Uncategorized';
}

// Group documents into years, each year into categories. `desc` => newest year
// first. "Undated" always sorts last. Docs within a category keep input order.
export function groupDocsByYear(docs: ApiDocument[], desc: boolean): YearGroup[] {
  const years = new Map<string, Map<string, ApiDocument[]>>();
  for (const d of docs) {
    const y = yearOf(d);
    const cat = categoryOf(d);
    const byCat = years.get(y) ?? years.set(y, new Map()).get(y)!;
    (byCat.get(cat) ?? byCat.set(cat, []).get(cat)!).push(d);
  }
  const out: YearGroup[] = [...years.entries()].map(([year, byCat]) => {
    const categories: CategoryGroup[] = [...byCat.entries()]
      .map(([category, ds]) => ({ category, docs: ds }))
      .sort((a, b) => a.category.localeCompare(b.category));
    const total = categories.reduce((n, c) => n + c.docs.length, 0);
    return { year, total, categories };
  });
  out.sort((a, b) => {
    if (a.year === 'Undated') return 1;
    if (b.year === 'Undated') return -1;
    return desc ? b.year.localeCompare(a.year) : a.year.localeCompare(b.year);
  });
  return out;
}
