interface EmptyStateProps {
  title: string;
  description?: string;
}

/** Used wherever a page has no real backend data yet — never fabricate values here. */
export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-base-600 py-16 text-center">
      <p className="text-sm font-medium text-slate-400">{title}</p>
      {description && <p className="mt-1 max-w-md text-xs text-slate-500">{description}</p>}
    </div>
  );
}
