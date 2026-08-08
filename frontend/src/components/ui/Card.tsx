import type { PropsWithChildren, ReactNode } from "react";

interface CardProps {
  title?: string;
  action?: ReactNode;
  className?: string;
}

export function Card({ title, action, className = "", children }: PropsWithChildren<CardProps>) {
  return (
    <section className={`rounded-lg border border-base-700 bg-base-900 ${className}`}>
      {(title || action) && (
        <header className="flex items-center justify-between border-b border-base-700 px-4 py-3">
          {title && <h3 className="text-sm font-medium text-slate-300">{title}</h3>}
          {action}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}
