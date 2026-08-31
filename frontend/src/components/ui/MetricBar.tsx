interface MetricBarProps {
  label: string;
  value: number;
  max: number;
  displayValue?: string;
  tone?: "accent" | "ok" | "warn" | "critical";
}

const TONE_FILL: Record<NonNullable<MetricBarProps["tone"]>, string> = {
  accent: "bg-accent-500",
  ok: "bg-status-ok",
  warn: "bg-status-warn",
  critical: "bg-status-critical",
};

/** A plain, dependency-free horizontal bar for comparing real metrics
 * (F1 scores, aggregation weights) — no charting library needed for
 * this project's small, fixed-size comparisons. */
export function MetricBar({ label, value, max, displayValue, tone = "accent" }: MetricBarProps) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="w-40 shrink-0 truncate text-xs text-slate-400" title={label}>
        {label}
      </span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-base-700">
        <div className={`h-full rounded-full ${TONE_FILL[tone]}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-16 shrink-0 text-right text-xs text-slate-300">{displayValue ?? value.toFixed(4)}</span>
    </div>
  );
}
