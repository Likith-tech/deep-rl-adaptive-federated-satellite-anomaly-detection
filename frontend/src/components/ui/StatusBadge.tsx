export type StatusTone = "ok" | "warn" | "critical" | "idle";

const TONE_CLASSES: Record<StatusTone, string> = {
  ok: "bg-status-ok/10 text-status-ok border-status-ok/30",
  warn: "bg-status-warn/10 text-status-warn border-status-warn/30",
  critical: "bg-status-critical/10 text-status-critical border-status-critical/30",
  idle: "bg-status-idle/10 text-status-idle border-status-idle/30",
};

interface StatusBadgeProps {
  tone: StatusTone;
  label: string;
}

export function StatusBadge({ tone, label }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${TONE_CLASSES[tone]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label}
    </span>
  );
}
