import { useBackendHealth } from "@/hooks/useBackendHealth";
import { StatusBadge } from "@/components/ui/StatusBadge";

interface TopNavProps {
  onToggleSidebar: () => void;
}

export function TopNav({ onToggleSidebar }: TopNavProps) {
  const { state } = useBackendHealth();

  return (
    <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-base-700 bg-base-900/80 px-4 backdrop-blur">
      <button
        type="button"
        onClick={onToggleSidebar}
        className="rounded-md p-2 text-slate-400 hover:bg-base-800 hover:text-slate-200 lg:hidden"
        aria-label="Toggle navigation"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M4 6h16M4 12h16M4 18h16" strokeLinecap="round" />
        </svg>
      </button>

      <div className="hidden text-sm text-slate-500 lg:block">
        Deep RL-Driven Adaptive Federated Framework for Satellite Network Anomaly Detection
      </div>

      <div className="flex items-center gap-3">
        {state === "checking" && <StatusBadge tone="idle" label="Connecting to backend" />}
        {state === "online" && <StatusBadge tone="ok" label="Backend online" />}
        {state === "offline" && <StatusBadge tone="critical" label="Backend offline" />}
      </div>
    </header>
  );
}
