import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { EmptyState } from "@/components/ui/EmptyState";
import { useBackendHealth } from "@/hooks/useBackendHealth";

const SYSTEM_COMPONENTS = [
  { label: "Backend API", key: "backend" as const },
  { label: "Satellite Simulator", key: "simulator" as const },
  { label: "Federated Learning Engine", key: "federated" as const },
  { label: "DRL Controller", key: "drl" as const },
];

export function Dashboard() {
  const { state, health } = useBackendHealth();

  return (
    <div>
      <PageHeader
        title="Mission Overview"
        description="Overall status of the OrbitShield satellite security platform."
      />

      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {SYSTEM_COMPONENTS.map((component) => {
          const isBackend = component.key === "backend";
          const tone = isBackend ? (state === "online" ? "ok" : state === "checking" ? "idle" : "critical") : "idle";
          const label = isBackend
            ? state === "online"
              ? `Online${health ? ` · v${health.version}` : ""}`
              : state === "checking"
                ? "Checking..."
                : "Offline"
            : "Not yet implemented";

          return (
            <Card key={component.key} className="flex flex-col gap-3">
              <p className="text-xs uppercase tracking-wide text-slate-500">{component.label}</p>
              <StatusBadge tone={tone} label={label} />
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="Satellite Network">
          <EmptyState
            title="Awaiting live data"
            description="Satellite network simulation has not been implemented yet (Phase 6)."
          />
        </Card>
        <Card title="Recent Threats">
          <EmptyState
            title="Awaiting live data"
            description="Anomaly detection has not been implemented yet (Phases 3-5)."
          />
        </Card>
        <Card title="Federated Learning Status">
          <EmptyState
            title="Awaiting live data"
            description="Federated learning has not been implemented yet (Phases 8-11)."
          />
        </Card>
        <Card title="DRL Controller">
          <EmptyState
            title="Awaiting live data"
            description="The DQN controller has not been implemented yet (Phases 12-14)."
          />
        </Card>
      </div>
    </div>
  );
}
