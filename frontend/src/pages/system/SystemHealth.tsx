import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { useBackendHealth } from "@/hooks/useBackendHealth";

export function SystemHealth() {
  const { state, health } = useBackendHealth();

  return (
    <div>
      <PageHeader title="System Health" description="Live status of the OrbitShield backend API." />
      <Card title="Backend API">
        <StatusBadge
          tone={state === "online" ? "ok" : state === "checking" ? "idle" : "critical"}
          label={
            state === "online"
              ? `Online${health ? ` · ${health.service} v${health.version}` : ""}`
              : state === "checking"
                ? "Checking..."
                : "Offline"
          }
        />
      </Card>
    </div>
  );
}
