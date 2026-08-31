import { useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { MetricBar } from "@/components/ui/MetricBar";
import { useApiResource } from "@/hooks/useApiResource";
import { fetchFederatedOverview, fetchFederatedRounds } from "@/services/federatedService";

export function FederatedRounds() {
  const overview = useApiResource(fetchFederatedOverview);
  const [selected, setSelected] = useState<string | null>(null);
  const experimentId = selected ?? overview.data?.best_mechanism.id ?? "";

  const rounds = useApiResource(
    () => (experimentId ? fetchFederatedRounds(experimentId) : Promise.resolve([])),
    [experimentId],
  );

  return (
    <div>
      <PageHeader
        title="Training Rounds"
        description="Real, round-by-round validation history for one federated experiment."
        action={
          overview.state === "ready" && overview.data ? (
            <select
              className="rounded-md border border-base-700 bg-base-900 px-3 py-1.5 text-sm text-slate-200"
              value={experimentId}
              onChange={(e) => setSelected(e.target.value)}
            >
              {overview.data.mechanisms.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          ) : null
        }
      />

      {(overview.state === "loading" || (experimentId && rounds.state === "loading")) && <LoadingState />}
      {overview.state === "error" && <EmptyState title="Could not load experiments" description={overview.error ?? undefined} />}
      {experimentId && rounds.state === "error" && (
        <EmptyState title="Could not load round history" description={rounds.error ?? undefined} />
      )}

      {rounds.state === "ready" && rounds.data && rounds.data.length > 0 && (
        <Card title={`${rounds.data.length} rounds`}>
          <div className="flex flex-col gap-2">
            {rounds.data.map((r) => (
              <MetricBar
                key={r.round_number}
                label={`Round ${r.round_number}`}
                value={r.val_f1}
                max={1}
                displayValue={`F1 ${r.val_f1.toFixed(4)}`}
              />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
