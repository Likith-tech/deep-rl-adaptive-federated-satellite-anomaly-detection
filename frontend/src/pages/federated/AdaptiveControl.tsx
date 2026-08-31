import { Link } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { useApiResource } from "@/hooks/useApiResource";
import { fetchAdaptiveControl } from "@/services/federatedService";

const SIGNAL_ORDER = ["performance", "data", "resource", "fairness"] as const;

export function AdaptiveControl() {
  const { state, data, error } = useApiResource(fetchAdaptiveControl);

  return (
    <div>
      <PageHeader
        title="Adaptive Control"
        description="Phase 8's deterministic (non-learned) client-scoring rule configs — real, fixed weights, not a runtime control panel."
      />
      {state === "loading" && <LoadingState />}
      {state === "error" && <EmptyState title="Could not load rule configs" description={error ?? undefined} />}
      {state === "ready" && data && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {data.map((rule) => (
            <Card key={rule.experiment_id} title={rule.name}>
              <div className="mb-3 flex flex-col gap-1.5">
                {SIGNAL_ORDER.map((signal) => (
                  <div key={signal} className="flex items-center justify-between text-xs">
                    <span className="capitalize text-slate-400">{signal}</span>
                    <span className="font-mono text-slate-200">{rule.weights[signal].toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <div className="flex items-center justify-between border-t border-base-700 pt-2 text-xs">
                <span className="text-slate-500">
                  F1 {rule.test_f1.toFixed(4)}
                  {rule.fairness_gap !== null && ` · fairness gap ${rule.fairness_gap.toFixed(4)}`}
                </span>
                <Link to={`/experiments/${rule.experiment_id}`} className="text-accent-400 hover:underline">
                  Details →
                </Link>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
