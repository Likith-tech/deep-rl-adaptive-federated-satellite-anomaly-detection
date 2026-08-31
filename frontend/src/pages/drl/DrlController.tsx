import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { useApiResource } from "@/hooks/useApiResource";
import { fetchDrlController } from "@/services/drlService";

export function DrlController() {
  const { state, data, error } = useApiResource(fetchDrlController);

  if (state === "loading") return <LoadingState />;
  if (state === "error" || !data) {
    return <EmptyState title="Could not load DRL controller" description={error ?? undefined} />;
  }

  return (
    <div>
      <PageHeader
        title="DRL Controller"
        description="Phase 9B — a DQN that picks an aggregation-weight profile each federated round, trained via real pretraining rounds."
      />

      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-4">
        <Card title="State dimensions">
          <p className="font-mono text-2xl text-slate-100">{data.state_dimensions}</p>
        </Card>
        <Card title="Action catalogue size">
          <p className="font-mono text-2xl text-slate-100">{data.action_catalogue.length}</p>
        </Card>
        <Card title="Episodes trained">
          <p className="font-mono text-2xl text-slate-100">{data.episodes_trained}</p>
        </Card>
        <Card title="Total real FL rounds">
          <p className="font-mono text-2xl text-slate-100">{data.total_env_steps}</p>
        </Card>
      </div>

      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Card title="KDDTest+ result (frozen policy, one-time eval)">
          <p className="font-mono text-2xl text-slate-100">F1 {data.test_f1.toFixed(4)}</p>
          <p className="mt-1 text-xs text-slate-500">ROC-AUC {data.test_roc_auc.toFixed(4)}</p>
        </Card>
        <Card title="Beat Phase 8's best rule-based baseline?">
          <StatusBadge
            tone={data.beat_best_rule_based_baseline ? "ok" : "warn"}
            label={data.beat_best_rule_based_baseline ? "Yes" : "No — honest negative result"}
          />
          <p className="mt-2 text-xs text-slate-500">
            Reported as measured: a fixed rule-based config still beats the learned controller on this project's data.
          </p>
        </Card>
      </div>

      <Card title="Action catalogue (9 discrete weighting profiles)">
        <div className="flex flex-wrap gap-2">
          {data.action_catalogue.map((action) => (
            <span key={action} className="rounded-full border border-base-700 px-3 py-1 text-xs text-slate-300">
              {action}
            </span>
          ))}
        </div>
      </Card>
    </div>
  );
}
