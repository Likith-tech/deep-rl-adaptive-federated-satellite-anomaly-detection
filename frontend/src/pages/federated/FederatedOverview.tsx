import { Link } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { MetricBar } from "@/components/ui/MetricBar";
import { useApiResource } from "@/hooks/useApiResource";
import { fetchFederatedOverview } from "@/services/federatedService";

export function FederatedOverview() {
  const { state, data, error } = useApiResource(fetchFederatedOverview);

  if (state === "loading") return <LoadingState />;
  if (state === "error" || !data) {
    return <EmptyState title="Could not load federated overview" description={error ?? undefined} />;
  }

  return (
    <div>
      <PageHeader
        title="Federated Overview"
        description={`${data.total_methods_compared} methods compared across every phase (Phase 6-9B).`}
      />

      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Card title="Best aggregation mechanism">
          <p className="text-sm text-slate-300">{data.best_mechanism.name}</p>
          <p className="mt-2 font-mono text-2xl text-slate-100">{data.best_mechanism.test_f1.toFixed(4)}</p>
          <p className="mt-1 text-xs text-slate-500">KDDTest+ F1</p>
          <Link to={`/experiments/${data.best_mechanism.id}`} className="mt-3 inline-block text-xs text-accent-400 hover:underline">
            View details →
          </Link>
        </Card>
        <Card title="Fairest aggregation mechanism">
          <p className="text-sm text-slate-300">{data.fairest_mechanism.name}</p>
          <p className="mt-2 font-mono text-2xl text-slate-100">{data.fairest_mechanism.fairness_gap?.toFixed(4)}</p>
          <p className="mt-1 text-xs text-slate-500">Client fairness gap (lower is better)</p>
          <Link to={`/experiments/${data.fairest_mechanism.id}`} className="mt-3 inline-block text-xs text-accent-400 hover:underline">
            View details →
          </Link>
        </Card>
      </div>

      <Card title="Same partition, same model — aggregation mechanism only">
        <div className="flex flex-col gap-2">
          {[...data.mechanisms]
            .sort((a, b) => b.test_f1 - a.test_f1)
            .map((m) => (
              <MetricBar
                key={m.id}
                label={m.name.replace(/^Phase \d+B? — /, "")}
                value={m.test_f1}
                max={1}
                tone={m.id === data.best_mechanism.id ? "ok" : "accent"}
              />
            ))}
        </div>
      </Card>
    </div>
  );
}
