import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { MetricBar } from "@/components/ui/MetricBar";
import { useApiResource } from "@/hooks/useApiResource";
import type { ExperimentDetail } from "@/types/experiments";

interface ExperimentDetailViewProps {
  id: string;
  fetcher: (id: string) => Promise<ExperimentDetail>;
}

export function ExperimentDetailView({ id, fetcher }: ExperimentDetailViewProps) {
  const { state, data, error } = useApiResource(() => fetcher(id), [id]);

  if (state === "loading") return <LoadingState />;
  if (state === "error" || !data) {
    return <EmptyState title="Could not load experiment" description={error ?? undefined} />;
  }

  return (
    <div>
      <PageHeader title={data.name} description={`${data.phase} · ${data.family} · source: ${data.source_report}`} />

      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card title="KDDTest+ F1">
          <p className="font-mono text-2xl text-slate-100">{data.test_f1.toFixed(4)}</p>
        </Card>
        <Card title="ROC-AUC">
          <p className="font-mono text-2xl text-slate-100">{data.test_roc_auc?.toFixed(4) ?? "not measured"}</p>
        </Card>
        <Card title="Client fairness gap">
          <p className="font-mono text-2xl text-slate-100">{data.fairness_gap?.toFixed(4) ?? "not measured"}</p>
        </Card>
      </div>

      {data.rounds.length > 0 ? (
        <Card title={`Validation F1 by round (${data.rounds.length} rounds)`}>
          <div className="flex flex-col gap-2">
            {data.rounds.map((r) => (
              <MetricBar
                key={r.round_number}
                label={`Round ${r.round_number}${r.action_name ? ` · ${r.action_name}` : ""}`}
                value={r.val_f1}
                max={1}
              />
            ))}
          </div>
        </Card>
      ) : (
        <EmptyState
          title="No round-by-round history"
          description="This experiment is centralized (non-federated) or a single-shot heterogeneity-sweep run, so no communication-round history applies."
        />
      )}
    </div>
  );
}
