import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { MetricBar } from "@/components/ui/MetricBar";
import { useApiResource } from "@/hooks/useApiResource";
import { fetchExperiments } from "@/services/experimentsService";

/** Cross-method analytics — the same real, already-committed ranking
 * Phase 14's full-evaluation report established, rendered as one
 * ranked chart rather than a report table. No new metric is computed
 * here; it's the /experiments list, sorted and visualized. */
export function ExperimentsAnalytics() {
  const { state, data, error } = useApiResource(fetchExperiments);

  return (
    <div>
      <PageHeader
        title="Cross-Experiment Analytics"
        description="Every method this project has built, ranked by KDDTest+ F1 (Phase 14's full cross-phase evaluation)."
      />
      {state === "loading" && <LoadingState />}
      {state === "error" && <EmptyState title="Could not load data" description={error ?? undefined} />}
      {state === "ready" && data && (
        <Card title={`${data.length} methods compared`}>
          <div className="flex flex-col gap-2">
            {[...data]
              .sort((a, b) => b.test_f1 - a.test_f1)
              .map((row) => (
                <MetricBar key={row.id} label={row.name} value={row.test_f1} max={1} />
              ))}
          </div>
        </Card>
      )}
    </div>
  );
}
