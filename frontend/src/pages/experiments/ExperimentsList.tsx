import { Link } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { useApiResource } from "@/hooks/useApiResource";
import type { ExperimentSummary } from "@/types/experiments";

interface ExperimentsListProps {
  title: string;
  description: string;
  basePath: "/experiments" | "/models";
  fetcher: () => Promise<ExperimentSummary[]>;
}

/** Shared list view for the Experiments and Model Registry pages —
 * both read the same real registry of 14 already-completed
 * experiments (Phases 2-9), just framed differently. */
export function ExperimentsList({ title, description, basePath, fetcher }: ExperimentsListProps) {
  const { state, data, error } = useApiResource(fetcher);

  return (
    <div>
      <PageHeader title={title} description={description} />
      {state === "loading" && <LoadingState />}
      {state === "error" && <EmptyState title="Could not load data" description={error ?? undefined} />}
      {state === "ready" && data && (
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-base-700 text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-4">Name</th>
                  <th className="py-2 pr-4">Phase</th>
                  <th className="py-2 pr-4">Family</th>
                  <th className="py-2 pr-4 text-right">KDDTest+ F1</th>
                  <th className="py-2 pr-4 text-right">ROC-AUC</th>
                  <th className="py-2 text-right">Fairness gap</th>
                </tr>
              </thead>
              <tbody>
                {[...data]
                  .sort((a, b) => b.test_f1 - a.test_f1)
                  .map((row) => (
                    <tr key={row.id} className="border-b border-base-800 last:border-0">
                      <td className="py-2 pr-4">
                        <Link to={`${basePath}/${row.id}`} className="text-accent-400 hover:underline">
                          {row.name}
                        </Link>
                      </td>
                      <td className="py-2 pr-4 text-slate-400">{row.phase}</td>
                      <td className="py-2 pr-4 text-slate-500">{row.family}</td>
                      <td className="py-2 pr-4 text-right font-mono text-slate-200">{row.test_f1.toFixed(4)}</td>
                      <td className="py-2 pr-4 text-right font-mono text-slate-400">
                        {row.test_roc_auc !== null ? row.test_roc_auc.toFixed(4) : "—"}
                      </td>
                      <td className="py-2 text-right font-mono text-slate-400">
                        {row.fairness_gap !== null ? row.fairness_gap.toFixed(4) : "—"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
