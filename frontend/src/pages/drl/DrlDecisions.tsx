import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { useApiResource } from "@/hooks/useApiResource";
import { fetchDrlDecisions } from "@/services/drlService";

export function DrlDecisions() {
  const { state, data, error } = useApiResource(fetchDrlDecisions);

  return (
    <div>
      <PageHeader
        title="Decision History"
        description="The DQN's frozen-policy final run — real, one action chosen per round, never re-run for this view."
      />
      {state === "loading" && <LoadingState />}
      {state === "error" && <EmptyState title="Could not load decision history" description={error ?? undefined} />}
      {state === "ready" && data && (
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-base-700 text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-4">Round</th>
                  <th className="py-2 pr-4">Action chosen</th>
                  <th className="py-2 pr-4 text-right">Val F1</th>
                  <th className="py-2 text-right">Val loss</th>
                </tr>
              </thead>
              <tbody>
                {data.map((d) => (
                  <tr key={d.round_number} className="border-b border-base-800 last:border-0">
                    <td className="py-2 pr-4 text-slate-400">{d.round_number}</td>
                    <td className="py-2 pr-4">
                      <span className="rounded-full border border-base-700 px-2 py-0.5 text-xs text-slate-300">
                        {d.action_name}
                      </span>
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-slate-200">{d.val_f1.toFixed(4)}</td>
                    <td className="py-2 text-right font-mono text-slate-400">{d.val_loss.toFixed(4)}</td>
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
