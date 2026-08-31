import { ExperimentsList } from "@/pages/experiments/ExperimentsList";
import { fetchModels } from "@/services/experimentsService";

export function ModelsPage() {
  return (
    <ExperimentsList
      title="Model Registry"
      description="One trained model per completed experiment — same underlying registry as Experiments, framed as model versions."
      basePath="/models"
      fetcher={fetchModels}
    />
  );
}
