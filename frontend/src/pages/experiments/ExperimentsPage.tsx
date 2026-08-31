import { ExperimentsList } from "@/pages/experiments/ExperimentsList";
import { fetchExperiments } from "@/services/experimentsService";

export function ExperimentsPage() {
  return (
    <ExperimentsList
      title="Experiments"
      description="Every real, already-completed ML/FL/DRL experiment this project has run (Phases 2-9), ranked by KDDTest+ F1."
      basePath="/experiments"
      fetcher={fetchExperiments}
    />
  );
}
