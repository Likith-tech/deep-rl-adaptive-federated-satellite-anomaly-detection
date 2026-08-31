import { useParams } from "react-router-dom";
import { ExperimentDetailView } from "@/pages/experiments/ExperimentDetailView";
import { fetchExperiment } from "@/services/experimentsService";

export function ExperimentDetailPage() {
  const { experimentId } = useParams<{ experimentId: string }>();
  if (!experimentId) return null;
  return <ExperimentDetailView id={experimentId} fetcher={fetchExperiment} />;
}
