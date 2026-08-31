import { useParams } from "react-router-dom";
import { ExperimentDetailView } from "@/pages/experiments/ExperimentDetailView";
import { fetchModel } from "@/services/experimentsService";

export function ModelDetailPage() {
  const { modelId } = useParams<{ modelId: string }>();
  if (!modelId) return null;
  return <ExperimentDetailView id={modelId} fetcher={fetchModel} />;
}
