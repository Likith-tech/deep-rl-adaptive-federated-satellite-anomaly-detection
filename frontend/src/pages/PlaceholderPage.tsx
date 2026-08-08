import { PageHeader } from "@/components/layout/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";

interface PlaceholderPageProps {
  title: string;
  description: string;
}

/**
 * Renders the route shell for a functional area whose backing system
 * (federated engine, DRL controller, simulator, etc.) hasn't been built
 * yet. Replaced page-by-page as each phase lands — never filled with
 * fabricated data in the meantime.
 */
export function PlaceholderPage({ title, description }: PlaceholderPageProps) {
  return (
    <div>
      <PageHeader title={title} description={description} />
      <EmptyState
        title="Awaiting live data"
        description="This module is scaffolded but not yet connected to a running backend system. It will populate once the corresponding phase is implemented."
      />
    </div>
  );
}
