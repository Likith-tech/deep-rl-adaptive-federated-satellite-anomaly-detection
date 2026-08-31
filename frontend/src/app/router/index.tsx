import { createBrowserRouter } from "react-router-dom";
import { AppLayout } from "@/app/layouts/AppLayout";
import { Dashboard } from "@/pages/Dashboard";
import { PlaceholderPage } from "@/pages/PlaceholderPage";
import { FederatedOverview } from "@/pages/federated/FederatedOverview";
import { FederatedRounds } from "@/pages/federated/FederatedRounds";
import { AdaptiveControl } from "@/pages/federated/AdaptiveControl";
import { DrlController } from "@/pages/drl/DrlController";
import { DrlDecisions } from "@/pages/drl/DrlDecisions";
import { ModelsPage } from "@/pages/experiments/ModelsPage";
import { ModelDetailPage } from "@/pages/experiments/ModelDetailPage";
import { ExperimentsPage } from "@/pages/experiments/ExperimentsPage";
import { ExperimentDetailPage } from "@/pages/experiments/ExperimentDetailPage";
import { ExperimentsAnalytics } from "@/pages/experiments/ExperimentsAnalytics";
import { SystemHealth } from "@/pages/system/SystemHealth";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <Dashboard /> },

      {
        path: "network/satellites",
        element: <PlaceholderPage title="Satellite Network" description="Topology, satellite status, links, connectivity and resources." />,
      },
      {
        path: "network/traffic",
        element: <PlaceholderPage title="Traffic" description="Network traffic analytics and time-series traffic information." />,
      },
      {
        path: "network/satellites/:satelliteId",
        element: <PlaceholderPage title="Satellite Details" description="Individual satellite state, resources, local model performance and FL participation." />,
      },

      {
        path: "security/threats",
        element: <PlaceholderPage title="Threats" description="Detected anomalies and threats." />,
      },
      {
        path: "security/threats/:threatId",
        element: <PlaceholderPage title="Threat Investigation" description="Detailed investigation of a specific anomaly." />,
      },

      {
        path: "federated/overview",
        element: <FederatedOverview />,
      },
      {
        path: "federated/rounds",
        element: <FederatedRounds />,
      },
      {
        path: "federated/clients",
        element: <PlaceholderPage title="Clients" description="Satellite FL client status and participation." />,
      },
      {
        path: "federated/aggregation",
        element: <PlaceholderPage title="Aggregation" description="Client contributions, aggregation weights, update quality and staleness." />,
      },
      {
        path: "federated/adaptive-control",
        element: <AdaptiveControl />,
      },

      {
        path: "drl/controller",
        element: <DrlController />,
      },
      {
        path: "drl/decisions",
        element: <DrlDecisions />,
      },

      {
        path: "models",
        element: <ModelsPage />,
      },
      {
        path: "models/:modelId",
        element: <ModelDetailPage />,
      },

      {
        path: "experiments",
        element: <ExperimentsPage />,
      },
      {
        path: "experiments/analytics",
        element: <ExperimentsAnalytics />,
      },
      {
        path: "experiments/:experimentId",
        element: <ExperimentDetailPage />,
      },

      {
        path: "system/health",
        element: <SystemHealth />,
      },
      {
        path: "system/activity",
        element: <PlaceholderPage title="Activity" description="System event stream." />,
      },
      {
        path: "system/settings",
        element: <PlaceholderPage title="Settings" description="Application and system configuration." />,
      },

      {
        path: "*",
        element: <PlaceholderPage title="Not Found" description="This page does not exist." />,
      },
    ],
  },
]);
