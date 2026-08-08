import { createBrowserRouter } from "react-router-dom";
import { AppLayout } from "@/app/layouts/AppLayout";
import { Dashboard } from "@/pages/Dashboard";
import { PlaceholderPage } from "@/pages/PlaceholderPage";

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
        element: <PlaceholderPage title="Federated Overview" description="Current FL state and global model information." />,
      },
      {
        path: "federated/rounds",
        element: <PlaceholderPage title="Training Rounds" description="Individual FL round history and metrics." />,
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
        element: <PlaceholderPage title="Adaptive Control" description="Adaptive FL decisions." />,
      },

      {
        path: "drl/controller",
        element: <PlaceholderPage title="DRL Controller" description="DQN state, action, reward and current policy information." />,
      },
      {
        path: "drl/decisions",
        element: <PlaceholderPage title="Decision History" description="Historical DRL decisions." />,
      },

      {
        path: "models",
        element: <PlaceholderPage title="Model Registry" description="Available trained model versions." />,
      },
      {
        path: "models/:modelId",
        element: <PlaceholderPage title="Model Details" description="Performance and training information for a specific model." />,
      },

      {
        path: "experiments",
        element: <PlaceholderPage title="Experiments" description="Actual experiment runs." />,
      },
      {
        path: "experiments/analytics",
        element: <PlaceholderPage title="Analytics" description="Cross-system performance analysis." />,
      },
      {
        path: "experiments/:experimentId",
        element: <PlaceholderPage title="Experiment Details" description="Detailed metrics and configuration for an experiment." />,
      },

      {
        path: "system/health",
        element: <PlaceholderPage title="System Health" description="Backend, ML engine, FL engine, DRL engine and simulator health." />,
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
