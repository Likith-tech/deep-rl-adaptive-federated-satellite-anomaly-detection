export interface NavItem {
  label: string;
  path: string;
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

export const NAV_SECTIONS: NavSection[] = [
  {
    title: "Main",
    items: [{ label: "Dashboard", path: "/" }],
  },
  {
    title: "Network",
    items: [
      { label: "Satellite Network", path: "/network/satellites" },
      { label: "Traffic", path: "/network/traffic" },
      { label: "Satellite Details", path: "/network/satellites/:satelliteId" },
    ],
  },
  {
    title: "Security",
    items: [
      { label: "Threats", path: "/security/threats" },
      { label: "Threat Investigation", path: "/security/threats/:threatId" },
    ],
  },
  {
    title: "Federated AI",
    items: [
      { label: "Federated Overview", path: "/federated/overview" },
      { label: "Training Rounds", path: "/federated/rounds" },
      { label: "Clients", path: "/federated/clients" },
      { label: "Aggregation", path: "/federated/aggregation" },
      { label: "Adaptive Control", path: "/federated/adaptive-control" },
    ],
  },
  {
    title: "DRL",
    items: [
      { label: "DRL Controller", path: "/drl/controller" },
      { label: "Decision History", path: "/drl/decisions" },
    ],
  },
  {
    title: "Models",
    items: [
      { label: "Model Registry", path: "/models" },
      { label: "Model Details", path: "/models/:modelId" },
    ],
  },
  {
    title: "Experiments",
    items: [
      { label: "Experiments", path: "/experiments" },
      { label: "Experiment Details", path: "/experiments/:experimentId" },
      { label: "Analytics", path: "/experiments/analytics" },
    ],
  },
  {
    title: "System",
    items: [
      { label: "System Health", path: "/system/health" },
      { label: "Activity", path: "/system/activity" },
      { label: "Settings", path: "/system/settings" },
    ],
  },
];

/** Nav items that link to a concrete route (excludes parameterized detail-page templates). */
export const SIDEBAR_LINKABLE_SECTIONS = NAV_SECTIONS.map((section) => ({
  ...section,
  items: section.items.filter((item) => !item.path.includes(":")),
}));
