import { useEffect, useState } from "react";
import { fetchBackendHealth } from "@/services/healthService";
import type { HealthResponse } from "@/types/health";

export type ConnectionState = "checking" | "online" | "offline";

interface UseBackendHealthResult {
  state: ConnectionState;
  health: HealthResponse | null;
}

export function useBackendHealth(): UseBackendHealthResult {
  const [state, setState] = useState<ConnectionState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetchBackendHealth()
      .then((result) => {
        if (cancelled) return;
        setHealth(result);
        setState("online");
      })
      .catch(() => {
        if (cancelled) return;
        setState("offline");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { state, health };
}
