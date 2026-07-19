import { API_FLEET } from '$lib/constants/api-endpoints';
import { apiFetch, apiFetchWithParams, apiPost } from '$lib/utils/api-fetch';

/** Default maximum number of log bytes requested from a worker. */
const DEFAULT_LOG_MAX_BYTES = 65536;

/**
 * FleetService - Stateless client for the `./fleet/*` server endpoints.
 *
 * Used by `fleetStore` to poll pipeline status and by the Fleet UI to
 * fetch worker logs and trigger worker restarts.
 *
 * Note: `./fleet/status` is exempt from the loading 503, so it can be
 * polled to render model load progress while the server is loading.
 */
export class FleetService {
	/**
	 * Fetches the fleet status (server state, pipeline devices, LAN discovery).
	 *
	 * @returns Fleet status snapshot
	 * @throws {Error} If the request fails
	 */
	static async fetchStatus(): Promise<ApiFleetStatusResponse> {
		return apiFetch<ApiFleetStatusResponse>(API_FLEET.STATUS, { authOnly: true });
	}

	/**
	 * Fetches the tail of a worker's log.
	 *
	 * @param endpoint - Worker endpoint (host:port)
	 * @param max - Maximum number of bytes to fetch
	 * @returns Worker log text
	 * @throws {Error} If the request fails or the endpoint is unknown
	 */
	static async fetchWorkerLog(
		endpoint: string,
		max: number = DEFAULT_LOG_MAX_BYTES
	): Promise<ApiFleetWorkerLogResponse> {
		return apiFetchWithParams<ApiFleetWorkerLogResponse>(
			API_FLEET.WORKER_LOG,
			{ ep: endpoint, max: String(max) },
			{ authOnly: true }
		);
	}

	/**
	 * Restarts a worker. Only allowed when the server runs with `--fleet-admin`
	 * (`fleet_admin: true` in the status response).
	 *
	 * @param endpoint - Worker endpoint (host:port)
	 * @returns Restart result including the recovery mode
	 * @throws {Error} If the request fails or fleet admin is disabled
	 */
	static async restartWorker(endpoint: string): Promise<ApiFleetWorkerRestartResponse> {
		return apiPost<ApiFleetWorkerRestartResponse>(API_FLEET.WORKER_RESTART, { endpoint });
	}

	/**
	 * Re-runs the ~0.5s bandwidth benchmark on a worker (proto 4.9). A busy or
	 * older worker reports success=false. Shares apply at the next (re)load.
	 */
	static async rescoreWorker(
		endpoint: string
	): Promise<{ success: boolean; bw_gbps?: number; mm_gflops?: number; reason?: string }> {
		return apiPost(API_FLEET.WORKER_RESCORE, { endpoint });
	}

	/**
	 * Reloads the coordinator to enlist newly discovered workers. The server
	 * exits and its restart policy brings it back with rediscovery, so all
	 * in-flight requests are dropped and the model reloads across the grown
	 * fleet. Only allowed when the server runs with `--fleet-admin`.
	 *
	 * @param endpoint - The worker endpoint being enlisted (for the server log)
	 * @returns Reload acknowledgement
	 * @throws {Error} If the request fails or fleet admin is disabled
	 */
	static async reloadFleet(endpoint?: string): Promise<ApiFleetReloadResponse> {
		return apiPost<ApiFleetReloadResponse>(API_FLEET.RELOAD, endpoint ? { endpoint } : {});
	}
}
