/**
 * Fleet monitoring types — responses of the `./fleet/*` endpoints.
 * Used by the Fleet section to render RPC pipeline devices and LAN discovery.
 */

/** Cumulative transfer statistics for an RPC device (may be absent). */
export interface ApiFleetDeviceStats {
	bytes_sent: number;
	bytes_recv: number;
	calls: number;
	ewma_latency_us: number;
}

/** Benchmark score of an RPC device (may be absent). */
export interface ApiFleetDeviceScore {
	bw_gbps: number;
	mm_gflops: number;
}

/** A device participating in the inference pipeline (local or RPC worker). */
export interface ApiFleetDevice {
	name: string;
	description: string;
	/** RPC endpoint (host:port). Null/absent for local (non-RPC) devices. */
	endpoint?: string | null;
	is_rpc: boolean;
	/** True when the worker exposes CPU RAM instead of GPU VRAM. */
	worker_is_cpu: boolean;
	reachable: boolean;
	failed: boolean;
	memory_free_mib: number;
	memory_total_mib: number;
	/** Share of the model split assigned to this device (0..1). */
	split_frac?: number | null;
	n_layers?: number | null;
	stats?: ApiFleetDeviceStats | null;
	score?: ApiFleetDeviceScore | null;
}

/** A worker announced on the LAN via discovery (may or may not be in the pipeline). */
export interface ApiFleetDiscoveredWorker {
	endpoint: string;
	payload: string;
	/** Parsed from the beacon line; null when the beacon omitted the token. */
	free_mib?: number | null;
	devs?: number | null;
	proto?: number | string | null;
	bw_gbps?: number | null;
	last_seen_ms: number;
	in_pipeline: boolean;
}

/** Model load progress — present only while `server_state` is `loading`. */
export interface ApiFleetLoadProgress {
	/** Progress 0..1 */
	value: number;
	stage: string;
}

/** Pipeline reload / recovery state. */
export interface ApiFleetReloadState {
	pending: boolean;
	any_endpoint_failed: boolean;
	auto_recovery: boolean;
}

export interface ApiFleetModelInfo {
	path: string;
	size_bytes: number;
}

export type ApiFleetServerState = 'loading' | 'reloading' | 'ready' | 'sleeping';

/** Response of `GET ./fleet/status`. Fields sourced from the model params
 * (model, split_mode, n_gpu_layers) are null until the initial load finishes. */
export interface ApiFleetStatusResponse {
	fleet_admin: boolean;
	server_state: ApiFleetServerState;
	load_progress?: ApiFleetLoadProgress | null;
	reload: ApiFleetReloadState;
	split_mode?: string | null;
	n_gpu_layers?: number | null;
	model?: ApiFleetModelInfo | null;
	devices: ApiFleetDevice[];
	discovered: ApiFleetDiscoveredWorker[];
}

/** Response of `GET ./fleet/worker/log`. */
export interface ApiFleetWorkerLogResponse {
	endpoint: string;
	log: string;
}

/** Response of `POST ./fleet/worker/restart`. */
export interface ApiFleetWorkerRestartResponse {
	success: boolean;
	recovery: 'auto' | 'manual';
}
