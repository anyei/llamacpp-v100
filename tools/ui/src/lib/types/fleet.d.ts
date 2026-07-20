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

/** Worker-side handler timing of the busiest graph command, parsed from the
 * worker's [rpc-timing] log dumps. Present only when the worker runs with
 * GGML_RPC_TIMING. */
export interface ApiFleetDeviceTiming {
	cmd: string;
	n: number;
	lock_avg_us: number;
	exec_avg_us: number;
	exec_max_us: number;
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
	health?: 'healthy' | 'degraded' | 'recovering';
	failure_count?: number | null;
	memory_free_mib: number;
	memory_total_mib: number;
	/** Share of the model split assigned to this device (0..1). */
	split_frac?: number | null;
	/** EP dedicated-attention owner: holds attention/KV/router, takes no expert share. */
	attn_owner?: boolean;
	n_layers?: number | null;
	stats?: ApiFleetDeviceStats | null;
	score?: ApiFleetDeviceScore | null;
	timing?: ApiFleetDeviceTiming | null;
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
	cache_mib?: number | null;
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

/** Result of the opt-in --fleet-preflight benchmark (null when disabled or not yet run).
 * The tps of a small dense model is the fleet's per-token boundary/latency floor. */
export interface ApiFleetPreflight {
	tps: number;
	model: string;
	n_tokens: number;
	load_s: number;
}

/** Rolling token-weighted average decode speed over the last `window_n`
 * completed generations (null until a request finishes). */
export interface ApiFleetPerf {
	tg_avg_tps: number;
	window_n: number;
}

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
	recent_failures?: number;
	discovered: ApiFleetDiscoveredWorker[];
	capacity?: {
		waiting: boolean;
		required_mib: number;
		available_mib: number;
		auto_recover?: boolean;
	} | null;
	preflight?: ApiFleetPreflight | null;
	perf?: ApiFleetPerf | null;
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

/** Response of `POST ./fleet/reload`. */
export interface ApiFleetReloadResponse {
	success: boolean;
}
