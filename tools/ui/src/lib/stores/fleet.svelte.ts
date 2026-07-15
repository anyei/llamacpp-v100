import { FleetService } from '$lib/services/fleet.service';

/** Poll interval for `./fleet/status` while the Fleet page is mounted. */
export const FLEET_POLL_INTERVAL_MS = 2000;

/** Number of throughput samples kept per endpoint for the sparkline. */
export const FLEET_RATE_HISTORY_LENGTH = 60;

const BYTES_PER_MB = 1024 * 1024;

/** Derived throughput of a single RPC endpoint. */
export interface FleetEndpointRates {
	/** MB/s received from the worker (bytes_recv delta). */
	inMbps: number;
	/** MB/s sent to the worker (bytes_sent delta). */
	outMbps: number;
	/** Total (in + out) MB/s history, oldest first, most recent last. */
	history: number[];
}

interface FleetCounterSample {
	bytesSent: number;
	bytesRecv: number;
	timeMs: number;
}

/**
 * fleetStore - RPC fleet monitoring state
 *
 * Polls `./fleet/status` every {@link FLEET_POLL_INTERVAL_MS} while the Fleet
 * page is mounted (started/stopped by the page component) and derives per
 * endpoint throughput (MB/s) from the cumulative byte counters between polls.
 *
 * **Architecture & Relationships:**
 * - **FleetService**: Stateless service for the `./fleet/*` endpoints
 * - **fleetStore** (this class): Reactive store consumed by the Fleet components
 */
class FleetStore {
	/**
	 *
	 *
	 * State
	 *
	 *
	 */

	status = $state<ApiFleetStatusResponse | null>(null);
	error = $state<string | null>(null);
	lastUpdatedMs = $state<number | null>(null);
	rates = $state<Record<string, FleetEndpointRates>>({});

	private pollTimer: ReturnType<typeof setInterval> | null = null;
	private inFlight = false;
	private previousSamples = new Map<string, FleetCounterSample>();

	/**
	 *
	 *
	 * Getters
	 *
	 *
	 */

	get isPolling(): boolean {
		return this.pollTimer !== null;
	}

	get isLoading(): boolean {
		return this.status?.server_state === 'loading';
	}

	getRates(endpoint: string | null | undefined): FleetEndpointRates | undefined {
		if (!endpoint) return undefined;

		return this.rates[endpoint];
	}

	/**
	 *
	 *
	 * Polling
	 *
	 *
	 */

	startPolling(): void {
		if (this.pollTimer) return;

		void this.poll();
		this.pollTimer = setInterval(() => void this.poll(), FLEET_POLL_INTERVAL_MS);
	}

	stopPolling(): void {
		if (this.pollTimer) {
			clearInterval(this.pollTimer);
			this.pollTimer = null;
		}

		this.inFlight = false;
		this.previousSamples.clear();
	}

	private async poll(): Promise<void> {
		if (this.inFlight) return;

		this.inFlight = true;

		try {
			const status = await FleetService.fetchStatus();
			const nowMs = Date.now();

			this.updateRates(status, nowMs);
			this.status = status;
			this.lastUpdatedMs = nowMs;
			this.error = null;
		} catch (error: unknown) {
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.inFlight = false;
		}
	}

	/**
	 *
	 *
	 * Utilities
	 *
	 *
	 */

	private updateRates(status: ApiFleetStatusResponse, nowMs: number): void {
		const nextRates: Record<string, FleetEndpointRates> = {};

		for (const device of status.devices ?? []) {
			if (!device.endpoint || !device.stats) continue;

			const previousRate = this.rates[device.endpoint];
			const previousSample = this.previousSamples.get(device.endpoint);
			const sample: FleetCounterSample = {
				bytesSent: device.stats.bytes_sent,
				bytesRecv: device.stats.bytes_recv,
				timeMs: nowMs
			};

			this.previousSamples.set(device.endpoint, sample);

			if (!previousSample || sample.timeMs <= previousSample.timeMs) {
				nextRates[device.endpoint] = previousRate ?? { inMbps: 0, outMbps: 0, history: [] };
				continue;
			}

			const elapsedSeconds = (sample.timeMs - previousSample.timeMs) / 1000;
			// Counters reset when a worker restarts — treat negative deltas as zero
			const sentDelta = Math.max(0, sample.bytesSent - previousSample.bytesSent);
			const recvDelta = Math.max(0, sample.bytesRecv - previousSample.bytesRecv);
			const outMbps = sentDelta / BYTES_PER_MB / elapsedSeconds;
			const inMbps = recvDelta / BYTES_PER_MB / elapsedSeconds;

			const history = [...(previousRate?.history ?? []), inMbps + outMbps].slice(
				-FLEET_RATE_HISTORY_LENGTH
			);

			nextRates[device.endpoint] = { inMbps, outMbps, history };
		}

		// Drop samples of endpoints that left the pipeline
		for (const endpoint of [...this.previousSamples.keys()]) {
			if (!(endpoint in nextRates)) {
				this.previousSamples.delete(endpoint);
			}
		}

		this.rates = nextRates;
	}
}

export const fleetStore = new FleetStore();

export const fleetStatus = () => fleetStore.status;
export const fleetError = () => fleetStore.error;
export const fleetRates = () => fleetStore.rates;
