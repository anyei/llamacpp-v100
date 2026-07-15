/**
 *
 * FLEET
 *
 * Components for monitoring the RPC worker fleet powering a distributed
 * inference pipeline. Polls `./fleet/status` while the Fleet page is mounted.
 *
 * The fleet system integrates with:
 * - `fleetStore` for status polling and throughput derivation
 * - `FleetService` for the `./fleet/*` server endpoints
 *
 */

/**
 * **FleetScreen** - Fleet monitoring page
 *
 * Full-page section showing the server state, model load progress, pipeline
 * devices (local and RPC), and workers discovered on the LAN.
 *
 * **Features:**
 * - Header with server state badge and model name/size
 * - Reload / endpoint failure warning banner
 * - Load progress bar while the model is loading
 * - Pipeline device cards grid with fastest/slowest ranking
 * - Discovered-on-LAN worker table with in-pipeline status
 * - Worker log slide-over panel
 */
export { default as FleetScreen } from './FleetScreen.svelte';

/**
 * **FleetDeviceCard** - Individual pipeline device card
 *
 * Shows device identity, RPC/local and CPU (RAM) badges, reachability status,
 * memory usage bar, split share, benchmark score, live throughput with a
 * sparkline, and Logs/Restart actions for RPC workers.
 */
export { default as FleetDeviceCard } from './FleetDeviceCard.svelte';

/**
 * **FleetWorkerLogs** - Worker log viewer sheet
 *
 * Slide-over panel that polls `./fleet/worker/log` for the selected worker.
 * Follows new output automatically; scrolling up pauses following.
 */
export { default as FleetWorkerLogs } from './FleetWorkerLogs.svelte';

/** Inline SVG sparkline used for per-worker throughput history. */
export { default as FleetSparkline } from './FleetSparkline.svelte';
