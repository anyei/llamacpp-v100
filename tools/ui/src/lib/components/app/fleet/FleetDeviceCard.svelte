<script lang="ts">
	import { Cpu, Gpu, RotateCcw, ScrollText, Gauge } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { DialogConfirmation } from '$lib/components/app/dialogs';
	import { FleetService } from '$lib/services/fleet.service';
	import type { FleetEndpointRates } from '$lib/stores/fleet.svelte';
	import { FLEET_RATE_HISTORY_LENGTH } from '$lib/stores/fleet.svelte';
	import FleetSparkline from './FleetSparkline.svelte';

	interface Props {
		device: ApiFleetDevice;
		fleetAdmin: boolean;
		rates?: FleetEndpointRates;
		rank?: 'fastest' | 'slowest' | null;
		/** True when this worker has the highest exec-avg in the fleet (long pole). */
		execSlowest?: boolean;
		/** 1-based position among devices sharing this endpoint, when > 1 total. */
		siblingIndex?: number | null;
		/** How many devices this worker (endpoint) exposes. */
		siblingCount?: number | null;
		onShowLogs?: (endpoint: string) => void;
	}

	let {
		device,
		fleetAdmin,
		rates,
		rank = null,
		execSlowest = false,
		siblingIndex = null,
		siblingCount = null,
		onShowLogs
	}: Props = $props();

	// worker-CPU devices masquerade as GPUs to the backend; the flag is the truth
	let KindIcon = $derived(device.worker_is_cpu ? Cpu : Gpu);
	let kindLabel = $derived(device.worker_is_cpu ? 'CPU (worker RAM)' : 'GPU');

	let showRestartDialog = $state(false);
	let isRescoring = $state(false);

	async function handleRescore() {
		if (!device.endpoint) return;
		isRescoring = true;
		try {
			const r = await FleetService.rescoreWorker(device.endpoint);
			if (r.success) {
				toast.success(
					`${device.endpoint}: ${r.bw_gbps?.toFixed(1)} GB/s — shares apply at the next reload`
				);
			} else {
				toast.error(r.reason ?? 'rescore failed');
			}
		} catch (e: unknown) {
			toast.error(e instanceof Error ? e.message : String(e));
		} finally {
			isRescoring = false;
		}
	}
	let isRestarting = $state(false);

	let memoryUsedMib = $derived(Math.max(0, device.memory_total_mib - device.memory_free_mib));
	let memoryUsedPercent = $derived(
		device.memory_total_mib > 0 ? (memoryUsedMib / device.memory_total_mib) * 100 : 0
	);

	let splitPercent = $derived(
		typeof device.split_frac === 'number' ? Math.round(device.split_frac * 100) : null
	);

	let latencyMs = $derived(device.stats ? (device.stats.ewma_latency_us / 1000).toFixed(1) : null);

	function statusColor(): string {
		if (device.failed) return 'bg-red-500';
		if (!device.reachable) return 'bg-gray-500';

		return 'bg-green-500';
	}

	function statusText(): string {
		if (device.failed) return 'FAILED';
		if (!device.reachable) return 'unreachable';

		return 'reachable';
	}

	function handleShowLogs() {
		if (device.endpoint) {
			onShowLogs?.(device.endpoint);
		}
	}

	async function handleRestartConfirm() {
		if (!device.endpoint) return;

		showRestartDialog = false;
		isRestarting = true;

		try {
			const result = await FleetService.restartWorker(device.endpoint);

			if (result.success) {
				toast.success(`Restart requested for ${device.endpoint} (${result.recovery} recovery)`);
			} else {
				toast.error(`Restart failed for ${device.endpoint}`);
			}
		} catch (error: unknown) {
			toast.error(error instanceof Error ? error.message : String(error));
		} finally {
			isRestarting = false;
		}
	}
</script>

<Card.Root
	class="!gap-3 p-4 {device.worker_is_cpu
		? 'border-dashed bg-muted/10'
		: 'bg-muted/30'} {device.failed ? 'border-destructive/50' : ''}"
>
	<div class="flex items-start justify-between gap-2">
		<div class="min-w-0">
			<div class="flex items-center gap-2">
				<KindIcon class="h-4 w-4 shrink-0 text-muted-foreground" aria-label={kindLabel} />

				<h3 class="truncate text-sm font-semibold" title={kindLabel}>{device.name}</h3>

				<div class="flex items-center gap-1.5" title={statusText()}>
					<div class="h-2 w-2 shrink-0 rounded-full {statusColor()}"></div>

					{#if device.failed}
						<Badge variant="destructive" class="text-[10px]">FAILED</Badge>
					{/if}
				</div>
			</div>

			<p class="truncate text-xs text-muted-foreground">{device.description}</p>

			{#if device.endpoint}
				<p class="truncate font-mono text-[10px] text-muted-foreground">{device.endpoint}</p>
			{/if}
		</div>

		<div class="flex shrink-0 flex-wrap justify-end gap-1">
			<Badge variant={device.is_rpc ? 'secondary' : 'outline'} class="text-[10px]">
				{device.is_rpc ? 'RPC' : 'Local'}
			</Badge>

			{#if device.worker_is_cpu}
				<Badge variant="tertiary" class="text-[10px]">CPU (RAM)</Badge>
			{/if}

			{#if siblingCount != null && siblingCount > 1 && siblingIndex != null}
				<Badge
					variant="outline"
					class="text-[10px]"
					title="this worker exposes {siblingCount} devices on {device.endpoint}"
				>
					box {siblingIndex}/{siblingCount}
				</Badge>
			{/if}

			{#if device.score}
				<Badge variant="outline" class="text-[10px]">
					{device.score.bw_gbps.toFixed(1)} GB/s
				</Badge>
			{/if}

			{#if device.timing}
				<Badge
					variant={execSlowest ? 'tertiary' : 'outline'}
					class="text-[10px]"
					title={`worker-side handler time (GGML_RPC_TIMING): ${device.timing.cmd} exec avg over ${device.timing.n.toLocaleString()} calls — compute + result send ON the worker; excludes network RTT and coordinator time (the wire RTT is the ms next to the transfer rates). exec max ${(device.timing.exec_max_us / 1000).toFixed(1)} ms · lock wait avg ${(device.timing.lock_avg_us / 1000).toFixed(2)} ms${execSlowest ? ' · slowest in the fleet — the pipeline long pole' : ''}`}
				>
					exec {(device.timing.exec_avg_us / 1000).toFixed(1)} ms
				</Badge>
			{/if}

			{#if rank === 'fastest'}
				<Badge class="text-[10px]">fastest</Badge>
			{:else if rank === 'slowest'}
				<Badge variant="tertiary" class="text-[10px]">slowest</Badge>
			{/if}
		</div>
	</div>

	<div class="space-y-1">
		<div class="flex justify-between text-[10px] text-muted-foreground">
			<span>Memory</span>

			<span>
				{device.memory_free_mib.toLocaleString()} / {device.memory_total_mib.toLocaleString()} MiB free
			</span>
		</div>

		<div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
			<div
				class="h-full rounded-full {memoryUsedPercent > 90 ? 'bg-destructive' : 'bg-primary'}"
				style="width: {memoryUsedPercent.toFixed(1)}%"
			></div>
		</div>
	</div>

	{#if device.attn_owner || device.n_layers != null || splitPercent !== null}
		<p class="text-xs text-muted-foreground">
			{[
				device.attn_owner ? 'attention owner' : null,
				device.n_layers != null ? `${device.n_layers} layers` : null,
				!device.attn_owner && splitPercent !== null ? `${splitPercent}%` : null
			]
				.filter(Boolean)
				.join(' · ')}
		</p>
	{/if}

	{#if device.stats}
		<div class="flex items-center justify-between gap-2 text-xs text-muted-foreground">
			<span class="truncate">
				in {(rates?.inMbps ?? 0).toFixed(2)} MB/s · out {(rates?.outMbps ?? 0).toFixed(2)} MB/s
				{#if latencyMs !== null}
					· {latencyMs} ms
				{/if}
			</span>

			<FleetSparkline
				values={rates?.history ?? []}
				capacity={FLEET_RATE_HISTORY_LENGTH}
				class="shrink-0"
			/>
		</div>
	{/if}

	{#if device.health === 'degraded' || device.health === 'recovering'}
		<div
			class="mb-2 rounded px-2 py-1 text-xs {device.health === 'degraded'
				? 'bg-destructive/15 text-destructive'
				: 'bg-orange-500/15 text-orange-600'}"
			title="failure history persists 5 min after recovery - a crash-looping worker no longer shows green between crashes"
		>
			{device.health === 'degraded' ? 'FAILING' : 'recovering'}
			{#if device.failure_count}&nbsp;· {device.failure_count} failure{device.failure_count > 1
					? 's'
					: ''}{/if}
		</div>
	{/if}

	{#if device.is_rpc && device.endpoint}
		<div class="mt-auto flex justify-end gap-2">
			<Button variant="outline" size="sm" onclick={handleShowLogs}>
				<ScrollText class="h-3.5 w-3.5" />
				Logs
			</Button>

			<span
				title={fleetAdmin
					? 'Re-run the bandwidth benchmark now (shares apply at the next reload)'
					: 'start server with --fleet-admin and an API key'}
			>
				<Button
					variant="outline"
					size="sm"
					disabled={!fleetAdmin || isRescoring}
					onclick={handleRescore}
				>
					<Gauge class="h-3.5 w-3.5 {isRescoring ? 'animate-pulse' : ''}" />
					Re-score
				</Button>
			</span>

			<span title={fleetAdmin ? undefined : 'start server with --fleet-admin and an API key'}>
				<Button
					variant="outline"
					size="sm"
					disabled={!fleetAdmin || isRestarting}
					onclick={() => (showRestartDialog = true)}
				>
					<RotateCcw class="h-3.5 w-3.5 {isRestarting ? 'animate-spin' : ''}" />
					Restart
				</Button>
			</span>
		</div>
	{/if}
</Card.Root>

<DialogConfirmation
	bind:open={showRestartDialog}
	title="Restart worker"
	description={`Restart the RPC worker at ${device.endpoint}? In-flight work on this worker will be interrupted while it re-provisions.`}
	confirmText="Restart"
	variant="destructive"
	icon={RotateCcw}
	onConfirm={handleRestartConfirm}
	onCancel={() => (showRestartDialog = false)}
/>
