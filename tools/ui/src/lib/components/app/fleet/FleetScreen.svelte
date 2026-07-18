<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { AlertTriangle, Network, Plus, X } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import { ActionIcon } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Table from '$lib/components/ui/table';
	import { DialogConfirmation } from '$lib/components/app/dialogs';
	import { ROUTES } from '$lib/constants';
	import { FleetService } from '$lib/services/fleet.service';
	import { fleetStore } from '$lib/stores/fleet.svelte';
	import { formatFileSize } from '$lib/utils';
	import FleetDeviceCard from './FleetDeviceCard.svelte';
	import FleetWorkerLogs from './FleetWorkerLogs.svelte';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let status = $derived(fleetStore.status);
	let devices = $derived<ApiFleetDevice[]>(status?.devices ?? []);
	let discovered = $derived<ApiFleetDiscoveredWorker[]>(status?.discovered ?? []);
	let rpcDevices = $derived(devices.filter((device) => device.is_rpc));
	let hasRpcDevices = $derived(rpcDevices.length > 0);

	// a worker can expose several devices on one endpoint (e.g. -d CUDA0,CPU);
	// count + index them so sibling cards read as one box with two roles
	let endpointSiblings = $derived.by(() => {
		const counts: Record<string, number> = {};
		for (const device of devices) {
			if (device.endpoint) counts[device.endpoint] = (counts[device.endpoint] ?? 0) + 1;
		}
		const seen: Record<string, number> = {};
		const index: Record<string, number> = {};
		for (const device of devices) {
			if (device.endpoint && (counts[device.endpoint] ?? 0) > 1) {
				seen[device.endpoint] = (seen[device.endpoint] ?? 0) + 1;
				index[device.name] = seen[device.endpoint];
			}
		}
		return { counts, index };
	});

	let modelName = $derived.by(() => {
		const path = status?.model?.path;
		if (!path) return null;

		return path.split('/').pop() || path;
	});

	// fleet-wide totals across the pipeline devices: worker-CPU devices contribute
	// RAM + cpu layers, everything else (local GPUs, GPU workers) VRAM + gpu layers
	let fleetTotals = $derived.by(() => {
		const t = {
			ramFree: 0,
			ramTotal: 0,
			vramFree: 0,
			vramTotal: 0,
			gpuLayers: 0,
			cpuLayers: 0,
			hasLayers: false
		};
		for (const device of devices) {
			if (device.worker_is_cpu) {
				t.ramFree += device.memory_free_mib;
				t.ramTotal += device.memory_total_mib;
				t.cpuLayers += device.n_layers ?? 0;
			} else {
				t.vramFree += device.memory_free_mib;
				t.vramTotal += device.memory_total_mib;
				t.gpuLayers += device.n_layers ?? 0;
			}
			if (device.n_layers != null) t.hasLayers = true;
		}
		return t;
	});

	function formatGib(mib: number): string {
		return (mib / 1024).toFixed(mib >= 100 * 1024 ? 0 : 1);
	}

	// Rank the fastest/slowest scored RPC devices (needs at least two to compare)
	let deviceRanks = $derived.by(() => {
		const ranks: Record<string, 'fastest' | 'slowest'> = {};
		const scored = rpcDevices.filter(
			(device) => device.endpoint && typeof device.score?.bw_gbps === 'number'
		);

		if (scored.length < 2) return ranks;

		let fastest = scored[0];
		let slowest = scored[0];

		for (const device of scored) {
			if (device.score!.bw_gbps > fastest.score!.bw_gbps) fastest = device;
			if (device.score!.bw_gbps < slowest.score!.bw_gbps) slowest = device;
		}

		if (fastest !== slowest) {
			// key by device name, not endpoint: a GPU worker's CPU sibling shares
			// the endpoint and must not inherit the GPU's rank badge
			ranks[fastest.name] = 'fastest';
			ranks[slowest.name] = 'slowest';
		}

		return ranks;
	});

	let logsOpen = $state(false);
	let logsEndpoint = $state<string | null>(null);

	let includeEndpoint = $state<string | null>(null);
	let showIncludeDialog = $state(false);
	let isReloading = $state(false);

	function requestInclude(endpoint: string) {
		includeEndpoint = endpoint;
		showIncludeDialog = true;
	}

	async function handleIncludeConfirm() {
		showIncludeDialog = false;
		if (!includeEndpoint) return;

		isReloading = true;

		try {
			const result = await FleetService.reloadFleet(includeEndpoint);

			if (result.success) {
				toast.success(
					`Reloading the coordinator to enlist ${includeEndpoint} — the model reloads across the grown fleet`
				);
			} else {
				toast.error('Fleet reload request failed');
			}
		} catch (error: unknown) {
			toast.error(error instanceof Error ? error.message : String(error));
		} finally {
			isReloading = false;
		}
	}

	let previousRouteId = $state<string | null>(null);

	$effect(() => {
		const currentId = page.route.id;
		return () => {
			previousRouteId = currentId;
		};
	});

	function handleClose() {
		const prevIsFleet = previousRouteId === '/fleet';
		if (browser && window.history.length > 1 && !prevIsFleet) {
			history.back();
		} else {
			goto(ROUTES.START);
		}
	}

	function showLogs(endpoint: string) {
		logsEndpoint = endpoint;
		logsOpen = true;
	}

	function serverStateColor(state: string | undefined): string {
		switch (state) {
			case 'ready':
				return 'bg-green-500';
			case 'loading':
				return 'bg-yellow-500';
			case 'waiting-capacity':
				return 'bg-orange-500';
			case 'sleeping':
				return 'bg-gray-500';
			default:
				return 'bg-gray-500';
		}
	}

	function formatLastSeen(ms: number): string {
		if (ms < 1000) return `${ms} ms ago`;
		if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s ago`;

		return `${Math.round(ms / 60_000)} min ago`;
	}

	onMount(() => {
		fleetStore.startPolling();
	});

	onDestroy(() => {
		fleetStore.stopPolling();
	});
</script>

<div in:fade={{ duration: 150 }}>
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
	</div>

	<div
		class="sticky top-0 z-10 mt-4 mb-2 flex items-start gap-4 p-0 px-4 md:justify-between md:p-4 md:px-8"
	>
		<div class="flex flex-wrap items-center gap-x-3 gap-y-2">
			<div class="flex items-center gap-2">
				<Network class="h-5 w-5 md:h-6 md:w-6" />

				<h1 class="text-lg font-semibold md:text-2xl">Fleet</h1>
			</div>

			{#if status}
				<div class="flex items-center gap-1.5">
					<div class="h-2 w-2 rounded-full {serverStateColor(status.server_state)}"></div>

					<span class="text-sm text-muted-foreground">{status.server_state}</span>
				</div>

				{#if modelName}
					<Badge variant="outline" class="max-w-64 text-xs">
						<span class="truncate">{modelName}</span>
					</Badge>

					{#if status.model?.size_bytes}
						<span class="text-xs text-muted-foreground">
							{formatFileSize(status.model.size_bytes)}
						</span>
					{/if}
				{/if}
			{/if}
		</div>
	</div>

	<div class="space-y-6 {className}">
		{#if fleetStore.error && !status}
			<div class="rounded-md border border-destructive/50 bg-destructive/10 p-4 text-sm">
				<p class="font-medium text-destructive">Failed to fetch fleet status</p>

				<p class="mt-1 text-xs text-muted-foreground">{fleetStore.error}</p>
			</div>
		{/if}

		{#if status?.reload && (status.reload.pending || status.reload.any_endpoint_failed)}
			<div
				class="flex items-start gap-2 rounded-md border p-3 text-sm {status.reload
					.any_endpoint_failed
					? 'border-destructive/50 bg-destructive/10'
					: 'border-yellow-500/50 bg-yellow-500/10'}"
			>
				<AlertTriangle
					class="mt-0.5 h-4 w-4 shrink-0 {status.reload.any_endpoint_failed
						? 'text-destructive'
						: 'text-yellow-500'}"
				/>

				<div>
					{#if status.reload.any_endpoint_failed}
						<p class="font-medium">One or more RPC endpoints have failed.</p>
					{/if}

					{#if status.reload.pending}
						<p class="font-medium">A pipeline reload is pending.</p>
					{/if}

					<p class="text-xs text-muted-foreground">
						{status.reload.auto_recovery
							? 'Auto recovery is enabled — the pipeline will re-provision automatically.'
							: 'Auto recovery is disabled — manual intervention may be required.'}
					</p>
				</div>
			</div>
		{/if}

		{#if status?.capacity?.waiting}
			<div class="rounded-md border border-orange-500/40 bg-orange-500/10 p-3 text-sm">
				<p class="font-medium">
					Insufficient fleet capacity — the model load is on hold.
				</p>

				<p class="mt-1 text-xs text-muted-foreground">
					The model needs {formatGib(status.capacity.required_mib)} GiB (weights + KV
					reserve) but the fleet pools only {formatGib(status.capacity.available_mib)} GiB
					of device memory ({formatGib(status.capacity.required_mib - status.capacity.available_mib)}
					GiB short). Inference becomes possible once more workers join the LAN — power on another
					box with <code class="font-mono">--announce</code>{status.capacity.auto_recover
						? ' and the load starts automatically'
						: ', then restart the coordinator with the grown --rpc list'}.
				</p>
			</div>
		{/if}

		{#if status?.load_progress}
			<div class="space-y-1">
				<div class="flex justify-between text-xs text-muted-foreground">
					<span>Loading model — {status.load_progress.stage}</span>

					<span>{Math.round(status.load_progress.value * 100)}%</span>
				</div>

				<div class="h-2 w-full overflow-hidden rounded-full bg-muted">
					<div
						class="h-full rounded-full bg-primary transition-[width] duration-500"
						style="width: {Math.min(100, Math.max(0, status.load_progress.value * 100)).toFixed(
							1
						)}%"
					></div>
				</div>
			</div>
		{/if}

		<section class="space-y-3">
			<h2 class="text-sm font-semibold">Pipeline devices</h2>

			{#if status && (status.split_mode || devices.length > 0)}
				<!-- fleet summary strip: one mini stat per column, fed by /fleet/status -->
				<div class="grid grid-cols-2 gap-px overflow-hidden rounded-md border bg-border sm:grid-cols-3 lg:grid-cols-6">
					{#if status.split_mode}
						<div class="bg-card p-2">
							<div class="text-[10px] uppercase tracking-wide text-muted-foreground">split</div>
							<div class="text-sm font-semibold">{status.split_mode}</div>
						</div>
					{/if}

					{#if status.n_gpu_layers != null}
						<div class="bg-card p-2" title="-ngl: layers requested onto accelerators">
							<div class="text-[10px] uppercase tracking-wide text-muted-foreground">gpu layers</div>
							<div class="text-sm font-semibold">{status.n_gpu_layers}</div>
						</div>
					{/if}

					{#if fleetTotals.hasLayers}
						<div class="bg-card p-2" title="layers placed on GPU devices vs worker-CPU (RAM) devices">
							<div class="text-[10px] uppercase tracking-wide text-muted-foreground">layer placement</div>
							<div class="text-sm font-semibold">
								{fleetTotals.gpuLayers} gpu
								<span class="font-normal text-muted-foreground">/</span>
								{fleetTotals.cpuLayers} cpu
							</div>
						</div>
					{/if}

					{#if fleetTotals.vramTotal > 0}
						<div class="bg-card p-2" title="free / total GPU VRAM across the pipeline devices">
							<div class="text-[10px] uppercase tracking-wide text-muted-foreground">vram</div>
							<div class="text-sm font-semibold">
								{formatGib(fleetTotals.vramFree)}
								<span class="font-normal text-muted-foreground">/ {formatGib(fleetTotals.vramTotal)} GiB</span>
							</div>
						</div>
					{/if}

					{#if fleetTotals.ramTotal > 0}
						<div class="bg-card p-2" title="free / total worker RAM across the pipeline devices">
							<div class="text-[10px] uppercase tracking-wide text-muted-foreground">worker ram</div>
							<div class="text-sm font-semibold">
								{formatGib(fleetTotals.ramFree)}
								<span class="font-normal text-muted-foreground">/ {formatGib(fleetTotals.ramTotal)} GiB</span>
							</div>
						</div>
					{/if}

					{#if status.preflight}
						<div
							class="bg-card p-2"
							title={`--fleet-preflight: ${status.preflight.n_tokens} timed single-token decodes of ${status.preflight.model} over this fleet (loaded in ${status.preflight.load_s.toFixed(1)} s). A small dense model's compute is negligible, so this is the fleet's per-token boundary/latency floor - an upper bound for any model on this topology, not a throughput estimate.`}
						>
							<div class="text-[10px] uppercase tracking-wide text-muted-foreground">preflight</div>
							<div class="text-sm font-semibold">
								{status.preflight.tps.toFixed(1)} t/s
								<span class="font-normal text-muted-foreground">@ {status.preflight.model.replace(/\.gguf.*$/i, '')}</span>
							</div>
						</div>
					{/if}
				</div>
			{/if}

			{#if !status && !fleetStore.error}
				<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
					Fetching fleet status…
				</div>
			{:else if devices.length === 0 && status}
				<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
					No devices reported by the server.
				</div>
			{:else if devices.length > 0}
				<div
					class="grid gap-3"
					style="grid-template-columns: repeat(auto-fill, minmax(min(24rem, calc(100dvw - 2rem)), 1fr));"
				>
					{#each devices as device (device.name)}
						<FleetDeviceCard
							{device}
							fleetAdmin={status?.fleet_admin ?? false}
							rates={fleetStore.getRates(device.endpoint)}
							rank={deviceRanks[device.name] ?? null}
							siblingIndex={endpointSiblings.index[device.name] ?? null}
							siblingCount={device.endpoint
								? (endpointSiblings.counts[device.endpoint] ?? null)
								: null}
							onShowLogs={showLogs}
						/>
					{/each}
				</div>
			{/if}

			{#if status && !hasRpcDevices}
				<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
					No RPC workers — start workers with <code class="font-mono">--announce</code> and the
					server with <code class="font-mono">--rpc-discover</code> to build a distributed pipeline.
				</div>
			{/if}
		</section>

		<section class="space-y-3">
			<h2 class="text-sm font-semibold">Discovered on LAN</h2>

			{#if discovered.length === 0}
				<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
					No workers discovered on the LAN.
				</div>
			{:else}
				<div class="overflow-x-auto rounded-md border">
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head>Endpoint</Table.Head>
								<Table.Head>Proto</Table.Head>
								<Table.Head class="text-right">Devices</Table.Head>
								<Table.Head class="text-right">Free MiB</Table.Head>
								<Table.Head class="text-right">BW</Table.Head>
								<Table.Head class="text-right">Cache</Table.Head>
								<Table.Head class="text-right">Last seen</Table.Head>
								<Table.Head>Status</Table.Head>
							</Table.Row>
						</Table.Header>

						<Table.Body>
							{#each discovered as worker (worker.endpoint)}
								<Table.Row>
									<Table.Cell class="font-mono text-xs" title={worker.payload}>
										{worker.endpoint}
									</Table.Cell>

									<Table.Cell class="text-xs">{worker.proto ?? '—'}</Table.Cell>

									<Table.Cell class="text-right text-xs">{worker.devs ?? '—'}</Table.Cell>

									<Table.Cell class="text-right text-xs">
										{worker.free_mib != null ? worker.free_mib.toLocaleString() : '—'}
									</Table.Cell>

									<Table.Cell class="text-right text-xs">
										{worker.bw_gbps != null ? `${worker.bw_gbps.toFixed(1)} GB/s` : '—'}
									</Table.Cell>

									<Table.Cell
										class="text-right text-xs"
										title="worker tensor-cache size on disk (cache-rpc)"
									>
										{worker.cache_mib != null ? `${formatGib(worker.cache_mib)} GiB` : '—'}
									</Table.Cell>

									<Table.Cell class="text-right text-xs text-muted-foreground">
										{formatLastSeen(worker.last_seen_ms)}
									</Table.Cell>

									<Table.Cell>
										{#if worker.in_pipeline}
											<Badge variant="secondary" class="text-[10px]">in pipeline</Badge>
										{:else}
											<div class="flex items-center gap-2">
												<Badge variant="outline" class="text-[10px]">available</Badge>

												{#if status?.server_state === 'ready'}
													<span
														title={status?.fleet_admin
															? 'Reload the coordinator to include this worker'
															: 'start server with --fleet-admin and an API key'}
													>
														<Button
															variant="outline"
															size="sm"
															class="h-6 px-2 text-[10px]"
															disabled={!status?.fleet_admin || isReloading}
															onclick={() => requestInclude(worker.endpoint)}
														>
															<Plus class="h-3 w-3 {isReloading ? 'animate-spin' : ''}" />
															Include
														</Button>
													</span>
												{/if}
											</div>
										{/if}
									</Table.Cell>
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
				</div>
			{/if}
		</section>
	</div>
</div>

<FleetWorkerLogs bind:open={logsOpen} endpoint={logsEndpoint} />

<DialogConfirmation
	bind:open={showIncludeDialog}
	title="Include worker in fleet"
	description={`Enlist ${includeEndpoint} by reloading the coordinator? All in-flight requests are dropped and the model reloads across the grown fleet (the split is recomputed; every worker re-receives its share — fast when workers have a warm cache or --model-dir).`}
	confirmText="Reload fleet"
	variant="destructive"
	icon={Plus}
	onConfirm={handleIncludeConfirm}
	onCancel={() => (showIncludeDialog = false)}
/>
