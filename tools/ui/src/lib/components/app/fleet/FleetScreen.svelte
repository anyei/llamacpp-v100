<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { AlertTriangle, Network, X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import * as Table from '$lib/components/ui/table';
	import { ROUTES } from '$lib/constants';
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

	let modelName = $derived.by(() => {
		const path = status?.model?.path;
		if (!path) return null;

		return path.split('/').pop() || path;
	});

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
			ranks[fastest.endpoint!] = 'fastest';
			ranks[slowest.endpoint!] = 'slowest';
		}

		return ranks;
	});

	let logsOpen = $state(false);
	let logsEndpoint = $state<string | null>(null);

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
			<div class="flex flex-wrap items-center gap-2">
				<h2 class="text-sm font-semibold">Pipeline devices</h2>

				{#if status?.split_mode}
					<Badge variant="tertiary" class="text-[10px]">split: {status.split_mode}</Badge>
				{/if}

				{#if status?.n_gpu_layers != null}
					<Badge variant="tertiary" class="text-[10px]">gpu layers: {status.n_gpu_layers}</Badge>
				{/if}
			</div>

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
							rank={device.endpoint ? (deviceRanks[device.endpoint] ?? null) : null}
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

									<Table.Cell class="text-right text-xs text-muted-foreground">
										{formatLastSeen(worker.last_seen_ms)}
									</Table.Cell>

									<Table.Cell>
										{#if worker.in_pipeline}
											<Badge variant="secondary" class="text-[10px]">in pipeline</Badge>
										{:else}
											<Badge variant="outline" class="text-[10px]">available</Badge>
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
