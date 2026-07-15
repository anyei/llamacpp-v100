<script lang="ts">
	import { tick } from 'svelte';
	import { ArrowDownToLine } from '@lucide/svelte';
	import * as Sheet from '$lib/components/ui/sheet';
	import { Button } from '$lib/components/ui/button';
	import { FleetService } from '$lib/services/fleet.service';

	const REFRESH_INTERVAL_MS = 2000;
	const FOLLOW_THRESHOLD_PX = 16;

	interface Props {
		open: boolean;
		endpoint: string | null;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), endpoint, onOpenChange }: Props = $props();

	let logText = $state('');
	let errorMessage = $state<string | null>(null);
	let isFollowing = $state(true);
	let scrollContainer = $state<HTMLDivElement | null>(null);

	$effect(() => {
		if (!open || !endpoint) return;

		const currentEndpoint = endpoint;
		let cancelled = false;

		logText = '';
		errorMessage = null;
		isFollowing = true;

		async function refresh() {
			try {
				const response = await FleetService.fetchWorkerLog(currentEndpoint);
				if (cancelled) return;

				errorMessage = null;

				if (response.log !== logText) {
					logText = response.log;

					if (isFollowing) {
						await tick();
						scrollToBottom();
					}
				}
			} catch (error: unknown) {
				if (cancelled) return;

				errorMessage = error instanceof Error ? error.message : String(error);
			}
		}

		void refresh();
		const timer = setInterval(() => void refresh(), REFRESH_INTERVAL_MS);

		return () => {
			cancelled = true;
			clearInterval(timer);
		};
	});

	function scrollToBottom() {
		if (scrollContainer) {
			scrollContainer.scrollTop = scrollContainer.scrollHeight;
		}
	}

	function handleScroll() {
		if (!scrollContainer) return;

		// Follow only while the user stays at (or near) the bottom —
		// scrolling up pauses auto-scroll until they return to the bottom
		isFollowing =
			scrollContainer.scrollTop + scrollContainer.clientHeight >=
			scrollContainer.scrollHeight - FOLLOW_THRESHOLD_PX;
	}

	function resumeFollow() {
		isFollowing = true;
		scrollToBottom();
	}

	function handleOpenChange(newOpen: boolean) {
		open = newOpen;
		onOpenChange?.(newOpen);
	}
</script>

<Sheet.Root {open} onOpenChange={handleOpenChange}>
	<Sheet.Content side="right" class="flex w-full flex-col gap-3 p-4 sm:max-w-xl">
		<Sheet.Header class="p-0">
			<Sheet.Title class="text-sm">Worker logs</Sheet.Title>

			<Sheet.Description class="font-mono text-xs">
				{endpoint ?? 'No worker selected'}
			</Sheet.Description>
		</Sheet.Header>

		{#if errorMessage}
			<p class="rounded bg-destructive/10 p-2 text-xs text-destructive">{errorMessage}</p>
		{/if}

		<div
			bind:this={scrollContainer}
			onscroll={handleScroll}
			class="min-h-0 flex-1 overflow-y-auto rounded bg-muted/50 p-2"
		>
			{#if logText}
				<pre
					class="font-mono text-[10px] break-all whitespace-pre-wrap text-foreground/80">{logText}</pre>
			{:else if !errorMessage}
				<p class="text-xs text-muted-foreground">Waiting for log output…</p>
			{/if}
		</div>

		<div class="flex items-center justify-between">
			<span class="text-[10px] text-muted-foreground">
				Refreshes every {REFRESH_INTERVAL_MS / 1000}s
				{#if isFollowing}
					· following
				{:else}
					· paused (scrolled up)
				{/if}
			</span>

			{#if !isFollowing}
				<Button variant="outline" size="sm" onclick={resumeFollow}>
					<ArrowDownToLine class="h-3.5 w-3.5" />
					Follow
				</Button>
			{/if}
		</div>
	</Sheet.Content>
</Sheet.Root>
