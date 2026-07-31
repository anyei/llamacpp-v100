<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { AlertTriangle, Loader2, Rocket, X } from '@lucide/svelte';
	import * as Alert from '$lib/components/ui/alert';
	import { modelsStore, routerModels } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { ServerModelStatus } from '$lib/enums/server.enums';
	import { modelLoadProgressText } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	let loadingModels = $derived(
		isRouterMode()
			? routerModels().filter((m) => m.status?.value === ServerModelStatus.LOADING)
			: []
	);
	let progressText = $derived(
		loadingModels.length > 0
			? modelLoadProgressText(modelsStore.getLoadProgress(loadingModels[0].id))
			: null
	);

	let dismissed = new SvelteSet<string>();
	let failedModels = $derived(
		isRouterMode() ? routerModels().filter((m) => m.status?.failed && !dismissed.has(m.id)) : []
	);
</script>

{#if loadingModels.length > 0}
	<div class="pointer-events-auto mx-auto mb-4 w-full max-w-[48rem] px-1">
		<Alert.Root class="border-primary/60 bg-primary/10 p-4">
			<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin" />

			<Alert.Title class="text-base">
				Loading {loadingModels[0].name ?? loadingModels[0].id}{progressText
					? ` — ${progressText}`
					: '…'}
			</Alert.Title>

			<Alert.Description>
				Weights are streaming in — large models can take several minutes. The chat unlocks as soon
				as the model is ready.
			</Alert.Description>
		</Alert.Root>
	</div>
{:else if failedModels.length > 0}
	{@const failed = failedModels[0]}
	<div class="pointer-events-auto mx-auto mb-4 w-full max-w-[48rem] px-1">
		<Alert.Root variant="destructive" class="p-4">
			<AlertTriangle class={ICON_CLASS_DEFAULT} />

			<Alert.Title class="flex items-center justify-between text-base">
				<span>
					{failed.name ?? failed.id} failed to load (exit {failed.status.exit_code ?? '?'})
				</span>

				<button
					aria-label="Dismiss"
					onclick={() => dismissed.add(failed.id)}
					class="rounded-lg p-1 hover:bg-destructive/20"
				>
					<X class="h-4 w-4" />
				</button>
			</Alert.Title>

			<Alert.Description>
				{#if failed.status.error_tail?.length}
					<pre
						class="mt-1 mb-2 max-h-40 overflow-auto rounded-md bg-muted/50 p-2 font-mono text-xs whitespace-pre-wrap text-foreground">{failed.status.error_tail.join(
							'\n'
						)}</pre>
				{/if}

				<a href="./wizard.html" class="inline-flex items-center gap-1 font-medium underline">
					<Rocket class="h-3.5 w-3.5" />
					Adjust the setup and relaunch in the wizard
				</a>
			</Alert.Description>
		</Alert.Root>
	</div>
{/if}
