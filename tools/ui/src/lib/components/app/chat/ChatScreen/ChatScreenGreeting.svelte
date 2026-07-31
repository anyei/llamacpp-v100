<script lang="ts">
	import { Rocket } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { serverStore, isRouterMode } from '$lib/stores/server.svelte';
	import { routerModels, loadedModelIds } from '$lib/stores/models.svelte';
	import { ServerModelStatus } from '$lib/enums/server.enums';

	interface Props {
		isEmpty: boolean;
	}

	let { isEmpty = false }: Props = $props();

	// router with nothing loaded or loading: guide the user to the launch wizard
	let noModel = $derived(
		isRouterMode() &&
			loadedModelIds().length === 0 &&
			!routerModels().some((m) => m.status?.value === ServerModelStatus.LOADING)
	);
</script>

<div
	class={[
		'pointer-events-none mb-4 hidden px-4 text-center text-balance',
		isEmpty && 'mb-[calc(50dvh-8rem)] md:mb-8 pointer-events-auto block!'
	]}
>
	{#if noModel}
		<h1 class="mb-2 text-2xl font-semibold tracking-tight md:text-3xl">No model is loaded</h1>

		<p class="mb-4 text-muted-foreground md:text-lg">
			Pick a model and how to run it in the launch wizard — it reads your hardware and the fleet,
			and pre-selects the best route.
		</p>

		<Button
			size="lg"
			onclick={() => {
				window.location.href = './wizard.html';
			}}
		>
			<Rocket class="mr-1 h-4 w-4" />
			Open the launch wizard
		</Button>
	{:else}
		<h1 class="mb-2 text-2xl font-semibold tracking-tight md:text-3xl">Hello there</h1>

		<p class="text-muted-foreground md:text-lg">
			{serverStore.props?.modalities?.audio ? 'Record audio, type a message ' : 'Type a message'} or upload
			files to get started
		</p>
	{/if}
</div>
