<script lang="ts">
	interface Props {
		/** Samples, oldest first. Rendered left-to-right. */
		values: number[];
		/** Total number of slots — keeps the x-scale stable while filling up. */
		capacity?: number;
		width?: number;
		height?: number;
		class?: string;
	}

	let { values, capacity = 60, width = 96, height = 24, class: className = '' }: Props = $props();

	const PADDING = 1;

	let points = $derived.by(() => {
		if (values.length < 2) return '';

		const slots = Math.max(capacity, values.length);
		const maxValue = Math.max(...values, 1e-6);
		const innerWidth = width - PADDING * 2;
		const innerHeight = height - PADDING * 2;

		return values
			.map((value, index) => {
				// Anchor the most recent sample at the right edge
				const x = PADDING + ((slots - values.length + index) / (slots - 1)) * innerWidth;
				const y = PADDING + (1 - value / maxValue) * innerHeight;

				return `${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join(' ');
	});
</script>

<svg
	viewBox="0 0 {width} {height}"
	class="text-primary {className}"
	style="width: {width}px; height: {height}px;"
	role="img"
	aria-label="Throughput history"
	preserveAspectRatio="none"
>
	{#if points}
		<polyline
			{points}
			fill="none"
			stroke="currentColor"
			stroke-width="1"
			stroke-linejoin="round"
			stroke-linecap="round"
		/>
	{:else}
		<line
			x1={PADDING}
			y1={height - PADDING}
			x2={width - PADDING}
			y2={height - PADDING}
			stroke="currentColor"
			stroke-width="1"
			opacity="0.3"
		/>
	{/if}
</svg>
