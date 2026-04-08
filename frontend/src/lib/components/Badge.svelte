<script>
  export let variant = "primary"; // primary, secondary, tertiary, success, warning, error
  export let text = "";
  export let pulse = false;

  const variants = {
    primary: "bg-primary-fixed text-primary",
    secondary: "bg-secondary-fixed text-secondary",
    tertiary: "bg-tertiary-fixed text-tertiary",
    success: "bg-primary-fixed text-primary", // alias
    warning: "bg-surface-dim text-on-surface-variant",
    error: "bg-error-container text-on-error-container",
  };
</script>

<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-tighter {variants[variant] || variants.primary}">
  {#if pulse}
    <span class="w-1.5 h-1.5 rounded-full bg-current animate-pulse"></span>
  {/if}
  {text}
</span>
