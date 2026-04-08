<script>
  export let label = "";
  export let value = "";
  export let icon = "";
  export let trend = ""; // Optional: e.g., "+12% vs last week"
  export let variant = "primary"; // primary, secondary, tertiary

  const borderColors = {
    primary: "border-primary",
    secondary: "border-secondary",
    tertiary: "border-tertiary",
  };

  const iconColors = {
    primary: "text-primary",
    secondary: "text-secondary",
    tertiary: "text-tertiary",
  };
</script>

<div class="bg-surface-container-lowest p-6 rounded-xl shadow-sm border-b-4 {borderColors[variant]} hover:-translate-y-1 transition-transform duration-300">
  <div class="flex items-center justify-between mb-4">
    <span class="material-symbols-outlined {iconColors[variant]} text-3xl">{icon}</span>
    {#if trend}
      <span class="text-[10px] font-bold text-primary-container bg-primary-fixed/30 px-2 py-1 rounded">
        {trend}
      </span>
    {/if}
  </div>
  <p class="text-on-surface-variant font-label text-[10px] uppercase tracking-widest">{label}</p>
  <h3 class="text-4xl font-extrabold text-on-surface font-headline mt-1">{value}</h3>
</div>
