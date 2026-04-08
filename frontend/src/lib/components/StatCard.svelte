<script>
  export let label = "";
  export let value = "";
  export let icon = "";
  export let trend = "";
  export let variant = "primary"; // primary, secondary, tertiary

  const bgColors = {
    primary: "bg-primary-fixed/20",
    secondary: "bg-secondary-fixed/20",
    tertiary: "bg-tertiary-fixed/20",
  };

  const borderColors = {
    primary: "border-primary-fixed",
    secondary: "border-secondary-fixed",
    tertiary: "border-tertiary-fixed",
  };

  const iconColors = {
    primary: "text-primary",
    secondary: "text-secondary",
    tertiary: "text-tertiary",
  };

  const glowColors = {
    primary: "bg-primary-fixed",
    secondary: "bg-secondary-fixed",
    tertiary: "bg-tertiary-fixed",
  };

  const pillColors = {
    primary: "bg-white/50 text-on-surface",
    secondary: "bg-white/50 text-on-surface",
    tertiary: "bg-white/50 text-on-surface",
  };
</script>

<div class="relative overflow-hidden {bgColors[variant]} p-6 rounded-xl shadow-md border-b-4 {borderColors[variant]} hover:-translate-y-1 transition-all duration-300 group">
  <!-- Softened corner glow -->
  <div class="absolute -top-12 -right-12 w-32 h-32 {glowColors[variant]} blur-[60px] opacity-40 group-hover:opacity-60 transition-opacity"></div>
  
  <div class="relative z-10">
    <div class="flex items-center justify-between mb-4">
      <div class="w-12 h-12 rounded-xl {bgColors[variant].replace('/20', '/40')} flex items-center justify-center shadow-inner">
        <span class="material-symbols-outlined {iconColors[variant]} text-2xl">{icon}</span>
      </div>
      {#if trend}
        <span class="text-[10px] font-bold {pillColors[variant]} px-2.5 py-1 rounded-full uppercase tracking-tight backdrop-blur-sm shadow-sm">
          {trend}
        </span>
      {/if}
    </div>
    
    <p class="text-on-surface-variant/80 font-label text-[10px] uppercase tracking-widest font-bold">{label}</p>
    <h3 class="text-4xl font-extrabold text-on-surface font-headline mt-1 tracking-tighter">{value}</h3>
  </div>
</div>
