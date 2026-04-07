# Design System Strategy: The Digital Conservationist

## 1. Overview & Creative North Star
The "Creative North Star" for this design system is **"The Digital Naturalist."** This concept bridges the gap between raw, biological conservation and high-precision monitoring technology. Unlike standard dashboard templates that feel clinical and rigid, this system prioritizes organic flow, tonal depth, and an editorial layout that feels as much like a premium scientific journal as it does a software interface.

We move away from "boxed-in" designs by utilizing **intentional asymmetry** and **tonal layering**. Elements are not merely placed on a grid; they are curated on a canvas. We use high-contrast typography scale (Manrope for impact, Inter for utility) to create a clear narrative path for the user's eye.

## 2. Colors: Tonal Deep-Sea Ecology
The palette is rooted in the deep greens of coastal mangroves and the teals of tropical shallows, balanced by soft, atmospheric grays.

- **Primary (`#134231`) & Secondary (`#276868`):** Use these for foundational brand moments.
- **The "No-Line" Rule:** 1px solid borders are strictly prohibited for sectioning. Structural boundaries must be defined solely by color shifts (e.g., a `surface-container-low` card sitting on a `surface` background).
- **Surface Hierarchy:**
  - **Background (`#f8faf9`):** The base atmospheric layer.
  - **Surface-Container-Lowest (`#ffffff`):** Reserved for the most prominent interactive cards and focal points.
  - **Surface-Container-High (`#e6e9e8`):** Used for sidebar navigation or secondary grouping areas.
- **The Glass & Gradient Rule:** To evoke the water's surface, floating elements (like hover tooltips or mobile overlays) should use a backdrop-blur (12px–20px) with a semi-transparent `surface` color.
- **Signature Textures:** Apply a subtle linear gradient from `primary` to `primary_container` on primary CTAs to provide a "lit from within" professional polish.

## 3. Typography: The Editorial Scale
We use a dual-font system to separate brand authority from data-heavy utility.

- **Display & Headlines (Manrope):** Large, bold, and authoritative. These are the "Title" of the scientific report. Use `display-lg` (3.5rem) for hero stats and `headline-md` (1.75rem) for section headers to establish dominance.
- **Body & Labels (Inter):** Highly legible and neutral. Inter handles the "work" of the system.
  - **Body-lg (1rem):** Primary reading text.
  - **Label-md (0.75rem):** Used for metadata, status indicators, and small captions.
- **Visual Hierarchy:** Pair a `headline-sm` with a `label-md` in `on_surface_variant` to create a "Caption" effect that feels intentional and researched.

## 4. Elevation & Depth: Tonal Layering
Depth is achieved through physical stacking of color rather than generic drop shadows.

- **The Layering Principle:** Instead of shadows, nest surfaces. Place a `surface-container-lowest` card on a `surface-container-low` background. This creates a natural, soft lift that mimics fine paper or layered glass.
- **Ambient Shadows:** Use shadows only for "floating" interactive states (like a dragged card). They must be ultra-diffused: `0px 12px 32px rgba(25, 28, 28, 0.06)`. The tint should use the `on_surface` color, not a neutral black.
- **The "Ghost Border" Fallback:** If accessibility requires a border, use `outline_variant` at **15% opacity**. It should be a suggestion of a line, not a hard barrier.
- **Glassmorphism:** Apply to the Top Navigation bar or any floating Video Feed controls. Use `surface_container_lowest` at 80% opacity with a `blur(16px)` to integrate the UI into the environment.

## 5. Components: Precision Conservation Tools

### Buttons
- **Primary:** Solid `primary` background with `on_primary` text. Use `xl` (1.5rem) roundedness for a modern, approachable feel. 
- **Secondary:** `surface_container_highest` background. No border. This creates a "recessed" look that doesn't compete with the primary action.

### Status Indicators
- **Active/Inactive:** Instead of simple dots, use a Chip-style pill with `primary_fixed` background for Active and `surface_dim` for Inactive. Text should be `label-sm` uppercase for a "System Readout" aesthetic.

### Input Fields
- **Surface-Driven:** Use `surface_container_low` as the background for the input area. On focus, transition the background to `surface_container_lowest` and apply a subtle 2px `primary` bottom-border (no full box).

### Cards & Analytics Grids
- **Rule:** Forbid divider lines. Use `xl` (1.5rem) rounded corners for main dashboard cards and `lg` (1rem) for inner nested items.
- **Video Feed Grids:** Video containers should have an `outline-variant` 10% opacity ghost border to contain the dark footage against the light UI.

### Custom Component: The "Confidence Badge"
- For sea turtle detection, use a radial progress ring or a tonal-shift badge using `tertiary_container`. The higher the confidence, the more saturated the `tertiary` color becomes.

## 6. Do's and Don'ts

### Do:
- **Do** use extreme white space. If a section feels crowded, double the padding (use the 32px or 48px increments).
- **Do** use "Low-Opacity Layers" to show nested information (e.g., a chart legend inside a card).
- **Do** prioritize the `manrope` font for any numerical data (detections, percentages) to give it a "Tech-Forward" feel.

### Don't:
- **Don't** use pure black `#000000` for text; always use `on_surface` (`#191c1c`) to maintain a soft, natural contrast.
- **Don't** use standard "Select" dropdowns. Use custom-styled cards or overlay menus with glassmorphism effects.
- **Don't** use 1px dividers to separate list items. Use an 8px vertical gap and a background shift on hover.
- **Don't** use sharp 90-degree corners. Everything must feel like it has been "eroded" by the sea—soft, rounded, and natural (`8px` minimum).
