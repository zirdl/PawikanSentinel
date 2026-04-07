# Pawikan Sentinel UI Revamp — Implementation Plan

## Overview
Migrate the current web app UI from the existing gray/turtle-green theme to the **Stitch "Digital Naturalist"** design system featuring deep mangrove greens, teals, and tonal layering.

---

## Design System Migration Strategy

### 1. **Color Palette Transformation**
**From:** Turtle/Earth tones (`turtle-500`, `ocean-400`, `sand-300`, etc.)  
**To:** Material Deep-Sea Ecology (`primary`, `secondary`, `tertiary`, surface hierarchy)

#### New Color Tokens (Tailwind Config)
```js
// Primary Brand
primary: '#134231',           // Deep forest green
primaryContainer: '#2d5a47',  // Medium forest green
primaryFixed: '#bcedd4',      // Light mint
primaryFixedDim: '#a1d1b9',   // Soft mint

// Secondary
secondary: '#276868',         // Deep teal
secondaryContainer: '#acebeb', // Light teal
secondaryFixed: '#afeeed',     // Pale teal
secondaryFixedDim: '#93d2d1',  // Soft teal

// Tertiary
tertiary: '#194135',          // Dark sea green
tertiaryContainer: '#31594b',  // Medium sea green
tertiaryFixed: '#c0ecda',      // Light seafoam
tertiaryFixedDim: '#a5d0be',   // Soft seafoam

// Surface Hierarchy (NO pure whites except surface-container-lowest)
background: '#f8faf9',
surface: '#f8faf9',
surfaceContainerLowest: '#ffffff',  // Only for prominent cards
surfaceContainerLow: '#f2f4f3',
surfaceContainer: '#eceeed',
surfaceContainerHigh: '#e6e9e8',
surfaceContainerHighest: '#e1e3e2',

// Text
onSurface: '#191c1c',          // Primary text (NOT black)
onSurfaceVariant: '#414944',   // Secondary text
onPrimary: '#ffffff',          // Text on primary
onPrimaryContainer: '#9fcfb7',
```

### 2. **Typography Overhaul**
**From:** Plus Jakarta Sans (single font)  
**To:** Dual-font system
- **Headlines/Numbers:** Manrope (700, 800)
- **Body/Labels:** Inter (400, 500, 600)

### 3. **Key Design Principles to Implement**
- ✅ **No-Line Rule:** Replace 1px borders with surface color shifts
- ✅ **Tonal Layering:** Use nested surfaces instead of shadows
- ✅ **Round Everything:** 8px minimum, 12px (xl) for cards
- ✅ **Glassmorphism:** Backdrop-blur on nav bars
- ✅ **No pure black text:** Always use `onSurface` (`#191c1c`)

---

## Implementation Phases

### **Phase 1: Foundation (Tailwind + Assets)** 🔧
1. **Update `tailwind.config.js`**
   - Replace turtle/ocean/sand palette with new color system
   - Change font family from Plus Jakarta Sans → Manrope + Inter
   - Update shadow definitions to use tonal shadows
   - Keep semantic colors (success, warning, danger, info)

2. **Update CSS files**
   - Update `tailwind.css` to import Manrope + Inter
   - Add custom utility classes for glassmorphism
   - Add surface layering utilities

3. **Download fonts & assets**
   - Ensure Manrope and Inter are loaded via Google Fonts
   - Update Material Symbols import

**Files to modify:**
- `tailwind.config.js`
- `src/web/static/css/tailwind.css`
- `src/web/templates/base.html` (font imports)

---

### **Phase 2: Base Template Revamp** 🏗️
**Goal:** Transform `base.html` with new design system

**Changes:**
1. **Top Navigation Bar**
   - Add glassmorphism: `bg-white/80 backdrop-blur-lg`
   - Change shadow to subtle tonal: `shadow-sm`
   - Update logo area with new brand colors
   - User avatar with `primary-fixed` border

2. **Sidebar Navigation**
   - Background: `bg-surface-container-high` (not white)
   - Active state: `bg-primary-fixed text-primary` (not turtle-50)
   - Hover states: tonal shifts instead of gray
   - Rounded nav items: `rounded-xl` (not rounded-lg)

3. **Footer**
   - Simplified with new color scheme
   - Use `on-surface-variant` for text

**Files to modify:**
- `src/web/templates/base.html`

---

### **Phase 3: Dashboard Screen** 📊
**Goal:** Match Stitch Dashboard design exactly

**Changes:**
1. **Stats Cards** (3 columns instead of 4)
   - Border-bottom accent instead of icon box: `border-b-4 border-primary`
   - Hover lift effect: `hover:-translate-y-1 transition-transform`
   - Large numbers with Manrope: `text-4xl font-extrabold font-headline`
   - Remove icon boxes, use inline icons

2. **Confidence History Chart**
   - Keep Chart.js but style with new colors
   - Update chart line color to `primary`
   - Add time filter buttons (24h, 7d)

3. **Detection Gallery**
   - Keep existing functionality
   - Update overlay colors to match new theme

4. **Recent Detections Table**
   - Remove border lines, use surface shifts
   - Add confidence progress bars (not just percentages)
   - Status badges: `bg-primary-fixed text-primary` (Verified)
   - Header with `on-surface-variant` uppercase labels

**Files to modify:**
- `src/web/templates/dashboard.html`

---

### **Phase 4: Camera Feeds Screen** 📹
**Goal:** Transform into modern video surveillance grid

**Changes:**
1. **Camera Cards**
   - Large video preview area with gradient overlay
   - Status badges top-left: `bg-primary-fixed` with pulse animation
   - Resolution badges: `bg-black/40 backdrop-blur-md`
   - Hover play button with glassmorphism
   - Confidence badges on each card
   - Dual action buttons (VLC + View)

2. **Grid Layout**
   - 3 columns on xl screens: `xl:grid-cols-3`
   - Larger gaps: `gap-8`
   - Add "Add Camera Feed" dashed card

**Files to modify:**
- `src/web/templates/cameras.html`

---

### **Phase 5: Settings Screen** ⚙️
**Goal:** Bento-grid layout with tonal cards

**Changes:**
1. **Layout Transformation**
   - Change from 2-column to 12-column grid
   - System Config: `lg:col-span-8`
   - Account: `lg:col-span-4`
   - Cameras: `lg:col-span-12` (full width)
   - Contacts + Backup: `lg:col-span-6` each

2. **Range Sliders**
   - Custom accent colors per slider:
     - Confidence: `accent-primary`
     - Frame Skip: `accent-secondary`
     - SMS Cooldown: `accent-tertiary`
   - Value badges with matching colors

3. **Input Fields**
   - Remove borders, use bottom-border on focus
   - Background: `bg-surface-container-low`
   - Focus state: `border-b-2 border-primary`

4. **Camera Items in Settings**
   - Compact card design
   - Hover-reveal edit/delete buttons
   - Status pills with tonal backgrounds

5. **Contacts Section**
   - Avatar circles with initials
   - Compact list layout
   - Action menus

**Files to modify:**
- `src/web/templates/settings.html`
- `src/web/templates/_camera_list.html`
- `src/web/templates/_contact_list.html`

---

### **Phase 6: Component Macros** 🧩
**Goal:** Update reusable components

**Changes:**
1. **Button Macro**
   - Primary: `bg-primary text-on-primary rounded-xl`
   - Secondary: `bg-surface-container-highest text-on-surface`
   - Update hover states
   - Add gradient support for primary

2. **Card Macro**
   - Remove border-based variants
   - Use surface layering: `bg-surface-container-lowest`
   - Rounded corners: `rounded-xl`
   - Subtle shadows only

3. **Badge Macro**
   - Use tonal backgrounds: `bg-primary-fixed text-on-primary-fixed`
   - Rounded-full always
   - Uppercase labels

4. **Input Macro**
   - Surface-driven design
   - Bottom-border focus state
   - No full box borders

5. **Stat Card Macro**
   - Border-bottom accent design
   - Larger typography
   - Hover lift effect

**Files to modify:**
- `src/web/templates/components/macros.html`

---

### **Phase 7: Login & Auth Screens** 🔐
**Goal:** Match new design system

**Files to modify:**
- `src/web/templates/login.html`

---

### **Phase 8: Polish & Testing** ✨
1. Cross-browser testing
2. Dark mode adjustments (if needed)
3. Mobile responsive tweaks
4. Performance optimization
5. Accessibility audit

---

## File Change Summary

| File | Changes | Priority |
|------|---------|----------|
| `tailwind.config.js` | Complete color/font overhaul | 🔴 Critical |
| `src/web/static/css/tailwind.css` | Font imports, utilities | 🔴 Critical |
| `src/web/templates/base.html` | Nav, sidebar, footer | 🔴 Critical |
| `src/web/templates/dashboard.html` | Stats, chart, gallery | 🟡 High |
| `src/web/templates/cameras.html` | Video cards grid | 🟡 High |
| `src/web/templates/settings.html` | Bento layout, forms | 🟡 High |
| `src/web/templates/components/macros.html` | All component updates | 🟡 High |
| `src/web/templates/login.html` | Auth screen | 🟢 Medium |
| `src/web/templates/_camera_list.html` | Camera list items | 🟢 Medium |
| `src/web/templates/_contact_list.html` | Contact items | 🟢 Medium |

---

## Migration Risks & Mitigations

### ⚠️ Risks
1. **Breaking existing custom classes:** Some hardcoded `turtle-500` classes in templates
2. **Chart.js color dependencies:** Need to update chart configuration
3. **HTMX partial templates:** May reference old color classes

### ✅ Mitigations
1. Keep old color names as aliases in Tailwind config during transition
2. Search and replace all `turtle-*`, `ocean-*`, `sand-*` references
3. Test all HTMX endpoints after migration

---

## Next Steps

1. ✅ Review this plan
2. ⏳ Start with **Phase 1** (Tailwind config + CSS)
3. ⏳ Proceed through phases sequentially
4. ⏳ Test after each phase
5. ⏳ Commit incremental changes

---

## Design System Reference
- **Stitch Project ID:** `16572683093760694977`
- **Local Assets:** `/home/gio/Desktop/PawikanSentinel/stitch-design/`
- **Design MD:** `/home/gio/Desktop/PawikanSentinel/stitch-design/design-system.md`
- **Color Palette:** `/home/gio/Desktop/PawikanSentinel/stitch-design/color-palette.md`
