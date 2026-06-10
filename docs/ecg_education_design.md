# ECG Education Layer — Design Specification

## Understanding Summary

| Item | Detail |
|---|---|
| **What** | Educational overlay for ECG Anomaly Detection dashboard |
| **Why** | Chart overlays 3 data layers without explanation — confusing for medical students |
| **Who** | Junior med students (need basics) + senior students/residents (need AI result interpretation) |
| **Constraints** | No backend changes, keep single-file React architecture, keep existing chart layout |
| **Non-goals** | No chart splitting, no new analysis features, no CMS |

---

## Decision Log

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| D1 | Both audiences with toggleable education mode | Separate apps, single audience | Maximizes reach; toggle keeps UI clean for advanced users |
| D2 | Global toggle + inline info per section | Global-only, inline-only, tabs | Best flexibility: power users toggle all, casual users click specific info |
| D3 | Full educational content | Charts-only, charts + thresholds | Medical education context demands comprehensive material |
| D4 | Bilingual VI/EN toggle | Vietnamese-only, English-only | Vietnamese med students benefit from native language; English for international |
| D5 | Keep single overlaid chart + explanations below | Split into 2-3 charts | Simpler implementation, no layout disruption |
| D6 | Expandable panels below analysis | Right drawer, tab-based | Natural scroll flow, easiest responsive handling |

---

## Implementation Plan

### Phase 1 — State & Controls
1. Add `eduMode`, `lang`, `expandedInfos` state variables
2. Add Education toggle button to `topbar-actions`
3. Add Language toggle button to `topbar-actions`
4. Wire localStorage persistence for both toggles

### Phase 2 — InfoButton Component
5. Create `InfoButton` component with collapsible content
6. Place InfoButton at 4 inline positions
7. Wire expand/collapse logic with `expandedInfos` Set
8. Implement auto-expand behavior when `eduMode = true`

### Phase 3 — Educational Content
9. Create bilingual content object
10. Build `EduSection` component
11. Build `ReferenceTable` component
12. Assemble 4 education sections below analysis panel
13. Add clinical disclaimer badge

### Phase 4 — Styling & Polish
14. Add CSS for all new components
15. Add collapse/expand animations
16. Add responsive rules for 3 breakpoints
17. Test dark mode + mobile layout
