# Frontend Mobile Optimization Baseline (Chat + Common Menus)

This baseline records desktop behavior before mobile-only optimizations are applied.
It is used as a regression checklist to ensure desktop UX remains unchanged.

## Scope

- Route: `/chat`
- Main page: `ui/frontend/src/pages/ChatPage.tsx`
- Desktop layout styles: `ui/frontend/public/theme/style.css`
- Dialog and shared UI styles: `ui/frontend/src/app/styles/globals.css`

## Desktop Baseline (Must Stay Unchanged)

1. Sidebar behavior
   - Sidebar is always visible on desktop.
   - Collapsed mode works with the current collapse button.
   - Session list interactions (open, rename, delete) behave as before.

2. Header and top controls
   - Pipeline selector layout and dropdown behavior stay unchanged.
   - Status badge (Ready / Loading / Failed) display and position stay unchanged.

3. Chat area
   - Message list spacing and scroll behavior remain unchanged.
   - Input container layout, send/stop button positions, and KB selector behavior remain unchanged.

4. Common menus and dialogs
   - Settings dropdown and language submenu behavior remain unchanged.
   - Auth dialog and account settings dialog dimensions and interactions remain unchanged.

5. Detail panel
   - Reference detail sidebar open/close behavior remains unchanged.

## Regression Viewports

- Desktop: `1440x900`
- Desktop narrow: `1280x800`
- Tablet: `992x768` and `820x1180`
- Mobile: `390x844` (optimization target)

## Desktop Regression Checklist

- [ ] Sidebar collapse/expand
- [ ] New chat + session switching
- [ ] Pipeline dropdown select flow
- [ ] Settings dropdown + language submenu hover/click
- [ ] Auth dialog open/close
- [ ] Account dialog open/close and scrolling
- [ ] Chat send/stop flow
- [ ] Citation click and detail panel open/close
