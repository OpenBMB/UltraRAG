# UltraRAG Frontend (`ui/frontend`)

React + TypeScript frontend for UltraRAG, built with Vite.

## Stack

- React 19
- TypeScript
- Vite
- React Router
- TanStack Query
- Zustand

## Prerequisites

- Node.js 22+ (recommended)
- npm 10+

## Local Development

Start backend (terminal 1):

```bash
ultrarag show ui --host 127.0.0.1 --port 5050
```

Start frontend dev server (terminal 2):

```bash
cd ui/frontend
npm install
npm run dev
```

Vite proxy target:

- `/api` -> `http://127.0.0.1:5050`

## Available Scripts

```bash
npm run dev        # start Vite dev server
npm run lint       # run ESLint
npm run typecheck  # run TypeScript checks
npm run build      # build production assets to dist/
npm run check      # lint + typecheck + build
```

## Production Build and Backend Serving

Build output directory:

- `ui/frontend/dist`

Backend static directory resolution:

1. `ULTRARAG_FRONTEND_DIR=/absolute/path/to/dist` (highest priority)
2. default `ui/frontend/dist`

The `ultrarag show ui` command serves static assets from the resolved frontend directory.

## Repository Convention

`dist/` is intentionally committed so users can run `ultrarag show ui` without a local frontend build toolchain.

When frontend source changes:

1. run `npm run build`
2. include updated `dist/` files in the same PR

## Directory Layout

- `src/app` - app entry, router, global providers
- `src/pages` - route-level pages
- `src/features` - domain features (auth/chat/pipeline/ui)
- `src/shared` - API clients, i18n, shared UI, utilities
- `public/theme` - static theme assets (CSS, fonts, i18n resources)
