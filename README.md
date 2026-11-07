# UltraRAG tutorials

Bilingual Mintlify documentation for [UltraRAG](https://github.com/OpenBMB/UltraRAG), maintained on the `page/tutorial` branch.

- Content: `pages/cn/` and `pages/en/`
- Navigation and site settings: `docs.json`
- Shared images and logos: `images/` and `logo/`

Use Node.js 20 or newer and run from this directory:

```bash
npx --yes mint@4.3.5 dev --port 3000
npx --yes mint@4.3.5 validate
npx --yes mint@4.3.5 broken-links
```

Keep both languages consistent, verify commands against the current framework, and check new pages in desktop and mobile previews before opening a pull request against `page/tutorial`.
