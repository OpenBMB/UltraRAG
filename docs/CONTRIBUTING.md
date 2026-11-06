# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.
Please note we have a [code of conduct](CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

## Built-in web search integrations

UltraRAG keeps its built-in web search backends small enough for the project to maintain. Before implementing a new provider, open an issue and wait for explicit maintainer agreement on the scope. Opening an issue alone does not establish acceptance.

Proposals should demonstrate community demand, explain the use case that existing backends do not adequately serve, link to a stable public API contract, and identify who will maintain the integration as the provider's API changes. Implementation tests are required but do not by themselves establish that a provider belongs in the built-in set.

Provider-specific integrations should generally be maintained as independent MCP servers or external adapters. Disclose any provider affiliation and any request fields or headers used for integration attribution. Inclusion in other frameworks or an earlier integration in UltraRAG does not guarantee acceptance. These criteria also guide reviews of existing backends; any removal will be communicated explicitly with migration guidance.

## Development environment setup

> **[?]**
> Proceed to describe how to setup local development environment.
> e.g:

To set up a development environment, please follow these steps:

1. Clone the repo

   ```sh
   git clone https://github.com/OpenBMB/UltraRAG
   ```
2. TODO

## Issues and feature requests

You've found a bug in the source code, a mistake in the documentation or maybe you'd like a new feature? You can help us by [submitting an issue on GitHub](https://github.com/OpenBMB/UltraRAG/issues). Before you create an issue, make sure to search the issue archive -- your issue may have already been addressed!

Please try to create bug reports that are:

- _Reproducible._ Include steps to reproduce the problem.
- _Specific._ Include as much detail as possible: which version, what environment, etc.
- _Unique._ Do not duplicate existing opened issues.
- _Scoped to a Single Bug._ One bug per report.

**Even better: Submit a pull request with a fix or new feature!**

### How to submit a Pull Request

1. Search our repository for open or closed
   [Pull Requests](https://github.com/OpenBMB/UltraRAG/pulls)
   that relate to your submission. You don't want to duplicate effort.
2. Fork the project
3. Create your feature branch (`git checkout -b feat/amazing_feature`)
4. Commit your changes (`git commit -m 'feat: add amazing_feature'`) UltraRAG uses [conventional commits](https://www.conventionalcommits.org), so please follow the specification in your commit messages.
5. Push to the branch (`git push origin feat/amazing_feature`)
6. [Open a Pull Request](https://github.com/OpenBMB/UltraRAG/compare?expand=1)
