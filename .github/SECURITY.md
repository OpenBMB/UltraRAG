# Security Notice

## CVE-2025-53000 (nbconvert)

**Status**: Known vulnerability, awaiting official fix

**Impact**: 
- Vulnerability affects `nbconvert` versions <= 7.16.6
- Primarily affects Windows systems
- Current project runs on Linux system, actual risk is low

**Details**:
- CVE ID: CVE-2025-53000
- Severity: High (CVSS 8.5)
- Vulnerability Type: Uncontrolled Search Path Element (CWE-427)
- Scope: When using `jupyter nbconvert --to pdf` on Windows to convert notebooks containing SVG output, malicious code may be executed

**Mitigation**:
1. Dependencies have been updated to the latest version (7.16.6)
2. `jupyter` is only used as a development dependency and not used in production
3. Project runs on Linux system and is not affected by this vulnerability
4. Will update immediately once official fix version is released

**Related Links**:
- [GitHub Advisory](https://github.com/advisories/GHSA-xm59-rqc7-hhvf)
- [Fix PR #2261](https://github.com/jupyter/nbconvert/pull/2261) (Not yet merged)

**Last Updated**: 2025-01-29
