# Security Policy

## Supported versions

SysPlug is pre-1.0. Security fixes are applied to the latest `0.1.x` release
and `main`.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues.**

Report privately via one of:

- GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  (Security tab → "Report a vulnerability"), or
- Email **arpitsinghgautam777@gmail.com** with a description and reproduction.

You can expect an acknowledgement within 5 business days. Once the issue is
confirmed and fixed, we will publish an advisory and credit the reporter unless
anonymity is requested.

## Scope notes

SysPlug is an analytic advisory library: it does not execute untrusted code,
deserialize untrusted data, or make network calls in its core. The optional
web dashboard under `frontend/` is intended for **local, trusted use**; do not
expose it to untrusted networks without adding authentication and input limits.
