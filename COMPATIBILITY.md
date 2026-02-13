# Compatibility Matrix

This document tracks the compatibility of `llama-stack-provider-ragas` with
different versions of [llama-stack](https://github.com/llamastack/llama-stack).

## Important Note

Prior to provider version 0.5.4, the provider versioning did not align to
llama-stack version generations. For example, provider versions 0.4.x and
0.5.0–0.5.1 all targeted llama-stack 0.2.x/0.3.x despite spanning two
provider major-minor lines. Starting with provider 0.5.4, the release
branches are organized by llama-stack compatibility:

| Release Branch   | Llama-Stack Target | Provider Versions |
|------------------|--------------------|-------------------|
| `release/0.4.x`  | 0.3.x              | 0.4.3+            |
| `release/0.5.x`  | 0.4.x              | 0.5.4+            |
| `main`           | 0.5.x+             | 0.6.0+            |

## Version Compatibility Table

| Provider Version | Llama-Stack Dependency        | Python  | Release Branch   | Notes                                |
|------------------|-------------------------------|---------|------------------|--------------------------------------|
| 0.6.0            | >=0.5.0                       | >=3.12  | `main`           | Current latest release               |
| 0.5.4            | [client]>=0.4.2,<0.5.0        | >=3.12  | `release/0.5.x`  | Maintenance release for lls 0.4.x   |
| 0.4.3            | [client]>=0.3.5,<0.4.0        | >=3.12  | `release/0.4.x`  | Maintenance release for lls 0.3.x   |
| 0.5.1            | >0.2.23 (loose)               | >=3.12  | —                | Legacy; use 0.4.3 for lls 0.3.x     |
| 0.5.0            | >0.2.23 (loose)               | >=3.12  | —                | Legacy; use 0.4.3 for lls 0.3.x     |
| 0.4.2            | >0.2.23 (loose)               | >=3.12  | —                | Legacy; use 0.4.3 for lls 0.3.x     |
| 0.4.1            | >=0.2.23 (loose)              | >=3.12  | —                | Legacy                               |
| 0.4.0            | >=0.2.23 (loose)              | >=3.12  | —                | Legacy                               |
| 0.3.6            | ==0.2.23                      | >=3.12  | —                | Legacy                               |
| 0.3.5            | ==0.2.23                      | >=3.12  | —                | Legacy                               |
| 0.3.4            | ==0.2.23                      | >=3.12  | —                | Legacy                               |
| 0.3.3            | ==0.2.23                      | >=3.12  | —                | Legacy                               |
| 0.3.2            | ==0.2.23                      | >=3.12  | —                | Legacy                               |
| 0.3.1            | ==0.2.23                      | >=3.12  | —                | Legacy                               |
| 0.3.0            | ==0.2.22                      | >=3.12  | —                | Legacy                               |

## Recommended Versions

If you need to target a specific llama-stack version, use the following
provider versions:

- **llama-stack 0.5.x**: use provider `>=0.6.0` (`pip install llama-stack-provider-ragas>=0.6.0`)
- **llama-stack 0.4.x**: use provider `==0.5.4` (`pip install llama-stack-provider-ragas==0.5.4`)
- **llama-stack 0.3.x**: use provider `==0.4.3` (`pip install llama-stack-provider-ragas==0.4.3`)
