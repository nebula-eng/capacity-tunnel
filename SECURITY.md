# Security policy

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/nebula-eng/capacity-tunnel/security/advisories/new). The
report is visible only to the maintainer. There is no mailing list, so nothing sent
to an address at this project's name reaches anyone.

Include the command you ran, the input that triggers the problem, and the Python
version. You will get an acknowledgement within 72 hours.

## What counts here

This project estimates how many bits per packet an encoder could actually push through a flow, from packet metadata alone, and classifies on that estimate. That shapes what a vulnerability is:

- The evaluation dataset is synthetic. Tunnel flows follow the real channel's emission schedule; benign flows come from three generators, not from traces.
- `b32` carries its payload in DNS-like label characters, which a metadata-only detector cannot read. It is caught by transport regularity instead, and the README says so.
- All traffic is loopback. No claim is made about a production sensor.

A wrong verdict with no crash is in scope. In a measurement artifact a silently wrong
number is worse than an exception, because nothing prompts a reader to check it.

## Out of scope

- Anything that requires modifying the repository before running it.
- Differences caused by the host, where the README already states that a number is
  machine-dependent.
- The limits listed under Limitations in the README. They are known and written
  down; a report that restates one is not a finding.

## Data handling

The live channel binds a UDP socket on `127.0.0.1` and nothing else. Generated captures are written under `results/`.
