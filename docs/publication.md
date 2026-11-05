# Public export notes

The public source export and public release exclude the personal research/chat
handoff. Public text metadata uses project-relative paths instead of the author's
local workspace path. Raw original experiment artifacts remain unchanged in the
local workspace. These path edits do not change model weights or numerical results.

Public archive manifests describe the exported bytes. Historical experiment
source hashes describe the source used for those runs, not a promise that later
documentation/packaging utilities are unchanged. The original reproduction record
was captured before this publication cleanup. Demo decisions/payoffs and numerical
results remain unchanged; path fields in public replay JSON are normalized.

Use the source export for repository files and the public full ZIP as a Release
asset. A browser upload does not unpack a ZIP into source files. Do not drag the
original workspace into the browser: it also contains private notes, environments
and raw artifacts, and `.gitignore` is not a browser-upload filter.

No GitHub repository has been created or published by this preparation step.
Git has been initialized locally. Licensing has not been selected; choose a
license separately if you intend to grant reuse permissions.
