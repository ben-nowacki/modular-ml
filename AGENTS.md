# Mission Control conventions

This repository is managed by Mission Control.

- Repo map: read `mission-control-notes/repo_summary.md`.
- Before editing any file, open `mission-control-notes/guides/index.md` and
  follow every guide that applies to the files you touch — they are the repo
  owner's conventions.
- For UI work, first read `mission-control-notes/design_system.md` — this
  project's Radix UI + Tailwind design system (tokens + primitive kit) — and
  conform to it; then follow the frontend guides named by
  `mission-control-notes/guides/index.md`. If the manifest is missing or
  not-established, bootstrap the design system before building ad-hoc UI.
- Before finishing any task: run the repo's tests/typecheck/lint for the files
  you changed, fix what they report, and re-read your full diff against the
  task's acceptance criteria.
- For commits and pull requests, follow
  `mission-control-notes/guides/git_branch_and_pr_guide.md`. Treat a PR
  template as an outline: use its meaningful concepts and order, but never
  paste the blank template and append notes separately.
- Never attribute repository work to Claude, Codex, AI, or an agent. Do not add
  co-author trailers, generated-by notices, signatures, or similar attribution
  to commits, pull requests, comments, documentation, or repository history.
