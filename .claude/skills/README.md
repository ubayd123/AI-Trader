# Installed skills

## Day Trader skill pack (11 skills)

- Source: https://github.com/Jakeschincariol/day-trader-skill (version 2.0.0, commit 3d0f4ef)
- Author: Jake Schincariol. MIT licence: see `LICENSE-day-trader-skill`.
- Installed unmodified, as project-local skills: `/trader-scanner`, `/trader-news`, `/trader-entry`,
  `/trader-exit`, `/trader-timemachine`, `/trader-budget`, `/trader-risk`, `/trader-broker`,
  `/trader-alerts`, `/trader-analyst`, `/trader-pelosi`.

What it is: a step-by-step, you-approve-every-order workflow for day trading US shares. It reads free
public data (Yahoo Finance, Nasdaq Trader, SEC EDGAR, the House Clerk) with standard-library Python
and writes its working files to a `trader/` folder (ignored by git). Its order step uses Alpaca's or
Interactive Brokers' connector, paper trading by default. It does not connect to IG or eToro.

It is separate from the automated bot in `trading_bot/`. The two do not share state.
