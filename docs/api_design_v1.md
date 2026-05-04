# EdgeAcademy API — Design Document v1

**Status:** Decisions locked. Review-ready. Brett + Claude joint design (2026-05-04).
**Predecessor:** `api_contract_v0.md` (brain dump with open questions — superseded by this document).
**Next step:** Brett reviews this v1. Once approved → formal OpenAPI 3.x spec → FastAPI scaffolding → first canary endpoint.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architectural Decisions](#architectural-decisions-locked)
3. [System Architecture](#system-architecture)
4. [Cross-Cutting Concerns](#cross-cutting-concerns)
5. [Entity Model](#entity-model)
6. [Key User Flows](#key-user-flows)
7. [Security Model](#security-model)
8. [Deployment Plan](#deployment-plan)
9. [Revenue Model Summary](#revenue-model-summary)
10. [Cost Projection](#cost-projection)
11. [Migration Plan (Streamlit → API)](#migration-plan-streamlit--api)
12. [Pre-Launch Action Items](#pre-launch-action-items)
13. [Phase 1 Scope (Launch)](#phase-1-scope-launch)
14. [Deferred to Phase 2+](#deferred-to-phase-2)
15. [Open Items](#open-items-still-tbd)

---

## Executive Summary

EdgeAcademy is a paper-betting educational platform with collectible card mechanics that funnels users to sportsbooks via affiliate links.

**Primary product surfaces:**
- Mobile RN/Expo app (iOS + Android) — primary
- Streamlit web app — hardcore-fan vertical (this codebase, evolving)
- Future RN apps for fantasy + draft verticals — separate repos, same API

**Single FastAPI server** powers all clients. Modular monolith — splittable into separate services later if scale or team size justifies it.

**Launch leagues:** NFL + CFB (Power 5 minimum). Architecture is sport-agnostic from day one (`league_id` first-class), enabling NBA/MLB/NHL/soccer expansion as Phase 2 content work.

**Public launch target:** August 2026, aligned with football season kickoff.

---

## Architectural Decisions (locked)

These decisions are settled. Don't relitigate without strong reason.

| # | Decision | Rationale |
|---|---|---|
| 1 | One FastAPI server (modular monolith) | Avoid premature service splitting; one team, one repo, one deploy at launch. |
| 2 | URL pattern: `/v1/{league}/...` league-scoped, `/v1/users/me/...` cross-cutting, `/v1/{vertical}/...` vertical-specific | Future-proof for sport expansion + vertical evolution without breaking changes. |
| 3 | All entities have `league_id` where applicable | Multi-sport from day one architecturally. |
| 4 | `Bet.mode: "paper" \| "real"` | Paper for Betting School, real for graduated users. Same entity, different mode. |
| 5 | `/v1/` versioning prefix from day one | Breaking changes go to `/v2/` with deprecation plan. |
| 6 | API stable shape from launch | Mobile apps cache aggressively; breaking changes are expensive. |
| 7 | Hosting: **Fly.io** | Postgres replicas + edge presence + flexible scaling. |
| 8 | Auth: **Supabase Auth** | Already using Supabase, free tier handles launch, magic-link + OAuth out of box. |
| 9 | Database: **Supabase Postgres** | Cheapest path; can migrate to dedicated Postgres later if scale demands. |
| 10 | Card art storage: **Supabase Storage** at launch | Migrate to **Cloudflare R2** when bandwidth >$50/mo (likely 6-12 months post-launch). |
| 11 | Lines/odds data: **OddsAPI Starter ($30/mo)** | Aggregator beats single-book scraping — multi-affiliate ready, line-shopping is a feature. |
| 12 | In-game data: **OddsAPI scores endpoint** (included in plan) | Per-scoring-event frequency. Per-play feeds (SportsRadar) deferred to Phase 3+. |
| 13 | Real-time delivery: **SSE** | Simpler than WebSocket, fits push-only model, degrades to polling cleanly. |
| 14 | Geolocation: **MaxMind GeoIP2 City** (self-hosted DB, $24-50/mo) | Industry standard for affiliate compliance, zero per-request cost/latency. |
| 15 | Card art generation: **(a) pre-generated templates at launch**, evolve to **(b) dynamic SVG** for premium variants | Predictable, cheap, controllable. Dynamic generation is Phase 2. |
| 16 | API documentation: **FastAPI auto-generated OpenAPI** | Sufficient at launch; promote to ReadMe/Stoplight if we open the API to third parties later. |
| 17 | Subscription billing: **In-app on iOS + Android (eat 15-30% cut)**, Stripe for web | Conversion friction destroys more revenue than platform fees do. |
| 18 | Pricing: **$8.99/mo or $79/yr — same across all platforms** | Pricing transparency > 30% margin recovery. |
| 19 | **Apple Small Business Program — apply Day 1** | Free, drops Apple cut from 30% → 15% under $1M revenue. |
| 20 | Google external billing: **Skip at launch** | Compliance overhead + user friction not worth ~15% margin. Phase 2 if Android volume warrants. |
| 21 | Pro tier feature gates: **API-layer enforcement** | Client checks are UX-only (show/hide upgrade prompts); API is source of truth. |
| 22 | Anonymous play: **Yes, with gate at 3 bets OR first rare+ card OR 24hr session** | Friction-free first delight, signup at moment of personal investment. |

---

## System Architecture

### Component diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  iOS RN App     │    │  Android RN App │    │ Streamlit Web   │
│  (App Store)    │    │  (Play Store)   │    │ (existing)      │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────┐
                │  EdgeAcademy API                 │
                │  FastAPI on Fly.io               │
                │  api.edgeacademy.app/v1/...      │
                └──────────────────────────────────┘
                                │
       ┌─────────────┬──────────┼──────────┬──────────────┐
       │             │          │          │              │
       ▼             ▼          ▼          ▼              ▼
┌──────────┐  ┌──────────┐ ┌────────┐ ┌─────────┐ ┌──────────────┐
│ Supabase │  │ Supabase │ │OddsAPI │ │ MaxMind │ │  Sportsbook  │
│ Postgres │  │ Auth +   │ │ (lines │ │ GeoIP   │ │  webhooks    │
│ + Storage│  │ Magic    │ │ +      │ │ (local  │ │ (Income      │
│          │  │ Link/    │ │ scores)│ │ DB)     │ │  Access,     │
│          │  │ OAuth)   │ │        │ │         │ │  Impact.com) │
└──────────┘  └──────────┘ └────────┘ └─────────┘ └──────────────┘
       ▲
       │
       └─── Stripe (web) + Apple IAP + Google IAP (mobile)
```

### Request lifecycle (typical authenticated request)

```
Client
  │  Authorization: Bearer <jwt>
  ▼
Cloudflare CDN (TLS, DDoS, basic WAF)
  │
  ▼
Fly.io load balancer (geo-routed)
  │
  ▼
FastAPI app instance
  │  → JWT verification (Supabase Auth public key)
  │  → User loaded from cache or Postgres
  │  → Rate limit check (per-user, per-API-key)
  │  → Subscription tier check (if Pro endpoint)
  │  → Geo lookup (MaxMind, local — microseconds)
  │  → Business logic
  │  → Postgres queries
  ▼
Response (JSON, with consistent error shape on failure)
```

---

## Cross-Cutting Concerns

### Authentication

- **First-party clients** (Streamlit, RN apps): API keys, server-side, scoped per client
- **End users:** JWT bearer tokens via Supabase Auth, refresh-token flow
- **OAuth providers:** Apple Sign In (required for iOS), Google Sign In (cross-platform). Add Apple as required, Google as preferred. Email/magic-link as fallback.
- **Anonymous reads:** allowed for some public endpoints (player stats, public algorithms) with stricter rate limiting
- **Anonymous play:** allowed pre-signup; gate triggered at first of:
  - 3 paper bets placed
  - 1 rare-or-better card dropped
  - 24 hours of session time
  - Anonymous identity = `device_id` cookie/secure-store; on signup, anonymous data is migrated into the new user account (well-known "guest account migration" pattern)

### Errors (consistent shape)

```json
{
  "error": {
    "code": "BET_NOT_FOUND",
    "message": "Bet with id abc123 does not exist",
    "details": { "bet_id": "abc123" }
  }
}
```

Error codes are machine-readable; messages are human-readable. Client apps switch on `code`, not on `message` text.

### Pagination

- Cursor-based for high-cardinality lists (bets, cards, lifecycle events): `?cursor=xxx&limit=50`
- Offset-based for small/bounded lists (algorithms, lessons): `?offset=0&limit=20`
- All list endpoints document their pagination strategy in OpenAPI

### Rate limiting

| Tier | Limit |
|---|---|
| First-party API key | 10,000 req/min |
| Authenticated user (Free) | 600 req/min |
| Authenticated user (Pro) | 1,200 req/min |
| Anonymous | 60 req/min |
| Webhook ingress (per sportsbook) | 1,000 req/min |

Implemented via Redis-backed rate limiting. Returns 429 with `Retry-After` header on exceedance.

### CORS

Allowlist only:
- `lions-rater.streamlit.app` (existing)
- `edgeacademy.app` and subdomains (when domain bought)
- `localhost:*` (dev only, behind env flag)

Reject all others.

### Webhooks (inbound)

- `POST /v1/webhooks/sportsbook/{sportsbook_id}/conversion` — affiliate conversions
- `POST /v1/webhooks/sportsbook/{sportsbook_id}/revenue` — periodic revenue events
- `POST /v1/webhooks/stripe` — Stripe subscription events
- `POST /v1/webhooks/apple-iap` — Apple StoreKit server notifications
- `POST /v1/webhooks/google-iap` — Google Play real-time developer notifications

All webhooks verify signatures before processing. Idempotency keys deduplicate retries.

---

## Entity Model

### 1. User & Auth

```
User
- id: uuid
- email: string
- display_name: string
- created_at: timestamp
- subscription_tier: "free" | "pro"
- preferred_league: nfl | cfb | ...
- preferred_vertical: gambling | fantasy | draft | fan
- onboarding_completed: bool
- anonymous_origin_device_id: nullable (set if migrated from anonymous session)

Session (JWT, mostly stateless)
- user_id, issued_at, expires_at

ApiKey (first-party clients only)
- key (hashed)
- client_name (e.g. "edgeacademy-rn-ios")
- scopes: [route_pattern]
- created_at, revoked_at
```

**Endpoints:**
- `POST /v1/auth/signup` (email, OAuth, or anonymous-merge)
- `POST /v1/auth/login`
- `POST /v1/auth/logout`
- `POST /v1/auth/refresh`
- `GET /v1/users/me`
- `PATCH /v1/users/me` (display_name, preferences)
- `DELETE /v1/users/me` (account deletion — App Store requires)

---

### 2. Sport & League

Enum-backed, not stored as separate tables. Used for scoping.

```
League: nfl | cfb | nba | mlb | nhl | epl | uefa | mls | ...
Sport (parent grouping): football | basketball | baseball | hockey | soccer
```

Every league-scoped entity carries `league_id`.

**Endpoints:**
- `GET /v1/leagues` — currently active leagues (helps clients render UI)

---

### 3. Bet (the central entity)

```
Bet
- id: uuid
- user_id
- mode: "paper" | "real"
- league_id
- market_type: "spread" | "moneyline" | "total" | "player_prop" | "parlay" | "futures"
- selection: { team_id?, player_id?, line, side: "over"|"under"|"home"|"away" }
- stake: decimal (virtual currency for paper, USD for real)
- odds_at_placement: american_odds (e.g. -110, +250)
- expected_value: decimal (computed via alpha at placement)
- placed_at: timestamp
- placed_via_sportsbook: nullable (self-declared, not verified)
- placed_via_referral_link_id: nullable (for affiliate attribution context)
- game_ids: [game_id]  (multiple for parlays)
- status: "pending" | "live" | "settled_won" | "settled_lost" | "settled_push" | "voided" | "cashed_out"
- settled_at: nullable
- settlement_payout: nullable decimal
- notes: optional string
- tags: optional [string]
```

**Lifecycle states:**

```
pending → live (game starts) → settled_won
                             ↘ settled_lost
                             ↘ settled_push
                             ↘ voided (postponement, weather, etc.)
                             ↘ cashed_out (real-mode only)
```

**Bet entry flow: import from sportsbook lines feed (NOT manual entry)**

User picks a market from a curated list of current odds (sourced from OddsAPI). The line, odds, and market context are pre-populated. User chooses side and stake. This makes alpha contextual (we know the line they're betting against) and reduces friction.

**Real-mode bet tracking: self-report only**

For "real" mode bets, we trust the user. `placed_via_sportsbook` and `placed_via_referral_link_id` are self-declared metadata — useful for analytics but never verified. Affiliate revenue is tied to user *signing up* via our link, not to verified individual bets.

**Parlay support: yes, at launch**

Parlays add complexity (multi-leg settlement, partial outcomes don't apply but cancellations/voids do), but they're a meaningful market and beginners often start with parlays. Built in.

**Endpoints:**
- `POST /v1/users/me/bets` — place a bet (paper or real)
- `GET /v1/users/me/bets` — list, filterable by mode/status/league/date
- `GET /v1/users/me/bets/{id}` — full detail
- `PATCH /v1/users/me/bets/{id}/notes` — update notes/tags
- `POST /v1/users/me/bets/{id}/settle` — manual settlement (admin/edge cases only)
- `DELETE /v1/users/me/bets/{id}` — only allowed if pending (paper mode)
- `GET /v1/users/me/bets/live` — currently live bets
- `GET /v1/users/me/bets/live/stream` (SSE) — real-time live bet updates

---

### 4. BetCard (the collectible)

The visual artifact generated when a bet is placed. The thing that makes EdgeAcademy *EdgeAcademy* and not just another picks site.

```
BetCard
- id: uuid
- bet_id (1:1 with bet)
- art_variant_id (which template was used)
- rarity: "common" | "uncommon" | "rare" | "epic" | "legendary" | "one_of_one"
- serial_number (within rarity tier)
- league_stamp
- generated_at
- pack_opening_seen_at: nullable
- showcase_pinned: bool
- moment_type: nullable enum ("upset" | "perfect_pick" | "biggest_loss" | "lucky_break" | curator_flagged)
- final_state_art: nullable (replaces base art on settlement, for dramatic settled bets)
```

**Card generation strategy: pre-generated templates at launch (option A)**

Library of ~50-100 art templates organized by league, market type, and theme. At bet placement:
- Choose template from pool weighted by rarity tier
- Overlay variable text (player name, line, stake, odds)
- Compute serial number within rarity tier
- Output as static image (PNG/WebP)

Phase 2: dynamic SVG generation for premium variants (Pro tier polish).

**Rarity weighting (approximate):**

| Rarity | Drop rate | Triggered by |
|---|---|---|
| common | 65% | Standard bets |
| uncommon | 20% | Above-average expected value |
| rare | 10% | High EV or notable underdog |
| epic | 4% | Significant upset or big stake |
| legendary | 0.9% | Massive upset, perfect parlay leg |
| one_of_one | 0.1% | Curator-flagged or rules-based 1/1 detection |

**1-of-1 detection: rules-based at launch**

Auto-flag criteria:
- Won at odds longer than +500 (50%+ underdog)
- Parlay with 4+ legs all hit
- Net profit > 10x stake
- Net loss > 5x stake (Hall of Shame variant)
- First lifetime win (welcome moment)
- Game-deciding bet (won/lost in final play)

Curator capability deferred to Phase 2 admin tool.

**Endpoints:**
- `GET /v1/users/me/cards` — list user's cards (paginated)
- `GET /v1/users/me/cards/{id}` — full card detail with linked bet
- `POST /v1/users/me/cards/{id}/open` — mark pack-opening seen
- `POST /v1/users/me/cards/{id}/showcase` — pin/unpin to public profile
- `GET /v1/users/{user_id}/showcase` — public read of someone's showcased cards (opt-in only)

---

### 5. AchievementCard (separate entity from BetCard)

Generated by educational and milestone events (lessons completed, courses finished, skills mastered, achievements earned). Visually distinct from BetCards — each achievement has ONE canonical card design.

```
AchievementCard
- id: uuid
- user_id
- achievement_id (links to Achievement entity)
- earned_at: timestamp
- rarity: "common" | "uncommon" | "rare" | "epic" | "legendary" | "one_of_one"
- canonical_art_url (single design per achievement, no random variants)
- showcase_pinned: bool
```

**Why separate from BetCard:**
- BetCards = dynamic, story-driven, random art variants. Variable rarity from outcome.
- AchievementCards = structured, badge-like, fixed art per achievement. Rarity from difficulty.
- Same drop animation, same binder slot system — but visually + structurally distinct.

**Trigger events:**

| Trigger | Achievement | Rarity |
|---|---|---|
| First lesson completed | "First Lesson" | common |
| First quiz with 100% | "Perfect Score" | uncommon |
| Complete a full course | Course-specific commemorative | rare |
| Reach Skill Mastery Level 3 | Skill-specific card | rare |
| First parlay win | "First Parlay Win" | rare |
| Complete entire curriculum | "Curriculum Complete" | legendary |
| Complete curriculum without quiz failures | "Honors Graduate" | one_of_one (per user) |

**Endpoints:**
- `GET /v1/users/me/achievement-cards` — list achievement cards
- `GET /v1/users/me/achievement-cards/{id}` — detail
- `POST /v1/users/me/achievement-cards/{id}/showcase` — pin/unpin

---

### 6. Binder (collection view)

Conceptually a *view* over both BetCards and AchievementCards. Not a separately stored entity.

**Endpoints:**
- `GET /v1/users/me/binder` — combined view, default sort by rarity desc, group by league
- `GET /v1/users/me/binder?type=bet|achievement` — filter to one card type
- `GET /v1/users/me/binder/stats` — completion progress
- `GET /v1/users/me/binder/showcases` — what's pinned
- `POST /v1/users/me/binder/sort-preferences` — save preferences

Binder stats response example:
```json
{
  "total_cards": 47,
  "by_type": { "bet": 35, "achievement": 12 },
  "by_rarity": { "common": 30, "uncommon": 10, "rare": 5, "epic": 2, "legendary": 0, "one_of_one": 0 },
  "by_league": { "nfl": 32, "cfb": 15 },
  "completion": {
    "achievements": "12/45 unlocked",
    "nfl_2025_season": "32 cards collected"
  }
}
```

---

### 7. BetLifecycle (timeline + live updates)

```
BetEvent
- id, bet_id, timestamp
- type: "placed" | "line_moved" | "game_started" | "scoring_event" | "settled" | "card_generated" | "moment_flagged"
- payload (type-specific JSON)
```

**In-game update frequency: per-scoring-event**

Updates fire when score changes (TD, FG, basket, run, goal). Sourced from OddsAPI scores endpoint (included in our $30/mo plan). Per-play tracking deferred to Phase 3+ if/when sharp-tier features justify $400-1000/mo SportsRadar feed.

**Real-time delivery: Server-Sent Events (SSE)**

`GET /v1/users/me/bets/live/stream` opens an SSE connection. Server pushes events as they occur. Clients reconnect with last-event-id on disconnect. Falls back to polling if SSE unsupported.

**Data sources:**

| Data type | Source |
|---|---|
| Historical / season stats | nflverse, cfbfastR (existing pipeline, unchanged) |
| Live game state / scores | OddsAPI scores endpoint |
| Live odds / lines | OddsAPI lines endpoint |
| Per-play feeds | (deferred — SportsRadar Phase 3+) |

---

### 8. HallOfFame / HallOfShame

Computed views over a user's bet history. Not separately stored.

**Period choices: weekly, season, all-time** (skip monthly — doesn't align with sports rhythms).

**Educational reflection on Hall of Shame: rules-based template strings at launch**

~30-50 templates indexed by bet type + outcome pattern. Filled with the user's actual numbers. LLM-generated reflections deferred to Phase 2 for 1/1 moments, weekly recap stories, and Pro-tier personalized email reflections.

**Global leaderboards: opt-in only**

Two separate toggles in user settings:
- "Show me on global Hall of Fame"
- "Show me on global Hall of Shame"

Default both off. Display name privacy controls when opted in (anonymous, region-only, or display name).

**Endpoints:**
- `GET /v1/users/me/hall-of-fame?period=week|season|all-time` — top-N best bets
- `GET /v1/users/me/hall-of-shame?period=week|season|all-time` — top-N worst with educational framing
- `GET /v1/leaderboards/global/hall-of-fame?league=nfl&period=week`
- `GET /v1/leaderboards/global/hall-of-shame?league=nfl&period=week`
- `GET /v1/leaderboards/global/one-of-one-moments` — feed of curated 1/1 cards across opted-in users

---

### 9. Betting School (educational scaffolding)

```
Lesson
- id, title, content (markdown), course_id, order_in_course
- prerequisites: [lesson_id]
- estimated_minutes
- skills_taught: [skill_id]
- questions: [Quiz YAML embedded in frontmatter]

Course
- id, title, description, difficulty
- league_scope: nullable (some content league-specific, some sport-agnostic)
- ordered_lessons: [lesson_id]

Skill
- id, name, description, mastery_levels: [criteria]

Achievement
- id, name, description, badge_art_url
- criteria (computed against user state)

UserProgress
- user_id, target_id, target_type
- status: "not_started" | "in_progress" | "completed"
- completed_at, mastery_level

GraduationReadiness (computed signal, not stored)
- user_id, ready: bool
- signals: { paper_win_rate, days_active, lessons_completed, bankroll_discipline_score }
- recommended_next_action: string
```

**Content authoring: markdown files in `content/` folder of API repo with YAML frontmatter.**

Migration to headless CMS (Sanity/Contentful/Strapi) is Phase 2 if content velocity exceeds ~5 lessons/week or non-engineer writers join.

**Quiz/assessment structure: lightweight quizzes embedded in lesson YAML frontmatter**

- 3-5 multiple-choice questions per lesson
- 80% pass threshold (4/5 or better)
- Unlimited retries (no friction; reinforcement, not gatekeeping)
- Lesson marked "complete" only after passing
- Scores feed into GraduationReadiness signal + Skill Mastery scoring
- First-time 100% → bonus AchievementCard drop

Lesson markdown example:
```markdown
---
id: bankroll-basics-01
title: What is a Bankroll?
course: bankroll-basics
order: 1
estimated_minutes: 4
skills_taught: [bankroll_management]
questions:
  - q: "Your bankroll should be money you..."
    options:
      - "Need for rent"
      - "Can afford to lose entirely"
      - "Plan to use for groceries"
      - "Borrowed from a friend"
    correct: 1
    explanation: "A bankroll is risk capital — money you can lose without affecting your life."
---

[lesson content here...]
```

**All lessons free for all users.**

Pro tier never gates the curriculum. Educational positioning IS the compliance moat. Pro monetizes alpha analytics, card art/animations, daily bonus packs, leaderboard access — never lessons.

(Phase 2 separately: premium *non-curriculum* content — newsletters, expert interviews, tape reviews — could be Pro-gated as a different content category.)

**Endpoints:**
- `GET /v1/courses` — list courses, filterable by difficulty/league
- `GET /v1/courses/{id}` — course detail with lesson list
- `GET /v1/lessons/{id}` — lesson detail
- `POST /v1/lessons/{id}/quiz-attempt` — submit quiz answers, returns score + pass/fail
- `POST /v1/users/me/progress/{lesson_id}` — mark lesson complete (gated by quiz pass)
- `GET /v1/users/me/progress` — full progress summary
- `GET /v1/users/me/achievements` — earned achievements
- `GET /v1/users/me/skills` — current mastery levels per skill
- `GET /v1/users/me/graduation-readiness` — computed signal

---

### 10. Affiliate / Funnel

```
Sportsbook
- id, name (DraftKings, FanDuel, Caesars, BetMGM, ...)
- supported_states: [state]
- in_state_college_betting_rules: { per_state: {allowed: bool, restricted_to: [...]} }
- payout_structure (CPA only / CPA + rev_share / etc.)
- partnership_status: "active" | "pending" | "paused"

ReferralLink
- id, user_id, sportsbook_id, campaign
- deeplink (URL with affiliate tag and encoded sub_id)
- created_at, expires_at

SportsbookConversion
- id, user_id, sportsbook_id, referral_link_id
- converted_at, attribution_data, first_real_bet_at

RevenueEvent
- id, conversion_id, event_type: "cpa" | "rev_share_period"
- period_start, period_end (for rev_share)
- gross_gaming_revenue: nullable
- our_share_amount
- earned_at, received_at  -- earned vs. cash-in-bank
- status: "earned" | "approved" | "paid" | "disputed"
- payout_threshold_met: bool
```

**NCAA in-state affiliate routing:**

- Geolocation: MaxMind GeoIP2 City (self-hosted DB, $24-50/mo, monthly cron updates)
- When generating a referral link for a college-football market, API checks:
  1. User's current state (MaxMind lookup)
  2. Game's home team(s) state(s)
  3. Per-state in-state-college rules
- If restricted: returns `link_available: false` + reason. Client shows "betting unavailable in your state" notice.
- Educational content + alpha analytics are always shown regardless of state.
- Geo-fence affiliate links to US-only at launch. UK/EU/AU compliance is Phase 2.

**Sub-ID encoding (compact):**

Sub-IDs are short and decodable to support sportsbook attribution platform constraints (typically 32-50 chars max):
```
sub_id format: u{user_id_short}-l{link_id_short}
example: u00123-l00456
```

**Conversion attribution: dual-source ingestion**

- Real-time postbacks (early signal) — sportsbook calls our webhook on conversion
- Daily CSV/API report ingestion (source of truth) — cron job pulls yesterday's report from sportsbook dashboard, upserts SportsbookConversion records
- Track every referral link click on our side independently (`ReferralLinkClick` table) for reconciliation
- Expected attribution accuracy: 70-90% (industry standard). Design financial reporting to accept 10-30% lossage.

**Revenue reconciliation:**

- CPA bounties paid weekly-monthly with 30-90 day anti-fraud hold
- Rev-share paid monthly, net 30-60 day terms
- Minimum payout threshold: $100-500 (varies by sportsbook)
- First 90 days post-launch will lag — design financial planning around accrual vs. cash basis

**Endpoints:**
- `GET /v1/sportsbooks` — list sportsbooks available for current user (geo-filtered)
- `POST /v1/users/me/referral-links` — generate a referral link
  - Body: `{ sportsbook_id, campaign }`
  - Response: `{ deeplink, link_available, restriction_reason, link_id }`
- `POST /v1/webhooks/sportsbook/{sportsbook_id}/conversion` — inbound postback
- `POST /v1/webhooks/sportsbook/{sportsbook_id}/revenue` — periodic revenue events
- `GET /v1/users/me/funnel-status` — user's conversion state (for analytics)

---

### 11. Player / Team / Game (supporting content, read-only)

Read-only endpoints exposing the existing data layer (parquets + Supabase). All league-scoped.

```
Player
- id, league_id, name, position, jersey_number
- current_team_id, headshot_url, bio
- birth_date, draft_year, college (NFL only)

Team
- id, league_id, name, abbreviation, full_name
- conference, division, home_state (for NCAA routing)
- logo_url, primary_color

Game
- id, league_id, season, week (or date)
- home_team_id, away_team_id, kickoff_at, venue
- status: "scheduled" | "live" | "final"
- final_home_score, final_away_score
- broadcast_info

PlayerSeasonStats / PlayerGameStats
- league_id, player_id
- raw_stats: {sport-specific JSON}
- z_scores: {sport-specific JSON}
- snap_counts (where applicable)
```

**Endpoints:**
- `GET /v1/{league}/players?team={abbr}&position={pos}&season={year}`
- `GET /v1/{league}/players/{id}`
- `GET /v1/{league}/players/{id}/stats?season={year}`
- `GET /v1/{league}/teams`
- `GET /v1/{league}/teams/{abbr}/leaderboard?position={pos}&season={year}`
- `GET /v1/{league}/games?week={n}&season={year}`
- `GET /v1/{league}/games/{id}` — full game detail with player stats

---

### 12. Algorithms & Scoring (existing community feature, preserved)

The community algorithms feature from the current Streamlit app keeps working through the API.

```
Algorithm
- id, name, description, owner_user_id
- league_id, position_group
- weights (JSON — bundle weights and per-stat overrides)
- created_at, forked_from_id, upvote_count, is_public
```

**Endpoints:**
- `GET /v1/{league}/algorithms?position={pos}` — list public algorithms
- `GET /v1/{league}/algorithms/{id}`
- `POST /v1/{league}/algorithms` — create
- `PATCH /v1/{league}/algorithms/{id}` — update (owner only)
- `POST /v1/{league}/algorithms/{id}/fork`
- `POST /v1/{league}/algorithms/{id}/upvote`
- `POST /v1/{league}/score` — score a player population given weights

---

### 13. Subscriptions / Pro Tier

```
Subscription
- user_id, tier
- started_at, current_period_end
- payment_provider: "stripe" | "apple_iap" | "google_iap"
- provider_subscription_id
- cancellation_scheduled_at: nullable
```

**Tier structure:**

| Tier | Price | What it includes |
|---|---|---|
| **Free** (the default — full Betting Academy experience) | **$0** | Unlimited paper betting, complete educational curriculum (all lessons free forever), standard binder, basic Hall of Fame/Shame, standard card art, basic pack-opening animations, sportsbook affiliate links, NCAA in-state routing |
| **Pro** (optional upgrade) | **$8.99/mo or $79/yr** | Premium card art + extended variant pool, premium pack-opening animations, daily login bonus pack, advanced alpha analytics (deep scheme-fit + all 4 factors at full depth), larger binder limit, global leaderboards access (opt-in), priority graduation bonuses, exclusive event-pack art |

**The app is FREE to download and use.** The entire Betting Academy — paper betting, full educational curriculum, basic collectible mechanics, sportsbook affiliate routing — is permanently free. Pro is purely an *amplification* upgrade for users who love the product enough to want more polish, more depth, and exclusive cosmetics.

Same model as Spotify (free with ads vs. ad-free Premium), Strava (free tracking vs. premium analytics), Duolingo (free lessons vs. unlimited hearts).

**Why Pro never gates paper betting or education:**

If we paywall paper betting or the curriculum, we break two load-bearing things:
1. **The compliance moat** — "educational/simulation" classification depends on the school being accessible. Paywalled curriculum invites gambling-app regulatory scrutiny.
2. **The funnel economics** — beginners who hit a paywall don't graduate to real money, so we lose affiliate revenue (which is the dominant revenue stream — see Revenue Model Summary).

Pro tier exists to monetize amplification, not access.

**Billing model:**
- Same price on all platforms — $8.99/mo or $79/yr (no platform price discrimination)
- In-app subscription on iOS + Android (we eat the 15-30% Apple/Google cut rather than friction users with web-only signup)
- Stripe for web subscriptions
- Apply for Apple Small Business Program Day 1 (drops Apple cut from 30% → 15% under $1M revenue)
- Google external billing skipped at launch — revisit Phase 2 if Android volume justifies the compliance overhead

**Pro tier feature gates: API-layer enforcement (security)**

Every Pro endpoint wrapped with subscription tier check on the server. Returns 403 with `"upgrade required"` payload if free user hits Pro endpoint. Client-side checks are UX-only (show/hide upgrade prompts).

```python
# FastAPI dependency pattern
def require_pro(user: User = Depends(get_current_user)):
    if user.subscription_tier != "pro":
        raise HTTPException(403, {"error": {"code": "PRO_REQUIRED", ...}})
    return user

@app.get("/v1/users/me/alpha/deep-scheme-fit")
async def deep_scheme_fit(user: User = Depends(require_pro)):
    ...
```

**Pro tier features:**

- Premium card art + extended variant pool
- Premium pack-opening animations (cinematic flips, particles, sound)
- Daily login bonus pack
- Advanced alpha analytics (deep scheme-fit, all 4 alpha factors at full depth)
- Larger binder limit
- Global leaderboards access (opt-in)
- Priority graduation bonuses (better sportsbook signup deals)
- Exclusive event-pack art (Super Bowl, March Madness, etc.)

**Endpoints:**
- `GET /v1/users/me/subscription`
- `POST /v1/users/me/subscription/checkout` — create Stripe/IAP checkout session
- `POST /v1/users/me/subscription/cancel`
- `POST /v1/webhooks/stripe`
- `POST /v1/webhooks/apple-iap`
- `POST /v1/webhooks/google-iap`

---

### 14. Notifications

```
Notification
- id, user_id, type, payload, created_at, read_at

NotificationPreference
- user_id, type, channel: "push" | "email" | "in_app", enabled
```

Notification types:
- `bet_settled` — your bet was settled
- `live_update` — meaningful in-game event for your live bet
- `graduation_ready` — readiness signal triggered
- `achievement_earned` — you earned a new achievement
- `1of1_drop` — you got a 1/1 card
- `weekly_recap` — Sunday night pack ready

**Endpoints:**
- `GET /v1/users/me/notifications` — paginated, recent first
- `POST /v1/users/me/notifications/{id}/read`
- `POST /v1/users/me/notifications/mark-all-read`
- `GET /v1/users/me/notification-preferences`
- `PATCH /v1/users/me/notification-preferences`

---

## Key User Flows

### Flow 1: Anonymous user → first bet → signup gate

```
1. User opens app (cold install, no account)
2. App generates device_id, stores securely, makes API call with device_id header
3. API creates anonymous "shadow user" tied to device_id
4. User browses lines, places paper bet:
   POST /v1/users/me/bets {market, line, stake}
   (auth = anonymous device_id)
5. API creates Bet (mode=paper) and BetCard
6. API responds; client shows card-reveal animation
7. User places bet #2, #3 (or hits a rare drop, or 24hrs pass)
8. API includes signup_required: true in response when gate hits
9. Client shows signup prompt with user's actual cards visible:
   "Save your collection forever — sign up?"
10. User signs up (Apple/Google/email magic link)
11. POST /v1/auth/signup {oauth_token, anonymous_device_id}
12. API merges anonymous user → new authenticated user
    (all bets, cards, progress now belong to the new account)
13. Subsequent calls use JWT instead of device_id
```

### Flow 2: Place a paper bet → card reveal

```
1. Client → POST /v1/users/me/bets
   Body: { mode: "paper", league_id: "nfl", market_type: "spread",
           selection: {team_id, line: -3, side: "home"}, stake: 50 }
2. API validates user, market is currently available (lines from OddsAPI cache)
3. API computes expected_value via alpha engine (uses scoring math from lib_shared.py)
4. API determines BetCard rarity (weighted by EV + market difficulty)
5. API selects art template from variant pool
6. API creates Bet (status=pending) + BetCard records (Postgres transaction)
7. API generates BetLifecycle "placed" event
8. API generates BetLifecycle "card_generated" event
9. API checks 1/1 detection rules (likely no on placement, more likely on settlement)
10. API returns combined response: { bet, bet_card, lifecycle_events }
11. Client plays card-reveal animation (~2-3 sec)
12. Client adds card to local binder cache
```

### Flow 3: Bet settlement → lifecycle update → notification

```
1. Background job (cron) polls OddsAPI scores every 60s during live games
2. Game.status changes to "final"
3. Settlement worker queries pending Bets on this Game
4. For each Bet:
   a. Compute outcome: won / lost / push / void
   b. Update Bet.status, settled_at, settlement_payout
   c. Create BetLifecycle "settled" event
   d. If dramatic outcome → update BetCard.final_state_art
   e. Run 1/1 detection rules:
      - If criteria met: update BetCard.rarity = one_of_one, moment_type
      - Create AchievementCard if first lifetime win, etc.
   f. Push notification: "Your Lions -3 bet settled — you won $95"
   g. Invalidate user's HallOfFame/HallOfShame cache
5. SSE stream pushes settlement event to live clients
```

### Flow 4: Complete a lesson → quiz → achievement

```
1. User opens Lesson "Bankroll Basics 101"
2. Reads content, taps "Take Quiz"
3. POST /v1/lessons/{id}/quiz-attempt
   Body: { answers: [1, 2, 0, 1] }
4. API scores quiz: 4/4 correct = 100%
5. API:
   a. Creates UserProgress with status=completed (since 80%+)
   b. Updates Skill Mastery for skills taught by this lesson
   c. If first-time 100%: generates AchievementCard "Perfect Score"
   d. Recomputes GraduationReadiness signal
6. API returns { score, passed: true, achievement_unlocked: {...}, mastery_change: {...} }
7. Client shows quiz result + achievement-card-reveal animation if earned
```

### Flow 5: Graduation moment → referral → conversion → revenue

```
1. User has 5+ paper wins, completed bankroll course
2. GraduationReadiness signal: ready=true
3. App shows gentle prompt: "You're showing real skill — try real?"
4. User taps "Try at DraftKings"
5. Client → POST /v1/users/me/referral-links
   Body: { sportsbook_id: "draftkings", campaign: "graduation_nudge_v1" }
6. API:
   a. Geo lookup (MaxMind): user is in NJ
   b. Game in question: Lions vs. Bears (no in-state college conflict)
   c. Generate sub_id: u00123-l00456
   d. Build deeplink: https://draftkings.com/?aff=edgeacademy&sub_id=u00123-l00456
   e. Create ReferralLink record + log click
7. API returns { deeplink, link_available: true }
8. Client opens deeplink in external browser → DraftKings site
9. User signs up at DraftKings
10. DraftKings (via Income Access) → POST /v1/webhooks/sportsbook/draftkings/conversion
    Body: { sub_id: "u00123-l00456", external_user_id: "dk-789", converted_at }
11. API decodes sub_id → finds ReferralLink → creates SportsbookConversion record
12. ~30 days later: DK sends revenue webhook with NGR data
13. API creates RevenueEvent (event_type=cpa, status=earned)
14. ~60 days later: cash hits our bank → status=paid via reconciliation cron
```

---

## Security Model

**Authentication layers:**
- TLS-everywhere (Cloudflare TLS termination)
- JWT for end users (Supabase Auth signs, FastAPI verifies via public key)
- API keys for first-party clients (hashed in DB, scopes per route)
- Device ID for anonymous users (rate-limited heavily, ephemeral)

**Authorization:**
- Postgres RLS (Row Level Security) on user-scoped tables — users can only read/write their own bets, cards, etc.
- API-layer enforcement on Pro endpoints (subscription tier check)
- Webhook signature verification (Stripe HMAC, Apple JWT, sportsbook-specific)

**Data privacy:**
- PII never leaves our infrastructure (geolocation is local, no third-party tracking)
- User opt-in required for any public visibility (leaderboards, showcases)
- Data export endpoint (GDPR Article 20, CCPA)
- Account deletion endpoint (Apple App Store requirement)

**Secrets management:**
- Fly.io secrets for runtime config (DATABASE_URL, ODDSAPI_KEY, STRIPE_SECRET_KEY, etc.)
- No secrets in repo or env files committed to git
- Rotate API keys quarterly

**Anti-abuse:**
- Rate limits per identity (anonymous, user, API key) — see Cross-Cutting
- Captcha on signup if bot signal detected (post-launch addition)
- Anomaly detection on bet placement (impossible odds, impossible stakes) — Phase 2

---

## Deployment Plan

### Hosting

**Fly.io** for FastAPI app + worker processes.

- 2x app instances (auto-scaling) at launch — geo-distributed (US-East, US-West)
- 1x Postgres replica via Fly Postgres (or stay on Supabase Postgres at launch — see decision below)
- 1x Redis instance (rate limiting, cache, background job queue via RQ or similar)

### Database

**Launch:** Supabase Postgres (already in use, has Auth + Storage + RLS).
**If/when needed:** migrate to Fly Postgres or dedicated AWS RDS for primary writes; keep Supabase for Auth.

### Environments

| Env | Purpose | URL |
|---|---|---|
| **Local dev** | Each developer's laptop | localhost:8000 |
| **Preview** | PR-based ephemeral envs (Fly.io makes this easy) | pr-{n}.api.edgeacademy.app |
| **Staging** | Pre-prod testing, integration with sandbox sportsbooks | staging.api.edgeacademy.app |
| **Production** | Public | api.edgeacademy.app |

### CI/CD

GitHub Actions:
- On PR: lint (ruff), type check (mypy), unit tests (pytest), Pydantic schema validation
- On merge to main: deploy to staging
- Manual promote to production after smoke tests

### Monitoring

- Sentry for error tracking ($26/mo Team)
- Fly.io built-in metrics for resource usage
- Custom Postgres queries for business metrics (signups/day, bets/day, conversions/day)
- Phase 2: dedicated dashboard tool (Grafana on Fly.io, or Tinybird)

---

## Revenue Model Summary

EdgeAcademy has **three revenue paths per user**, and they have very different economics. Understanding the mix is critical to product strategy and prevents misreading the business.

### The three paths

| Path | Expected % of users | Revenue per user |
|---|---|---|
| **Pro subscription** | 2-10% | $8.99/mo (~$108/yr) |
| **Sportsbook affiliate conversion** | 10-30% | $200-500 CPA + $20-50/mo lifetime rev-share |
| **Free, paper-only forever** | 60-85% | $0 direct; community, virality, alpha calibration, conversion candidates over time |

### The critical insight

A single sportsbook conversion is worth **22+ months of Pro subscription** in CPA alone, plus perpetual rev-share on top.

Strategic implications:
- A free user who never subscribes but converts to a sportsbook is **dramatically more valuable** than a Pro subscriber who never graduates.
- Hobbyist free users (who never graduate) are still genuinely valuable — community, virality, conversion candidates over time, alpha-factor calibration data.
- **Do NOT optimize for Pro conversion at the cost of free-user happiness.** That kills affiliate revenue, which is the actual moat.
- The freemium model exists to keep the funnel wide; the Pro tier exists to monetize a small slice of power users without damaging the rest.
- The educational positioning is non-negotiable — paywalling lessons would crush both the compliance moat and the funnel.

### Projected revenue scenarios at 50K total users (Month 12)

These are illustrative scenarios, not promises. They show sensitivity to conversion rates.

| Scenario | Pro conv | Affiliate conv | Pro MRR | CPA total (cumulative) | Lifetime rev-share/mo |
|---|---|---|---|---|---|
| Conservative | 5% | 15% | $22K | $1.5M | $150K |
| Realistic | 5% | 20% | $22K | $2M | $200-300K |
| Optimistic | 10% | 25% | $45K | $3M+ | $400-500K |

Notes:
- **CPA is one-time per conversion at signup.** The $1.5-3M figure is cumulative across all conversions over the year, not monthly recurring.
- **Lifetime rev-share is monthly recurring** (typically 20-30% of Net Gaming Revenue, paid monthly with 30-60 day lag).
- Infrastructure cost holds at ~$875/mo regardless of paid mix (it scales with total active users, not paid users).
- These scenarios assume successful funnel mechanics. Without sportsbook partnerships activated and effective graduation flow, affiliate conversion drops to <5% and the model relies more heavily on Pro subscriptions.

### Industry benchmarks (for context on Pro conversion rates)

| App | Pro conversion |
|---|---|
| Spotify | 40-50% (best-in-class) |
| Robinhood Gold | 8-12% |
| Duolingo | 7-8% |
| Strava | 5-7% |
| Most freemium consumer apps | 2-5% |

EdgeAcademy is more analogous to Strava/Duolingo than Spotify — the free tier is fully usable; paid is amplification. Plan for **2-5% Pro conversion at launch, growing to 5-10% if the product is sticky**. Don't model Spotify-tier conversion; that's an outlier driven by audio-quality + offline + ads-free, none of which directly map to our value proposition.

### Why this section exists

This is operational/strategic context, not API design. It lives in this document because:
1. The cost projection (next section) is meaningless without revenue context — $875/mo at 50K users is a number; whether that's profitable depends on the revenue model.
2. Future-Brett, future investors, future engineers reading this should understand the funnel logic immediately, so they don't optimize for the wrong metric (e.g., Pro conversion at the expense of affiliate conversion).
3. Several API design decisions only make sense given this revenue model — for example, gentle graduation nudging, opt-in leaderboards, and the refusal to paywall lessons all derive from preserving the affiliate funnel.

---

## Cost Projection

### Launch month (~100 users)

| Item | Cost |
|---|---|
| Fly.io (2 small instances + Postgres) | $20-50 |
| Supabase Postgres + Auth + Storage | $0 (free tier) |
| OddsAPI Starter | $30 |
| MaxMind GeoIP2 City Lite | $24 |
| Sentry Team | $26 |
| Apple Developer | $8/mo (annualized) |
| Google Play Developer | $2/mo (year 1 amortized) |
| Domain registration | $1 |
| **Total** | **~$110/mo** |

### Year 1, month 6 (~5K users)

| Item | Cost |
|---|---|
| Fly.io (scaled) | $80 |
| Supabase Pro | $25 |
| OddsAPI Pro | $59 |
| MaxMind | $24 |
| Sentry | $26 |
| Email (SendGrid/Postmark) | $20 |
| **Total** | **~$235/mo** |

### Year 1, month 12 (~50K users)

| Item | Cost |
|---|---|
| Fly.io (scaled + replicas) | $300 |
| Supabase Pro + scaling | $75 |
| Cloudflare R2 (storage migration) | $20 |
| OddsAPI Enterprise | $299 |
| MaxMind full City | $50 |
| Sentry | $80 |
| Email | $50 |
| **Total** | **~$875/mo** |

These are infrastructure costs only. Marketing, content, legal, contractor costs are separate.

---

## Migration Plan (Streamlit → API)

The existing Streamlit app on `lions-rater.streamlit.app` keeps running during migration. Pages migrate to consume the API one at a time (strangler fig pattern).

### Phase 1: API canary (Week 1-2 of post-freeze sprint)

- Stand up FastAPI server with first endpoints: `/v1/nfl/players/{id}`, `/v1/nfl/teams/{abbr}/leaderboard`
- Streamlit `pages/WR.py` migrated as canary — calls API instead of reading parquets directly
- Validates the API design against real usage

### Phase 2: Bulk migration (Week 3-6)

- Migrate remaining Streamlit pages: QB.py, TE.py, RB, OL, DE, DT, LB, CB, S, K, P, coaches, OC, DC, GM
- Each page = one PR, one merge, one verification
- Mixed state is OK during migration: some pages on API, others still on parquets

### Phase 3: Community algorithms migration

- Algorithm CRUD endpoints stand up
- Streamlit's `community_section()` calls migrate to API
- Existing user-saved algorithms migrate via data backfill (one-time job)

### Phase 4: Streamlit becomes pure API client

- Remove direct parquet reads from Streamlit codebase
- Streamlit reads come exclusively through API
- Streamlit becomes the "hardcore-fan vertical" client of EdgeAcademy

### Data migration: Streamlit users + community algorithms

- Existing Streamlit users → migrate to Supabase Auth users (email-based)
- Existing community algorithms → migrate to Algorithm entity (with `owner_user_id` mapped from old user ID)
- One-time migration script in API repo, runs once during cutover

---

## Pre-Launch Action Items

These are operational tasks, not API decisions. Tracked separately from the API contract.

### Sportsbook affiliate applications (start ~6-8 weeks before launch)

Approval timeline ~2-4 weeks each. They'll ask for: privacy policy draft, domain (parked is OK), basic traffic projections, app description.

- [ ] DraftKings affiliate program (Income Access)
- [ ] FanDuel affiliate program (Impact.com)
- [ ] Caesars affiliate program (Income Access)
- [ ] BetMGM affiliate program (Impact.com)
- [ ] ESPN BET, Fanatics, Hard Rock — Tier 2, after launch

### Legal & compliance

- [ ] Form LLC or C-corp (depending on equity structure plans)
- [ ] Engage sports gambling/affiliate attorney (~$3-5K) in July to validate affiliate flow before public launch
- [ ] Draft and publish: Privacy Policy, Terms of Service, GDPR/CCPA notices
- [ ] Apple App Store + Google Play developer accounts:
  - Apple Developer Program: $99/year
  - Google Play: $25 one-time
- [ ] Apple Small Business Program application (Day 1 — drops fee to 15%)

### Domain & branding

- [ ] Reserve final domain (post-naming sprint in June)
- [ ] DNS setup (Cloudflare)
- [ ] App Store listing copy + screenshots + app icon

### Payment infrastructure

- [ ] Stripe account setup (web subscriptions)
- [ ] Apple In-App Purchase configuration
- [ ] Google In-App Billing configuration
- [ ] Webhook endpoints registered in Stripe + Apple + Google dashboards

### Data infrastructure

- [ ] OddsAPI account ($30/mo Starter)
- [ ] MaxMind GeoIP2 license + cron job for monthly DB updates
- [ ] Audit existing `data/college/` for CFB launch readiness (Power-5 coverage, snap counts, 3+ seasons)

### App Store readiness (4-6 weeks before public launch)

- [ ] App Store / Play Store submissions
- [ ] TestFlight beta program (iOS)
- [ ] Internal Testing track (Android)
- [ ] Marketing website / landing page

---

## Phase 1 Scope (Launch — August 2026)

In scope for August 2026 public launch:

✅ NFL + CFB (Power 5 minimum) coverage
✅ Anonymous play with signup gate
✅ Paper betting + bet card collection
✅ BetCards + AchievementCards in unified Binder
✅ Per-bet card reveals + weekly recap pack + milestone packs
✅ HallOfFame + HallOfShame (personal + opt-in global)
✅ 1-of-1 detection (rules-based)
✅ Betting School lessons + courses + quizzes (markdown content)
✅ Graduation readiness signal + gentle nudging
✅ Affiliate links to DraftKings, FanDuel, Caesars, BetMGM (whichever approved)
✅ NCAA in-state routing
✅ Stripe + Apple IAP + Google IAP subscriptions
✅ Pro tier with feature gates at API layer
✅ Real-time live bet updates via SSE
✅ Push notifications for bet settlements + live updates
✅ All existing Streamlit functionality (community algorithms, leaderboards, player stats)

---

## Deferred to Phase 2+

| Item | Phase | Why deferred |
|---|---|---|
| NBA, MLB, NHL, soccer leagues | Phase 2 | Architecture supports; data + alpha research per sport is months of work |
| Per-play in-game data (SportsRadar) | Phase 3+ | $400-1000/mo cost only justified by sharp-tier features |
| LLM-generated reflections (HoF/HoS) | Phase 2 | Templates carry 80%+ of value; LLM amplifies special moments |
| Human curator for 1-of-1 cards | Phase 2 | Admin tool when user volume justifies headcount |
| Headless CMS for content authoring | Phase 2 | Markdown is fine until ~5+ lessons/week or non-engineer writers |
| Camp Battles user-pickable outcomes | Phase 2 | Existing concept; not load-bearing for launch |
| Search-engine-style filterable alpha views | Phase 2-3 | Bloomberg-terminal vision; needs validated user base first |
| Verified bet tracking (sportsbook integration) | Phase 3 | Sportsbook APIs don't exist for non-enterprise; self-report at launch |
| Cash-prize contests | Phase 3 | Would require verified bets + state licensing |
| "Verified Bettor" Pro+ tier | Phase 3 | Requires verified tracking |
| B2B widget (embedded paper-trading on sportsbook sites) | Phase 3 | After validating direct B2C model |
| Trading-card binder marketplace | Phase 4 | Requires NFLPA Group License; only when real currency enters |
| International expansion (UK/EU/AU affiliate) | Phase 2-3 | Each region has separate compliance regime |
| VPN detection (geolocation) | Phase 2 | Industry-standard 95%+ accuracy without it; add layer if abuse seen |
| Google Android external billing | Phase 2 | Compliance overhead not worth ~15% margin at launch scale |
| Dynamic SVG card art generation | Phase 2 | Pre-generated templates work fine at launch |
| Premium non-curriculum content (newsletters, expert interviews) | Phase 2 | Pro tier already has alpha analytics + cosmetics |

---

## Open Items (Still TBD)

These are minor and can be decided closer to implementation:

- [ ] Email service provider: SendGrid vs. Postmark vs. AWS SES (likely Postmark — better deliverability for transactional)
- [ ] Push notification provider: FCM (free, Google) confirmed for Android; APNs direct for iOS
- [ ] Background job framework: RQ (Redis Queue) vs. Celery vs. Dramatiq (lean RQ — simpler)
- [ ] Final domain name (June naming sprint)
- [ ] App Store / Play Store app names + tagline copy (June naming sprint)
- [ ] Card art style direction (engage designer in June)
- [ ] Final list of 30-50 Hall of Shame template strings (June content sprint)
- [ ] Full educational curriculum outline — courses + lesson topics (June content sprint)

---

## Next Steps

1. **Brett reviews this v1.** Mark up anything that's wrong, missing, or needs more thought.
2. Once approved, **Claude produces formal OpenAPI 3.x spec** (machine-readable). Generates Pydantic models + route stubs.
3. **FastAPI repo scaffolding** — Brett bootstraps the repo (or we pair on it); brother can advise on Fly.io deployment specifics.
4. **First canary endpoint shipped:** `GET /v1/nfl/players/{id}` against existing parquet data.
5. **Streamlit WR page migration** — first client integration to validate the API design end-to-end.
6. **Iterate from there.** Each subsequent endpoint + Streamlit migration is one PR.
