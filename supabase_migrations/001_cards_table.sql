-- Community trading-card gallery
-- Run this in the Supabase SQL editor before shipping the gallery feature.

create table if not exists cards (
    id              uuid primary key default gen_random_uuid(),
    created_at      timestamptz not null default now(),

    -- Who / where
    player_id       text not null,
    player_name     text not null,
    position_group  text not null,
    team_abbr       text,
    season          integer,                -- null = career view
    season_label    text,                   -- "2024" or "career"

    -- The preset that produced this card
    bundle_weights  jsonb not null,
    score           numeric,

    -- User context
    author          text default 'Anonymous',
    caption         text,
    upvotes         integer not null default 0,

    -- Optional link to a saved algorithm
    algorithm_id    uuid references algorithms(id) on delete set null
);

create index if not exists cards_position_idx  on cards (position_group);
create index if not exists cards_team_idx      on cards (team_abbr);
create index if not exists cards_created_idx   on cards (created_at desc);
create index if not exists cards_upvotes_idx   on cards (upvotes desc);

-- RLS — anonymous users can read all cards and insert / upvote, but
-- can't update or delete other people's rows.
alter table cards enable row level security;

create policy "cards_public_read"
    on cards for select
    using (true);

create policy "cards_anon_insert"
    on cards for insert
    with check (true);

create policy "cards_anon_upvote"
    on cards for update
    using (true)
    with check (true);
