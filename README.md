# obsidian-wiki

A knowledge mgmt system inspired by [gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) published by Andrej Karpathy about maintaining a personal knowledge base with LLMs : the "LLM Wiki" pattern.

Instead of asking an LLM the same questions over over (or doing RAG every time), you compile knowledge once into interconnected markdown files and keep them current. In this case Obsidian is the viewer and the LLM is the maintainer.

We took that and built a framework around it. The whole thing is a set of markdown skill files that any AI coding agent (Claude Code, Cursor, Windsurf, whatever you use) can read and execute. You point it at your Obsidian vault and tell it what to do.

## Quick Start

```bash
git clone https://github.com/Ar9av/obsidian-wiki.git
cd obsidian-wiki
bash setup.sh      # ← configures your agent automatically
```

Set your vault path in `.env`:

```
OBSIDIAN_VAULT_PATH=/path/to/your/vault
```

Open the project in your agent and say **"set up my wiki"**. That's it.

## Agent Compatibility

This framework works with **any AI coding agent** that can read files. The `setup.sh` script automatically configures skill discovery for each one:

| Agent                                                     | Bootstrap File                     | Skills Directory    | Slash Commands                              |
| --------------------------------------------------------- | ---------------------------------- | ------------------- | ------------------------------------------- |
| **[Claude Code](https://claude.ai/code)**                 | `CLAUDE.md`                        | `.claude/skills/`   | ✅ `/obsidian-ingest`, `/wiki-status`, etc. |
| **[Cursor](https://cursor.com)**                          | `.cursor/rules/obsidian-wiki.mdc`  | `.cursor/skills/`   | ✅ `/obsidian-ingest`, `/wiki-status`, etc. |
| **[Windsurf](https://windsurf.com)**                      | `.windsurf/rules/obsidian-wiki.md` | `.windsurf/skills/` | ✅ via Cascade                              |
| **[Codex (OpenAI)](https://openai.com/codex)**            | `AGENTS.md`                        | — (uses AGENTS.md)  | —                                           |
| **[Antigravity (Google)](https://aistudio.google.com)**   | `GEMINI.md`                        | `.agents/skills/`   | ✅ via skill triggers                       |
| **[GitHub Copilot](https://github.com/features/copilot)** | `.github/copilot-instructions.md`  | —                   | —                                           |

> **How it works:** Each agent has its own convention for discovering skills. `setup.sh` symlinks the canonical `.skills/` directory into each agent's expected location, and creates the bootstrap file that tells the agent about the project. You write skills once, every agent can use them.

### Manual setup (if you prefer)

If you don't want to run `setup.sh`, you can configure your agent manually:

<details>
<summary><b>Claude Code</b></summary>

Skills are auto-discovered from `.claude/skills/`. Either:

- Run `setup.sh` to create symlinks, OR
- Copy `.skills/*` to `.claude/skills/`

The `CLAUDE.md` file at the repo root is automatically loaded as project context.

```bash
cd /path/to/obsidian-wiki && claude "set up my wiki"
```

</details>

<details>
<summary><b>Cursor</b></summary>

Skills are auto-discovered from `.cursor/skills/`. The `.cursor/rules/obsidian-wiki.mdc` file provides always-on context. Either:

- Run `setup.sh` to create symlinks, OR
- Copy `.skills/*` to `.cursor/skills/`

Open the project in Cursor and type `/obsidian-setup` in the chat.

</details>

<details>
<summary><b>Windsurf</b></summary>

Cascade reads rules from `.windsurf/rules/` and skills from `.windsurf/skills/`. Either:

- Run `setup.sh` to create symlinks, OR
- Copy `.skills/*` to `.windsurf/skills/`

Open in Windsurf and tell Cascade: "set up my wiki".

</details>

<details>
<summary><b>Codex (OpenAI)</b></summary>

Codex reads the `AGENTS.md` file at the repo root for project context. Skills are referenced by path in AGENTS.md — no symlinks needed.

```bash
cd /path/to/obsidian-wiki && codex "set up my wiki"
```

</details>

<details>
<summary><b>Antigravity / Gemini</b></summary>

Gemini agents read `GEMINI.md` at the repo root and discover skills from `.agents/skills/` or `.skills/`. Either:

- Run `setup.sh` to create symlinks, OR
- The `.skills/` directory is already compatible

Open in AI Studio and say "set up my wiki".

</details>

<details>
<summary><b>GitHub Copilot</b></summary>

Copilot reads `.github/copilot-instructions.md` for project context. Skills are referenced by path — Copilot will follow the instructions to read the relevant SKILL.md files.

Use Copilot Chat in VS Code and say "set up my wiki".

</details>

## How it works

Every ingest runs through four stages:

**1. Ingest** — The agent reads your source material directly. It handles whatever you throw at it: markdown files, PDFs (with page ranges), JSONL conversation exports, plain text logs, chat exports, meeting transcripts. No preprocessing step, no pipeline to run. The agent reads the file the same way it reads code.

**2. Extract** — From the raw source, the agent pulls out concepts, entities, claims, relationships, and open questions. A conversation about debugging a React hook yields a "stale closure" pattern. A research paper yields the key idea and its caveats. A work log yields decisions and their rationale. Noise gets dropped, signal gets kept.

**3. Resolve** — New knowledge gets merged against what's already in the wiki. If a concept page exists, the agent updates it — merging new information, noting contradictions, strengthening cross-references. If it's genuinely new, a page gets created. Nothing is duplicated. Sources are tracked in frontmatter so every claim stays attributable.

**4. Schema** — The wiki schema isn't fixed upfront. It emerges from your sources and evolves as you add more. The agent maintains coherence: categories stay consistent, wikilinks point to real pages, the index reflects what's actually there. When you add a new domain (a new project, a new field of study), the schema expands to accommodate it without breaking what exists.

A `.manifest.json` tracks every source that's been ingested — path, timestamps, which wiki pages it produced. On the next ingest, the agent computes the delta and only processes what's new or changed.

## What we added on top of Karpathy's pattern

- **Delta tracking.** A manifest tracks every source file that's been ingested: path, timestamps, which wiki pages it produced. When you come back later, it computes the delta and only processes what's new or changed. You're not re-ingesting your entire document library every time.

- **Project-based organization.** Knowledge gets filed under projects when it's project-specific, globally when it's not. Both are cross-referenced with wikilinks. If you're working on 10 different codebases, each one gets its own space in the vault.

- **Archive and rebuild.** When the wiki drifts too far from your sources, you can archive the whole thing (timestamped snapshot, nothing lost) and rebuild from scratch. Or restore any previous archive.

- **Multi-agent ingest.** Documents, PDFs, Claude Code history (`~/.claude`), Codex sessions (`~/.codex/`), Windsurf data (`~/.windsurf`), ChatGPT exports, Slack logs, meeting transcripts, raw text. There's a specific skill for Claude history that understands the JSONL format and memory files, and a catch-all skill that figures out whatever format you throw at it.

- **Audit and lint.** Find orphaned pages, broken wikilinks, stale content, contradictions, missing frontmatter. See a dashboard of what's been ingested vs what's pending.

- **Automated cross-linking.** After ingesting new pages, the cross-linker scans the vault for unlinked mentions and weaves them into the knowledge graph with `[[wikilinks]]`. No more orphan pages.

- **Tag taxonomy.** A controlled vocabulary of canonical tags stored in `_meta/taxonomy.md`, with a skill that audits and normalizes tags across your entire vault.

## Skills

Everything lives in `.skills/`. Each skill is a markdown file the agent reads when triggered:

| Skill                   | What it does                                      | Slash Command            |
| ----------------------- | ------------------------------------------------- | ------------------------ |
| `obsidian-setup`        | Initialize vault structure                        | `/obsidian-setup`        |
| `obsidian-ingest`       | Distill documents into wiki pages                 | `/obsidian-ingest`       |
| `claude-history-ingest` | Mine your `~/.claude` conversations and memories  | `/claude-history-ingest` |
| `data-ingest`           | Ingest any text — chat exports, logs, transcripts | `/data-ingest`           |
| `wiki-status`           | Show what's ingested, what's pending, the delta   | `/wiki-status`           |
| `wiki-rebuild`          | Archive, rebuild from scratch, or restore         | `/wiki-rebuild`          |
| `obsidian-query`        | Answer questions from the wiki                    | `/obsidian-query`        |
| `obsidian-lint`         | Find broken links, orphans, contradictions        | `/obsidian-lint`         |
| `cross-linker`          | Auto-discover and insert missing wikilinks        | `/cross-linker`          |
| `tag-taxonomy`          | Enforce consistent tag vocabulary across pages    | `/tag-taxonomy`          |
| `llm-wiki`              | The core pattern and architecture reference       | `/llm-wiki`              |
| `skill-creator`         | Create new skills                                 | `/skill-creator`         |

> **Note:** Slash commands (`/skill-name`) work in Claude Code, Cursor, and Windsurf. In other agents, just describe what you want and the agent will find the right skill.

## Project Structure

```
obsidian-wiki/
├── .skills/                          # ← Canonical skill definitions (source of truth)
│   ├── obsidian-setup/SKILL.md
│   ├── obsidian-ingest/SKILL.md
│   ├── claude-history-ingest/SKILL.md
│   ├── data-ingest/SKILL.md
│   ├── wiki-status/SKILL.md
│   ├── wiki-rebuild/SKILL.md
│   ├── obsidian-query/SKILL.md
│   ├── obsidian-lint/SKILL.md
│   ├── cross-linker/SKILL.md
│   ├── tag-taxonomy/SKILL.md
│   ├── llm-wiki/SKILL.md
│   └── skill-creator/SKILL.md
│
├── CLAUDE.md                         # Bootstrap → Claude Code
├── GEMINI.md                         # Bootstrap → Gemini / Antigravity
├── AGENTS.md                         # Bootstrap → Codex / OpenAI
├── .cursor/rules/obsidian-wiki.mdc   # Bootstrap → Cursor
├── .windsurf/rules/obsidian-wiki.md  # Bootstrap → Windsurf
├── .github/copilot-instructions.md   # Bootstrap → GitHub Copilot
│
├── .claude/skills/   → symlinks to .skills/*  (created by setup.sh)
├── .cursor/skills/   → symlinks to .skills/*  (created by setup.sh)
├── .windsurf/skills/ → symlinks to .skills/*  (created by setup.sh)
├── .agents/skills/   → symlinks to .skills/*  (created by setup.sh)
│
├── setup.sh                          # One-command agent setup
├── .env.example                      # Configuration template
├── README.md                         # You are here
└── SETUP.md                          # Detailed setup guide
```

## Contributing

This is early. The skills work but there's a lot of room to make them smarter — better cross-referencing, smarter deduplication, handling larger vaults, new ingest sources. If you've been thinking about this problem or have a workflow that could be a skill, PRs are welcome.

### Adding a new skill

1. Create a folder in `.skills/your-skill-name/`
2. Add a `SKILL.md` with YAML frontmatter (`name`, `description`) and markdown instructions
3. Run `bash setup.sh` to symlink into all agent directories
4. Test with your agent by saying something that matches the description

See `.skills/skill-creator/SKILL.md` for the full guide on writing effective skills.
