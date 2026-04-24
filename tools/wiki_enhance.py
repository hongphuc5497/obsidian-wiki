#!/usr/bin/env python3
"""
wiki_enhance.py — GoClaw-inspired enhancements for the Obsidian knowledge wiki.

Subcommands (easiest → most impactful):
  index          SHA-256 content hashes — detect changed files for incremental work
  links          Extract [[wikilink]] contexts (~50 chars) for richer backlinks
  suggest-links  Use QMD embeddings to suggest semantically related links
  search         Cross-source unified search (wiki + raw + conversations)
  suggest-tags   Suggest canonical tags from taxonomy.md via embedding similarity
  classify       Auto-classify notes into concept/reference/project/synthesis types

Usage:
  python3 wiki_enhance.py index
  python3 wiki_enhance.py links
  python3 wiki_enhance.py suggest-links [--threshold 0.72] [--top-k 5]
  python3 wiki_enhance.py search "query" [--sources wiki,raw]
  python3 wiki_enhance.py suggest-tags <filepath>
  python3 wiki_enhance.py classify [--dry-run]
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".obsidian-wiki" / "config"
STATE_FILENAME = ".wiki_enhance.json"
TAXONOMY_PATH = "_meta/taxonomy.md"

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
TAG_LINE_RE = re.compile(r"tags:\s*\[(.*?)\]")

# ── Helpers ─────────────────────────────────────────────────────────────────


def read_config() -> Tuple[Path, Optional[Path]]:
    """Read OBSIDIAN_VAULT_PATH and OBSIDIAN_WIKI_REPO from config."""
    vault_path: Optional[Path] = None
    repo_path: Optional[Path] = None
    if CONFIG_PATH.exists():
        for line in CONFIG_PATH.read_text().splitlines():
            if line.startswith("OBSIDIAN_VAULT_PATH="):
                vault_path = Path(line.split("=", 1)[1].strip().strip('"'))
            elif line.startswith("OBSIDIAN_WIKI_REPO="):
                repo_path = Path(line.split("=", 1)[1].strip().strip('"'))
    # Fallback: read .env in current repo
    if vault_path is None:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OBSIDIAN_VAULT_PATH="):
                    vault_path = Path(line.split("=", 1)[1].strip().strip('"'))
    if vault_path is None or not vault_path.exists():
        print("error: OBSIDIAN_VAULT_PATH not found in ~/.obsidian-wiki/config or .env", file=sys.stderr)
        sys.exit(1)
    return vault_path, repo_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def parse_frontmatter(text: str) -> Tuple[Optional[dict], str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    try:
        # Simple YAML-like parsing for tags, title, category
        fm = {}
        for line in m.group(1).splitlines():
            if ":" in line and not line.strip().startswith("#"):
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"').strip("'")
        return fm, text[m.end():]
    except Exception:
        return None, text


def extract_tags_from_frontmatter(fm: dict) -> List[str]:
    raw = fm.get("tags", "")
    if raw.startswith("["):
        return [t.strip().strip('"').strip("'") for t in raw[1:-1].split(",") if t.strip()]
    return []


def load_state(vault: Path) -> dict:
    path = vault / STATE_FILENAME
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "files": {}, "last_run": None}


def save_state(vault: Path, state: dict) -> None:
    path = vault / STATE_FILENAME
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def list_md_files(vault: Path) -> List[Path]:
    """List all .md files excluding _archives, .obsidian, and hidden dirs."""
    files = []
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault)
        parts = rel.parts
        if any(part.startswith(".") or part == "_archives" for part in parts):
            continue
        files.append(p)
    return sorted(files)


# ═════════════════════════════════════════════════════════════════════════════
# 1. INDEX — Content hash incremental tracking
# ═════════════════════════════════════════════════════════════════════════════


def cmd_index(vault: Path, args: argparse.Namespace) -> None:
    state = load_state(vault)
    files = list_md_files(vault)
    changed: List[Path] = []
    unchanged: List[Path] = []
    new: List[Path] = []

    for f in files:
        rel = str(f.relative_to(vault))
        h = sha256_file(f)
        if rel not in state["files"]:
            new.append(f)
            state["files"][rel] = {"hash": h, "indexed_at": None}
        elif state["files"][rel].get("hash") != h:
            changed.append(f)
            state["files"][rel]["hash"] = h
            state["files"][rel]["indexed_at"] = None
        else:
            unchanged.append(f)

    # Mark any removed files
    current_rels = {str(f.relative_to(vault)) for f in files}
    removed = [r for r in state["files"] if r not in current_rels]
    for r in removed:
        del state["files"][r]

    save_state(vault, state)

    print(f"📊  Content Hash Index — {vault.name}")
    print(f"   Total files:     {len(files)}")
    print(f"   New:             {len(new)}")
    print(f"   Changed:         {len(changed)}")
    print(f"   Unchanged:       {len(unchanged)}")
    print(f"   Removed:         {len(removed)}")
    if changed:
        print(f"\n📝  Changed files ({len(changed)}):")
        for f in changed[:20]:
            print(f"      {f.relative_to(vault)}")
        if len(changed) > 20:
            print(f"      ... and {len(changed)-20} more")
    if new:
        print(f"\n✨  New files ({len(new)}):")
        for f in new[:10]:
            print(f"      {f.relative_to(vault)}")
        if len(new) > 10:
            print(f"      ... and {len(new)-10} more")
    print(f"\n💡  Run `qmd update` + `qmd embed` to re-index changed files.")
    print(f"   Or use `wiki_enhance.py suggest-links` to find connections.")


# ═════════════════════════════════════════════════════════════════════════════
# 2. LINKS — Extract wikilink contexts
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class WikilinkMatch:
    target: str
    context: str
    offset: int


def extract_wikilinks(content: str) -> List[WikilinkMatch]:
    matches = []
    for m in WIKILINK_RE.finditer(content):
        target = m.group(1).strip()
        if not target:
            continue
        start = max(m.start() - 25, 0)
        end = min(m.end() + 25, len(content))
        ctx = content[start:end].replace("\n", " ")
        matches.append(WikilinkMatch(target=target, context=ctx, offset=m.start()))
    return matches


def cmd_links(vault: Path, args: argparse.Namespace) -> None:
    state = load_state(vault)
    files = list_md_files(vault)
    total_links = 0
    backlinks: Dict[str, List[Tuple[str, str]]] = {}  # target -> [(source, context)]

    for f in files:
        rel = str(f.relative_to(vault))
        text = f.read_text()
        _, body = parse_frontmatter(text)
        links = extract_wikilinks(body)
        state["files"].setdefault(rel, {})
        state["files"][rel]["links"] = [
            {"target": l.target, "context": l.context, "offset": l.offset}
            for l in links
        ]
        total_links += len(links)
        for l in links:
            backlinks.setdefault(l.target, []).append((rel, l.context))

    state["backlinks"] = {k: [{"from": src, "context": ctx} for src, ctx in v] for k, v in backlinks.items()}
    save_state(vault, state)

    print(f"🔗  Wikilink Context Extraction — {vault.name}")
    print(f"   Files scanned:    {len(files)}")
    print(f"   Links found:      {total_links}")
    print(f"   Unique targets:   {len(backlinks)}")
    print(f"   Backlinks stored: {sum(len(v) for v in backlinks.values())}")

    # Show some examples
    if backlinks:
        print(f"\n📌  Sample backlinks:")
        for target, sources in list(backlinks.items())[:5]:
            print(f"   [[{target}]] ← {len(sources)} reference(s)")
            for src, ctx in sources[:2]:
                print(f"      from {src}: \"{ctx[:60]}...\"")

    print(f"\n💾  Saved to {STATE_FILENAME}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. SUGGEST-LINKS — Semantic auto-link suggestions
# ═════════════════════════════════════════════════════════════════════════════


def qmd_vsearch(query: str, top_k: int = 10, cwd: Optional[Path] = None) -> List[dict]:
    """Run qmd vsearch and parse plain-text output."""
    try:
        result = subprocess.run(
            ["qmd", "vsearch", query, "-n", str(top_k * 2)],
            capture_output=True, text=True, timeout=15, cwd=str(cwd) if cwd else None
        )
        if result.returncode != 0:
            print(f"warning: qmd vsearch failed: {result.stderr[:200]}", file=sys.stderr)
            return []
        return _parse_qmd_vsearch(result.stdout, top_k)
    except subprocess.TimeoutExpired:
        print(f"warning: qmd vsearch timed out for query: {query[:60]}...", file=sys.stderr)
        return []
    except Exception as e:
        print(f"warning: qmd vsearch error: {e}", file=sys.stderr)
        return []


def _parse_qmd_vsearch(text: str, top_k: int) -> List[dict]:
    """Parse qmd vsearch plain-text output into structured results."""
    results = []
    current: Optional[dict] = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("qmd://"):
            if current and current.get("path"):
                results.append(current)
                if len(results) >= top_k:
                    break
            # Parse path line: qmd://collection/path.md:line #docid
            parts = line.split()
            path_part = parts[0] if parts else ""
            # Strip qmd:// prefix and docid suffix
            path_clean = path_part.replace("qmd://", "")
            if " #" in path_clean:
                path_clean = path_clean.split(" #")[0]
            if ":" in path_clean:
                # Remove :line suffix
                path_clean = ":".join(path_clean.split(":")[:-1])
            current = {"path": path_clean, "score": 0.0, "title": "", "snippet": ""}
        elif line.startswith("Title:") and current is not None:
            current["title"] = line[len("Title:"):].strip()
        elif line.startswith("Score:") and current is not None:
            score_str = line[len("Score:"):].strip().rstrip("%")
            try:
                current["score"] = float(score_str) / 100.0
            except ValueError:
                current["score"] = 0.0
        elif line.startswith("@@") and current is not None:
            # Snippet indicator line — next line(s) are the actual snippet
            i += 1
            snippet_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("qmd://"):
                snippet_lines.append(lines[i])
                i += 1
            current["snippet"] = " ".join(snippet_lines)[:200]
            continue  # i already advanced
        i += 1
    if current and current.get("path") and len(results) < top_k:
        results.append(current)
    return results


def existing_links_for_file(state: dict, rel_path: str) -> Set[str]:
    """Get set of already-linked targets for a file."""
    links = state.get("files", {}).get(rel_path, {}).get("links", [])
    return {l["target"] for l in links}


def cmd_suggest_links(vault: Path, args: argparse.Namespace) -> None:
    state = load_state(vault)
    files = list_md_files(vault)
    threshold = args.threshold
    top_k = args.top_k

    # Only process changed/new files (or all if --force)
    if args.force:
        candidates = files
    else:
        candidates = []
        for f in files:
            rel = str(f.relative_to(vault))
            file_state = state.get("files", {}).get(rel, {})
            if file_state.get("hash") and file_state.get("indexed_at") is None:
                candidates.append(f)
        if not candidates:
            print("ℹ️  No changed files detected. Use --force to process all files.")
            return

    print(f"🤖  Auto-Link Suggestions — {vault.name}")
    print(f"   Files to process: {len(candidates)}")
    print(f"   Threshold:        {threshold}")
    print(f"   Top-K:            {top_k}")
    print()

    suggestions: Dict[str, List[Tuple[str, float]]] = {}  # rel_path -> [(target, score)]

    for f in candidates:
        rel = str(f.relative_to(vault))
        text = f.read_text()
        fm, body = parse_frontmatter(text)
        title = fm.get("title", f.stem) if fm else f.stem
        summary = fm.get("summary", "") if fm else ""

        # Build query from title + summary + first 500 chars of body
        query = f"{title}. {summary}" if summary else title
        if len(query) < 50:
            query = f"{query}. {body[:500]}"

        existing = existing_links_for_file(state, rel)
        existing.add(f.stem)  # Don't suggest self

        results = qmd_vsearch(query, top_k=top_k + len(existing), cwd=vault)
        found = []
        for r in results[:top_k + len(existing)]:
            target_path = r.get("path", "")
            target_name = Path(target_path).stem
            score = r.get("score", 0.0)
            if target_name in existing or target_name == f.stem:
                continue
            if score >= threshold:
                found.append((target_name, score))
            if len(found) >= top_k:
                break

        if found:
            suggestions[rel] = found
            print(f"   📝 {rel}")
            for target, score in found:
                print(f"      → [[{target}]]  (score: {score:.3f})")

        # Mark file as processed so we don't re-suggest on next run
        state["files"].setdefault(rel, {})
        state["files"][rel]["indexed_at"] = True

    # Save suggestions to state
    state["link_suggestions"] = {
        rel: [{"target": t, "score": s} for t, s in founds]
        for rel, founds in suggestions.items()
    }
    save_state(vault, state)

    total = sum(len(v) for v in suggestions.values())
    print(f"\n✅  {total} suggestions across {len(suggestions)} files")
    print(f"   💾 Saved to {STATE_FILENAME} under 'link_suggestions'")


# ═════════════════════════════════════════════════════════════════════════════
# 4. SEARCH — Cross-source unified search
# ═════════════════════════════════════════════════════════════════════════════


def cmd_search(vault: Path, args: argparse.Namespace) -> None:
    sources = args.sources.split(",") if args.sources else ["wiki"]
    query = args.query
    top_n = args.n

    print(f"🔍  Unified Search — \"{query}\"")
    print(f"   Sources: {', '.join(sources)}")
    print()

    all_results: List[Tuple[str, float, dict]] = []  # (source, normalized_score, result)

    # Source 1: Wiki pages (QMD hybrid search)
    if "wiki" in sources:
        try:
            result = subprocess.run(
                ["qmd", "query", query, "-n", str(top_n * 2), "--json", "--collection=knowledge-base-wiki"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                wiki_results = data if isinstance(data, list) else data.get("results", [])
                if wiki_results:
                    max_score = max(r.get("score", 0.0) for r in wiki_results)
                    for r in wiki_results:
                        score = (r.get("score", 0.0) / max_score * 0.5) if max_score > 0 else 0.0
                        all_results.append(("wiki", score, r))
        except Exception as e:
            print(f"warning: wiki search error: {e}", file=sys.stderr)

    # Source 2: Raw sources (files in _raw/)
    if "raw" in sources:
        raw_dir = vault / "_raw"
        if raw_dir.exists():
            # Simple keyword grep + recency boost
            raw_files = sorted(raw_dir.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
            matches = []
            q_lower = query.lower()
            for rf in raw_files[:100]:
                text = rf.read_text().lower()
                score = 0.0
                for term in q_lower.split():
                    if term in text:
                        score += 0.15
                if score > 0:
                    matches.append((rf, score, text[:200]))
            matches.sort(key=lambda x: x[1], reverse=True)
            if matches:
                max_score = max(m[1] for m in matches)
                for rf, score, snippet in matches[:top_n]:
                    norm_score = (score / max_score * 0.3) if max_score > 0 else 0.0
                    all_results.append(("raw", norm_score, {
                        "path": str(rf.relative_to(vault)),
                        "title": rf.stem,
                        "snippet": snippet,
                    }))

    # Merge, dedup by path, sort by score
    seen_paths = set()
    deduped = []
    all_results.sort(key=lambda x: x[1], reverse=True)
    for source, score, result in all_results:
        path = result.get("path", result.get("file", ""))
        if path in seen_paths:
            continue
        seen_paths.add(path)
        deduped.append((source, score, result))

    print(f"   Total results: {len(deduped[:top_n])}")
    print()
    for i, (source, score, r) in enumerate(deduped[:top_n], 1):
        title = r.get("title", Path(r.get("path", "")).stem)
        path = r.get("path", r.get("file", ""))
        badge = {"wiki": "📄", "raw": "📥", "chat": "💬"}.get(source, "❓")
        print(f"{i}. {badge} [{source.upper()}] {title}")
        print(f"   Path: {path}")
        if "snippet" in r:
            snippet = r["snippet"].replace("\n", " ")[:120]
            print(f"   {snippet}...")
        print(f"   Score: {score:.3f}")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# 5. SUGGEST-TAGS — Auto-tag from taxonomy
# ═════════════════════════════════════════════════════════════════════════════


def load_taxonomy_tags(vault: Path) -> List[str]:
    """Extract canonical tags from taxonomy.md."""
    tax_path = vault / TAXONOMY_PATH
    if not tax_path.exists():
        return []
    text = tax_path.read_text()
    tags = set()
    # Find lines like `| `tag-name` | ... |`
    for line in text.splitlines():
        if "`" in line and "|" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if p.startswith("`") and p.endswith("`") and " " not in p.strip("`"):
                    tag = p.strip("`").strip()
                    if tag and not tag.startswith("visibility/"):
                        tags.add(tag)
    return sorted(tags)


def cmd_suggest_tags(vault: Path, args: argparse.Namespace) -> None:
    target = vault / args.filepath
    if not target.exists():
        print(f"error: file not found: {target}", file=sys.stderr)
        sys.exit(1)

    tags = load_taxonomy_tags(vault)
    if not tags:
        print("warning: no tags found in taxonomy.md", file=sys.stderr)
        return

    text = target.read_text()
    fm, body = parse_frontmatter(text)
    current_tags = extract_tags_from_frontmatter(fm) if fm else []
    title = fm.get("title", target.stem) if fm else target.stem

    # Query QMD with title + short body preview for similar pages
    preview = body[:150].replace("\n", " ").strip()
    query = f"{title}. {preview}" if preview else title
    results = qmd_vsearch(query, top_k=10, cwd=vault)

    # Collect tags from top similar pages
    tag_scores: Dict[str, float] = {}
    for r in results:
        sim_path = vault / r.get("path", "")
        if not sim_path.exists() or sim_path == target:
            continue
        sim_text = sim_path.read_text()
        sim_fm, _ = parse_frontmatter(sim_text)
        if sim_fm:
            sim_tags = extract_tags_from_frontmatter(sim_fm)
            score = r.get("score", 0.0)
            for t in sim_tags:
                if t not in current_tags and t in tags:
                    tag_scores[t] = max(tag_scores.get(t, 0.0), score)

    # Sort by score
    suggested = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"🏷️  Tag Suggestions for {args.filepath}")
    print(f"   Current tags: {current_tags or '(none)'}")
    print(f"   Taxonomy tags available: {len(tags)}")
    print()
    if suggested:
        print("   Suggested tags (from similar pages):")
        for tag, score in suggested[:8]:
            print(f"      + {tag}  (similarity: {score:.3f})")
    else:
        print("   No strong tag suggestions found.")
    print()
    print("   💡 To apply: edit frontmatter tags: [...]")


# ═════════════════════════════════════════════════════════════════════════════
# 6. CLASSIFY — Auto-classify document type
# ═════════════════════════════════════════════════════════════════════════════


# Heuristic classification based on frontmatter + content
DOC_TYPE_RULES = [
    ("project", ["projects/"], ["project", "application", "framework", "cli-tool"]),
    ("concept", ["concepts/"], ["concept", "pattern", "architecture"]),
    ("entity", ["entities/"], ["entity", "person", "tool", "library"]),
    ("skill", ["skills/"], ["skill", "how-to", "technique", "procedure"]),
    ("reference", ["references/"], ["reference", "spec", "api", "config"]),
    ("synthesis", ["synthesis/"], ["synthesis", "analysis", "cross-cutting"]),
]


def classify_document(path: Path, text: str, fm: Optional[dict]) -> str:
    rel = str(path)
    category = fm.get("category", "") if fm else ""

    # Path-based classification (strong signal)
    for dtype, prefixes, _ in DOC_TYPE_RULES:
        for prefix in prefixes:
            if prefix in rel:
                return dtype

    # Category-based fallback
    cat_lower = category.lower()
    for dtype, _, cat_hints in DOC_TYPE_RULES:
        for hint in cat_hints:
            if hint in cat_lower:
                return dtype

    # Content-based fallback: check for common patterns
    body_lower = text.lower()
    if "how to" in body_lower or "steps:" in body_lower or "usage:" in body_lower:
        return "skill"
    if "pattern" in body_lower or "architecture" in body_lower or "design decision" in body_lower:
        return "concept"
    if "project" in body_lower and ("tech stack" in body_lower or "dependencies" in body_lower):
        return "project"

    return "note"  # default


def cmd_classify(vault: Path, args: argparse.Namespace) -> None:
    files = list_md_files(vault)
    changes = []

    for f in files:
        text = f.read_text()
        fm, body = parse_frontmatter(text)
        if fm is None:
            continue
        detected = classify_document(f, text, fm)
        current_type = fm.get("doc_type", "")

        if current_type != detected:
            changes.append((f, current_type, detected))
            if not args.dry_run:
                # Update frontmatter
                lines = text.splitlines()
                in_fm = False
                fm_end = 0
                for i, line in enumerate(lines):
                    if line.strip() == "---" and i == 0:
                        in_fm = True
                    elif line.strip() == "---" and in_fm:
                        fm_end = i
                        break
                # Find where to insert doc_type (alphabetically among existing fields, or after category)
                insert_idx = fm_end
                for i in range(1, fm_end):
                    if lines[i].startswith("category:"):
                        insert_idx = i + 1
                        break
                lines.insert(insert_idx, f"doc_type: {detected}")
                f.write_text("\n".join(lines))

    print(f"📂  Document Classification — {vault.name}")
    print(f"   Files scanned: {len(files)}")
    print(f"   Changes:       {len(changes)}")
    if changes:
        print()
        for f, old, new in changes[:20]:
            old_str = old or "(none)"
            print(f"   {f.relative_to(vault)}: {old_str} → {new}")
        if len(changes) > 20:
            print(f"   ... and {len(changes)-20} more")
    if args.dry_run:
        print(f"\n   🚫  Dry-run mode — no files modified. Use without --dry-run to apply.")
    else:
        print(f"\n   ✅  Applied {len(changes)} classifications.")


# ═════════════════════════════════════════════════════════════════════════════
# CLI Entrypoint
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="wiki_enhance.py — Enhance your Obsidian knowledge wiki")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = subparsers.add_parser("index", help="Content hash incremental tracking")

    # links
    p_links = subparsers.add_parser("links", help="Extract wikilink contexts")

    # suggest-links
    p_suggest = subparsers.add_parser("suggest-links", help="Suggest semantically related wikilinks")
    p_suggest.add_argument("--threshold", type=float, default=0.72, help="Similarity threshold (default: 0.72)")
    p_suggest.add_argument("--top-k", type=int, default=5, help="Max suggestions per file (default: 5)")
    p_suggest.add_argument("--force", action="store_true", help="Process all files, not just changed ones")

    # search
    p_search = subparsers.add_parser("search", help="Cross-source unified search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--sources", default="wiki,raw", help="Comma-separated sources: wiki,raw,chat (default: wiki,raw)")
    p_search.add_argument("-n", type=int, default=10, help="Max results (default: 10)")

    # suggest-tags
    p_tags = subparsers.add_parser("suggest-tags", help="Suggest canonical tags from taxonomy")
    p_tags.add_argument("filepath", help="Relative path to markdown file in vault")

    # classify
    p_classify = subparsers.add_parser("classify", help="Auto-classify document types")
    p_classify.add_argument("--dry-run", action="store_true", help="Show changes without applying")

    args = parser.parse_args()
    vault, _ = read_config()

    commands = {
        "index": cmd_index,
        "links": cmd_links,
        "suggest-links": cmd_suggest_links,
        "search": cmd_search,
        "suggest-tags": cmd_suggest_tags,
        "classify": cmd_classify,
    }

    commands[args.command](vault, args)


if __name__ == "__main__":
    main()
