"""
Seed script — populates CompeteIQ with 4 realistic pipeline runs spread
over the past 2 weeks so the dashboard has data to display immediately.

Run once:
    python scripts/seed_demo_data.py

Safe to re-run — skips runs whose run_id already exists in SQLite.
No API keys required (data is hard-coded realistic samples).
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

# Ensure project root is on the path when called from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from memory.semantic import SemanticMemory
from models.schemas import Briefing, RunResult, Signal
from tools.formatter import BriefingFormatter

# ---------------------------------------------------------------------------
# Seed data — 4 runs, newest first
# ---------------------------------------------------------------------------

_RUNS: list[dict] = [
    # -----------------------------------------------------------------------
    # Run 1 — 2 days ago (most recent)
    # -----------------------------------------------------------------------
    {
        "run_id": "demo-run-001",
        "days_ago": 2,
        "competitors": ["Anthropic", "Google", "Meta", "Mistral", "Perplexity"],
        "executive_summary": (
            "This week saw Anthropic dominate headlines with the Claude 4 launch, "
            "delivering a 1M-token context window that directly challenges GPT-4o on "
            "long-document enterprise workloads. Google responded with aggressive Gemini "
            "2.0 Flash pricing cuts, signalling a commoditisation push targeting cost-"
            "sensitive developers. Meta's Llama 4 Scout open-weight release continues to "
            "compress the value of proprietary base models."
        ),
        "strategic_implications": [
            "Anthropic's 1M context window creates a near-term moat for document intelligence — OpenAI must respond within 1–2 quarters.",
            "Google's 40% price cut signals confidence in infrastructure efficiency gains; margin compression across the sector will accelerate.",
            "Open-weight Llama 4 reduces enterprise dependency on API providers, commoditising base-model capability.",
            "Perplexity's enterprise search push is eroding Bing Copilot market share faster than Microsoft anticipated.",
            "Mistral's EU cloud partnerships represent a regulatory-arbitrage play for GDPR-sensitive verticals.",
        ],
        "signals_to_watch": [
            "Anthropic enterprise pricing announcement expected Q3 2025 — could reshape Claude 4 adoption trajectory.",
            "Google I/O bundle announcements — Gemini integration into Workspace Premium tiers.",
            "Meta Llama 4 fine-tuning ecosystem: watch for major cloud managed fine-tuning offerings.",
        ],
        "signals": [
            Signal(
                competitor="Anthropic",
                signal_type="product_launch",
                title="Claude 4 Released with 1M Token Context Window",
                summary="Anthropic launched Claude 4 featuring a 1-million token context window and improved reasoning benchmarks, outperforming GPT-4o on long-document tasks. Enterprise tier pricing unchanged from Claude 3.5.",
                impact_assessment="high",
                source_url="https://anthropic.com/news/claude-4",
                confidence=0.97,
                date_detected=datetime.utcnow() - timedelta(days=2),
            ),
            Signal(
                competitor="Google",
                signal_type="pricing_change",
                title="Gemini 2.0 Flash API Price Cut by 40%",
                summary="Google reduced Gemini 2.0 Flash input pricing from $0.075 to $0.04 per 1M tokens. Output pricing dropped from $0.30 to $0.15, targeting cost-sensitive developer segments.",
                impact_assessment="high",
                source_url="https://cloud.google.com/blog/gemini-pricing-update",
                confidence=0.95,
                date_detected=datetime.utcnow() - timedelta(days=3),
            ),
            Signal(
                competitor="Meta",
                signal_type="research_release",
                title="Llama 4 Scout Open-Weight Model Released",
                summary="Meta released Llama 4 Scout with 17B active parameters (109B total via MoE) under a permissive licence allowing commercial fine-tuning without royalties.",
                impact_assessment="medium",
                source_url="https://llama.meta.com/llama4",
                confidence=0.93,
                date_detected=datetime.utcnow() - timedelta(days=4),
            ),
            Signal(
                competitor="Perplexity",
                signal_type="market_expansion",
                title="Perplexity Launches Enterprise Search for Fortune 500",
                summary="Perplexity announced an enterprise-tier product with SSO, audit logs, and private data connectors, directly competing with Microsoft Copilot and Google Enterprise Search.",
                impact_assessment="medium",
                source_url="https://perplexity.ai/enterprise",
                confidence=0.88,
                date_detected=datetime.utcnow() - timedelta(days=2),
            ),
            Signal(
                competitor="Mistral",
                signal_type="partnership",
                title="Mistral Signs Multi-Year Agreement with OVHcloud",
                summary="Mistral AI announced a strategic partnership with OVHcloud to offer Mistral models via EU-sovereign infrastructure, positioning for GDPR-compliant enterprise deployments.",
                impact_assessment="medium",
                source_url="https://mistral.ai/news/ovhcloud-partnership",
                confidence=0.91,
                date_detected=datetime.utcnow() - timedelta(days=5),
            ),
            Signal(
                competitor="Google",
                signal_type="product_launch",
                title="Gemini 2.5 Pro Achieves State-of-the-Art on MMLU",
                summary="Google DeepMind published results showing Gemini 2.5 Pro achieving 92.3% on MMLU, surpassing the previous record. Model available in limited preview for select API customers.",
                impact_assessment="high",
                source_url="https://deepmind.google/research/gemini-2-5",
                confidence=0.94,
                date_detected=datetime.utcnow() - timedelta(days=1),
            ),
        ],
    },
    # -----------------------------------------------------------------------
    # Run 2 — 7 days ago
    # -----------------------------------------------------------------------
    {
        "run_id": "demo-run-002",
        "days_ago": 7,
        "competitors": ["Anthropic", "Google", "Meta", "Mistral", "Perplexity"],
        "executive_summary": (
            "The competitive landscape shifted mid-week as Anthropic raised a $2.5B Series E "
            "at a $25B valuation, signalling sustained investor confidence in frontier model "
            "development. Google's Project Astra multimodal demo previewed real-time visual "
            "understanding capabilities not yet available in production APIs. Meta quietly "
            "released a Llama 3.3 patch addressing key safety benchmark regressions."
        ),
        "strategic_implications": [
            "Anthropic's $25B valuation and Amazon backing gives it a 2-year runway to compete at frontier scale.",
            "Google Astra's real-time vision capabilities — if productised — would leapfrog current GPT-4V response latencies.",
            "Meta's safety patch suggests internal pressure from enterprise customers blocking Llama 3 adoption.",
            "Perplexity's daily active user growth (40% MoM) is outpacing traditional search engines among 18-34 demographics.",
        ],
        "signals_to_watch": [
            "Google Astra API availability timeline — could shift enterprise multimodal vendor decisions in Q3.",
            "Anthropic Series E use-of-proceeds focus: training infrastructure vs. enterprise sales buildout.",
            "Mistral Le Chat consumer app growth — potential threat to Perplexity's consumer positioning.",
        ],
        "signals": [
            Signal(
                competitor="Anthropic",
                signal_type="market_expansion",
                title="Anthropic Raises $2.5B Series E at $25B Valuation",
                summary="Anthropic closed a $2.5B funding round led by Amazon with participation from Google, valuing the company at $25B. Funds earmarked for next-generation model training and enterprise go-to-market expansion.",
                impact_assessment="high",
                source_url="https://anthropic.com/news/series-e",
                confidence=0.99,
                date_detected=datetime.utcnow() - timedelta(days=7),
            ),
            Signal(
                competitor="Google",
                signal_type="product_launch",
                title="Project Astra Multimodal Real-Time Vision Demo",
                summary="Google demoed Project Astra processing live video input with sub-500ms latency, identifying objects and answering contextual questions in real time. No production API date announced.",
                impact_assessment="high",
                source_url="https://deepmind.google/research/project-astra",
                confidence=0.87,
                date_detected=datetime.utcnow() - timedelta(days=8),
            ),
            Signal(
                competitor="Meta",
                signal_type="product_launch",
                title="Llama 3.3 Safety Patch Released",
                summary="Meta pushed a silent update to Llama 3.3 addressing safety benchmark regressions flagged in independent red-teaming reports. Updated weights available on HuggingFace.",
                impact_assessment="low",
                source_url="https://huggingface.co/meta-llama",
                confidence=0.82,
                date_detected=datetime.utcnow() - timedelta(days=9),
            ),
            Signal(
                competitor="Perplexity",
                signal_type="market_expansion",
                title="Perplexity Hits 100M Monthly Active Users",
                summary="Perplexity AI announced crossing 100M monthly active users, with 40% month-over-month growth driven by mobile app adoption in the 18-34 demographic. Monetisation via Pro subscriptions at $20/month.",
                impact_assessment="medium",
                source_url="https://perplexity.ai/blog/100m-users",
                confidence=0.96,
                date_detected=datetime.utcnow() - timedelta(days=7),
            ),
            Signal(
                competitor="Mistral",
                signal_type="product_launch",
                title="Mistral Le Chat Consumer App Reaches 5M Downloads",
                summary="Mistral's consumer chat app Le Chat surpassed 5M downloads across iOS and Android, positioning the company as a direct-to-consumer competitor alongside ChatGPT and Claude.ai.",
                impact_assessment="medium",
                source_url="https://mistral.ai/news/le-chat-milestone",
                confidence=0.89,
                date_detected=datetime.utcnow() - timedelta(days=8),
            ),
        ],
    },
    # -----------------------------------------------------------------------
    # Run 3 — 10 days ago
    # -----------------------------------------------------------------------
    {
        "run_id": "demo-run-003",
        "days_ago": 10,
        "competitors": ["Anthropic", "Google", "Meta", "Mistral", "Perplexity"],
        "executive_summary": (
            "Anthropic's expanded AWS partnership and native Bedrock integration mark a "
            "significant enterprise distribution milestone, while Google's Gemini integration "
            "into all Workspace SKUs at no additional cost puts pressure on standalone AI "
            "assistant pricing. Mistral's new code-specialised model Codestral 2 directly "
            "targets GitHub Copilot's developer workflow positioning."
        ),
        "strategic_implications": [
            "Anthropic-AWS native Bedrock integration removes the primary barrier to Claude adoption in regulated enterprise verticals.",
            "Google bundling Gemini into Workspace at no extra cost sets a price floor that undercuts most standalone AI assistant products.",
            "Mistral Codestral 2 benchmarks suggesting parity with GPT-4o on HumanEval could erode GitHub Copilot premium positioning.",
            "Meta's open-weight strategy is forcing closed-model providers to compete on trust and integration depth rather than raw capability.",
        ],
        "signals_to_watch": [
            "GitHub Copilot pricing response to Codestral 2 benchmarks — a price cut would confirm competitive pressure.",
            "Anthropic Bedrock GA date — will determine enterprise procurement cycle timing.",
            "Google Workspace Gemini adoption rates — bundling success metric for Q2 earnings call.",
        ],
        "signals": [
            Signal(
                competitor="Anthropic",
                signal_type="partnership",
                title="Anthropic Expands AWS Bedrock Partnership to Native Integration",
                summary="Anthropic and Amazon deepened their partnership with Claude becoming a first-class citizen on AWS Bedrock, including native IAM integration, VPC support, and SOC2/HIPAA compliance certifications.",
                impact_assessment="high",
                source_url="https://aws.amazon.com/blogs/machine-learning/anthropic-bedrock",
                confidence=0.96,
                date_detected=datetime.utcnow() - timedelta(days=10),
            ),
            Signal(
                competitor="Google",
                signal_type="pricing_change",
                title="Gemini Bundled into All Google Workspace SKUs at No Extra Cost",
                summary="Google announced Gemini AI features will be included in all Workspace Business and Enterprise tiers at no additional cost starting Q2, effective immediately for US customers.",
                impact_assessment="high",
                source_url="https://workspace.google.com/blog/gemini-bundled",
                confidence=0.98,
                date_detected=datetime.utcnow() - timedelta(days=11),
            ),
            Signal(
                competitor="Mistral",
                signal_type="product_launch",
                title="Codestral 2 Code Model Released with HumanEval Parity",
                summary="Mistral released Codestral 2, a code-specialised model achieving 84.2% on HumanEval benchmark, matching GPT-4o's reported score. Available via API and self-hosted deployment.",
                impact_assessment="medium",
                source_url="https://mistral.ai/news/codestral-2",
                confidence=0.91,
                date_detected=datetime.utcnow() - timedelta(days=10),
            ),
            Signal(
                competitor="Meta",
                signal_type="research_release",
                title="Meta AI Research Publishes MEGALODON Architecture Paper",
                summary="Meta AI published research on MEGALODON, a linear-complexity sequence model architecture that matches Transformer performance at 1/3 the inference cost at 7B parameter scale.",
                impact_assessment="medium",
                source_url="https://arxiv.org/meta-megalodon",
                confidence=0.85,
                date_detected=datetime.utcnow() - timedelta(days=12),
            ),
        ],
    },
    # -----------------------------------------------------------------------
    # Run 4 — 14 days ago
    # -----------------------------------------------------------------------
    {
        "run_id": "demo-run-004",
        "days_ago": 14,
        "competitors": ["Anthropic", "Google", "Meta", "Mistral", "Perplexity"],
        "executive_summary": (
            "Two weeks ago the market was dominated by executive movement and early signals "
            "of a pricing war. Google's hire of Demis Hassabis's former right hand from "
            "DeepMind for its cloud AI division signalled a renewed push to close the "
            "enterprise gap. Perplexity's Series C and aggressive hiring suggest an "
            "imminent enterprise product expansion. Meta's internal memo leak on Llama 5 "
            "timelines created significant market speculation."
        ),
        "strategic_implications": [
            "Ilya Sutskever's SSI stealth mode combined with top-tier talent acquisitions from OpenAI represents a longer-term frontier threat.",
            "Perplexity's Series C at $3B valuation signals VC confidence in search disruption despite Google's defensive bundling moves.",
            "Meta Llama 5 timeline leak (Q4 2025) would compress the window for proprietary model differentiation.",
            "Anthropic's Constitutional AI v2 paper release strengthens its regulatory positioning ahead of EU AI Act enforcement.",
        ],
        "signals_to_watch": [
            "SSI (Ilya Sutskever) first public model announcement — could reshape frontier model competitive dynamics.",
            "Perplexity Series C deployment: sales team buildout vs. model fine-tuning investment.",
            "Meta Llama 5 confirmation or denial — silence will increase speculation pressure.",
        ],
        "signals": [
            Signal(
                competitor="Google",
                signal_type="executive_move",
                title="Google Cloud Hires DeepMind VP to Lead Enterprise AI Division",
                summary="Google Cloud announced the hire of Dr. Sarah Chen (former VP Research at DeepMind) as SVP of Enterprise AI, signalling a push to accelerate enterprise Gemini deployments and close the gap with Microsoft Copilot.",
                impact_assessment="medium",
                source_url="https://blog.google/technology/ai/enterprise-ai-leadership",
                confidence=0.93,
                date_detected=datetime.utcnow() - timedelta(days=14),
            ),
            Signal(
                competitor="Perplexity",
                signal_type="market_expansion",
                title="Perplexity Closes $250M Series C at $3B Valuation",
                summary="Perplexity AI raised $250M in Series C funding led by SoftBank Vision Fund, valuing the company at $3B. CEO cited plans for enterprise product expansion and international expansion into Japan and South Korea.",
                impact_assessment="high",
                source_url="https://perplexity.ai/blog/series-c",
                confidence=0.97,
                date_detected=datetime.utcnow() - timedelta(days=14),
            ),
            Signal(
                competitor="Meta",
                signal_type="research_release",
                title="Internal Memo Leak Suggests Llama 5 Targeting Q4 2025",
                summary="An internal Meta memo circulated online suggesting Llama 5 is targeting a Q4 2025 release with a claimed 10x efficiency improvement over Llama 4. Meta has not confirmed or denied the timeline.",
                impact_assessment="medium",
                source_url="https://theverge.com/meta-llama5-leak",
                confidence=0.65,
                date_detected=datetime.utcnow() - timedelta(days=13),
            ),
            Signal(
                competitor="Anthropic",
                signal_type="research_release",
                title="Anthropic Publishes Constitutional AI v2 Technical Report",
                summary="Anthropic released Constitutional AI v2, detailing updated RLHF training methodology with measurable reductions in harmful outputs. Report timed ahead of EU AI Act enforcement deadline.",
                impact_assessment="medium",
                source_url="https://anthropic.com/research/constitutional-ai-v2",
                confidence=0.94,
                date_detected=datetime.utcnow() - timedelta(days=15),
            ),
            Signal(
                competitor="Mistral",
                signal_type="pricing_change",
                title="Mistral Drops Mixtral 8x22B API Price by 50%",
                summary="Mistral halved the API price of Mixtral 8x22B to $0.90 per 1M input tokens, directly undercutting GPT-4o-mini and Claude Haiku on cost-per-token for reasoning-class tasks.",
                impact_assessment="medium",
                source_url="https://mistral.ai/news/pricing-update-april",
                confidence=0.92,
                date_detected=datetime.utcnow() - timedelta(days=14),
            ),
        ],
    },
]


# ---------------------------------------------------------------------------
# Seeding logic
# ---------------------------------------------------------------------------

def seed() -> None:
    """Insert all demo runs, signals, and briefings into the SQLite database."""
    db = SemanticMemory()
    existing_runs = {r["run_id"] for r in db.get_runs(limit=100)}

    seeded = 0
    for run_data in _RUNS:
        run_id = run_data["run_id"]
        if run_id in existing_runs:
            print(f"  skip  {run_id} (already exists)")
            continue

        run_date = datetime.utcnow() - timedelta(days=run_data["days_ago"])
        signals: list[Signal] = run_data["signals"]

        # Build briefing markdown
        briefing_md = BriefingFormatter.format_briefing(
            signals=signals,
            competitors=run_data["competitors"],
            run_id=run_id,
            executive_summary=run_data["executive_summary"],
            strategic_implications=run_data["strategic_implications"],
            signals_to_watch=run_data["signals_to_watch"],
            run_date=run_date,
        )

        # Persist signals
        db.save_signals(signals, run_id=run_id)

        # Persist briefing
        from models.schemas import Briefing
        db.save_briefing(Briefing(
            run_id=run_id,
            created_at=run_date,
            competitors_monitored=run_data["competitors"],
            signal_count=len(signals),
            content=briefing_md,
        ))

        # Persist run record
        db.save_run(RunResult(
            run_id=run_id,
            started_at=run_date - timedelta(minutes=3),
            completed_at=run_date,
            competitors=run_data["competitors"],
            signal_count=len(signals),
            error_count=0,
            briefing_id=run_id,
            success=True,
        ))

        print(f"  seeded {run_id} — {len(signals)} signals — {run_date.strftime('%b %d, %Y')}")
        seeded += 1

    print(f"\nDone. {seeded} new run(s) seeded, {len(existing_runs)} already existed.")


if __name__ == "__main__":
    print("Seeding CompeteIQ demo data...\n")
    seed()
