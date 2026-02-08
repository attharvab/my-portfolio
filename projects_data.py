# projects_data.py
# Data only. No FastAPI code.

PROJECTS = [
    {
        "slug": "ford-corporate-finance-valuation",
        "title": "Ford Motor Company – Corporate Finance & Valuation",
        "one_liner": "Modeled how capital structure, working capital discipline, and buybacks flow through to equity value.",
        "summary": (
            "I analyzed Ford through a corporate finance lens rather than as a short-term stock idea, focusing on "
            "what management can control to improve free cash flow and shareholder returns."
        ),
        "sections": [
            {
                "heading": "Overview",
                "body": (
                    "In this project, I looked at Ford through a corporate finance lens rather than as a short-term stock idea. "
                    "The goal was to understand how value is actually created inside a large, complex company like Ford, and what "
                    "management can realistically do to improve free cash flow and shareholder returns.\n"
                    "I built a valuation model for Ford and then tested how changes in capital structure, working capital efficiency, "
                    "and share repurchases flow through to equity value."
                ),
            },
            {
                "heading": "Project Motivation",
                "body": (
                    "I wanted to move past the usual question of whether Ford is cheap or expensive. Instead, I focused on what management "
                    "can directly control.\n"
                    "For a company of this size, big strategic changes are hard and slow. But improvements in working capital discipline, "
                    "capital allocation, and buyback consistency can quietly compound over time. I wanted to see how much impact those "
                    "levers could actually have when modeled properly."
                ),
            },
            {
                "heading": "Technical Details",
                "body": (
                    "I built an internally consistent valuation model that tied Ford’s value to its operating performance and cost of capital. "
                    "I paid close attention to Ford’s capital structure, especially the role of its credit business, since most of the company’s "
                    "debt sits there and cannot be ignored without distorting the analysis.\n"
                    "I compared Ford’s leverage profile to peers like General Motors to anchor assumptions in reality. I then modeled working capital "
                    "changes by adjusting accounts receivable and accounts payable days toward peer and historical benchmarks, and quantified how those "
                    "changes affect free cash flow.\n"
                    "Finally, I built a structured share buyback plan and translated it directly into EPS growth and equity value impact, rather than "
                    "treating buybacks as a vague positive."
                ),
            },
            {
                "heading": "Technical Challenges and Solutions",
                "body": (
                    "Ford’s business mix is complicated\n"
                    "Ford’s automotive and credit businesses are tightly linked. Instead of stripping out the credit arm to make the model cleaner, I kept "
                    "it fully embedded so the valuation reflected economic reality.\n\n"
                    "Working capital targets need to be realistic\n"
                    "Aggressive assumptions can look good in a model but fail in practice. I used peer benchmarks and Ford’s own pre-COVID history to keep "
                    "recommendations achievable.\n\n"
                    "Turning finance decisions into valuation impact\n"
                    "Rather than describing changes qualitatively, I explicitly modeled how improvements in working capital and reductions in share count flow "
                    "through to free cash flow, EPS, and price per share."
                ),
            },
            {
                "heading": "Results and Impact",
                "body": (
                    "The analysis led to three clear takeaways.\n"
                    "First, reducing accounts receivable days toward peer levels meaningfully increased free cash flow, improving liquidity without changing operations.\n"
                    "Second, extending accounts payable days modestly provided an additional cash flow lift without stressing supplier relationships.\n"
                    "Third, a disciplined annual buyback program steadily reduced the share count and drove EPS growth over time.\n"
                    "After incorporating these changes, the model implied roughly 26% upside to Ford’s equity value, driven entirely by financial execution rather "
                    "than aggressive growth assumptions."
                ),
            },
            {
                "heading": "Conclusion",
                "body": (
                    "This project reinforced something I strongly believe in: value creation is often about execution and discipline, not big strategic shifts.\n"
                    "Ford did not need a new product cycle or a dramatic turnaround to improve shareholder outcomes. By tightening working capital, maintaining a sensible "
                    "capital structure, and committing to consistent buybacks, the company could create value steadily over time. Small financial decisions, when repeated "
                    "year after year, compound in a meaningful way."
                ),
            },
        ],
    },

    {
        "slug": "wayfair-dcf-comps-damodaran",
        "title": "Equity Valuation – Wayfair (DCF & Comparable Analysis, Damodaran Framework)",
        "one_liner": "End-to-end intrinsic and relative valuation to test if price reflected fundamentals or narrative.",
        "summary": (
            "I valued Wayfair as a business, not a ticker, combining a bottom-up DCF with EV/Sales comps to test whether "
            "market expectations on growth and margins were realistic."
        ),
        "sections": [
            {
                "heading": "Overview",
                "body": (
                    "This project was an end-to-end valuation of Wayfair Inc., focused on one core question: Does the current stock price reflect the business reality?\n"
                    "I approached Wayfair as a business, not a ticker. The analysis combined a bottom-up DCF with a relative valuation framework to understand how growth, "
                    "margins, reinvestment, and risk actually translate into value. Both approaches independently pointed to the same conclusion: that the stock price was "
                    "being driven more by narrative than fundamentals."
                ),
            },
            {
                "heading": "Project Motivation",
                "body": (
                    "Wayfair’s stock had moved sharply upward over a short period, largely on optimism around competitor bankruptcies and expectations of strong margin expansion. "
                    "At the same time, the U.S. home furnishings industry remains mature, competitive, and structurally constrained.\n"
                    "I wanted to test whether the optimism embedded in the stock price was realistic given:\n"
                    "Industry growth of roughly 3–4%\n"
                    "Wayfair’s declining market share\n"
                    "A shift toward physical retail that changes capital intensity\n"
                    "Higher fixed costs and reinvestment needs\n"
                    "The goal was not to argue sentiment, but to see what the numbers say when tied tightly to how the business actually operates."
                ),
            },
            {
                "heading": "Technical Details",
                "body": (
                    "I built a 10-year intrinsic DCF model followed by a stable terminal phase.\n"
                    "Key modeling choices were driven by business logic:\n"
                    "Revenue growth was aligned with industry growth and realistic market share recovery, not double-digit assumptions\n"
                    "Operating margins were capped around 6–7%, consistent with management targets and retail economics rather than optimistic 10–12% scenarios\n"
                    "Reinvestment reflected declining capital efficiency as Wayfair expands physical stores\n"
                    "Cost of capital was set using a WACC of ~8.9%, based on the U.S. 10-year risk-free rate and Wayfair’s risk profile\n"
                    "For relative valuation, I used EV/Sales as the primary multiple due to negative operating margins and high leverage. The peer set included logistics-heavy "
                    "e-commerce platforms and an omnichannel benchmark to reflect Wayfair’s evolving model."
                ),
            },
            {
                "heading": "Technical Challenges and Solutions",
                "body": (
                    "Narrative vs fundamentals\n"
                    "The market was pricing aggressive growth and margin expansion. I explicitly modeled what those assumptions would require in terms of market share gains and "
                    "cost structure, and found them inconsistent with industry dynamics.\n\n"
                    "Negative equity and ROC distortion\n"
                    "Wayfair’s negative book equity inflated return metrics. I treated book equity conservatively to avoid misleading conclusions.\n\n"
                    "Terminal value sensitivity\n"
                    "Instead of relying on optimistic terminal assumptions, I focused on the explicit forecast period where operating changes and reinvestment pressures are most visible.\n\n"
                    "Comps selection\n"
                    "Wayfair does not fit neatly into a single peer group. I limited the peer set to companies with similar logistics intensity and margin profiles to avoid false comfort from unrelated comps."
                ),
            },
            {
                "heading": "Results and Impact",
                "body": (
                    "DCF intrinsic value: ~$32 per share\n"
                    "Comps implied value: ~$76 per share\n"
                    "Market price at the time: ~$99 per share\n"
                    "Even under optimistic assumptions, the stock only approached the market price when operating margins exceeded 10%, a level inconsistent with industry structure and internal targets.\n"
                    "Both valuation approaches independently pointed to the same conclusion: the stock offered no margin of safety."
                ),
            },
            {
                "heading": "Conclusion",
                "body": (
                    "This project ended with a Sell recommendation, even though I initially approached the analysis with a more positive bias. The modeling process forced me to confront where the story broke down.\n"
                    "The biggest takeaway was not the target price, but the discipline of letting fundamentals override narrative. When valuation is anchored to industry structure, reinvestment reality, and operating economics, the conclusion becomes hard to ignore."
                ),
            },
        ],
    },

    {
        "slug": "atharva-capital-portfolio-simulation",
        "title": "Portfolio Management Simulation – Atharva Capital",
        "one_liner": "Managed a $1M simulated portfolio for 45 days, rotating from defense into undervalued equities to generate alpha vs S&P 500.",
        "summary": (
            "A live portfolio simulation using Bloomberg PRTU where I started defensively, waited for volatility, then rotated into high-conviction positions "
            "based on valuation, margin of safety, and sizing discipline."
        ),
        "sections": [
            {"heading": "1-line summary", "body": "I managed a $1M simulated portfolio over 45 days, starting defensively and rotating into undervalued equities during market drawdowns to generate meaningful alpha versus the S&P 500."},
            {
                "heading": "Overview",
                "body": (
                    "This project was a live portfolio management simulation where I managed a $1,000,000 portfolio using Bloomberg’s PRTU function over a 45-day period, from March 1 to April 15, 2025.\n"
                    "Rather than treating this as a trading exercise, I approached it as if I were managing real client capital. Every allocation decision was driven by valuation, margin of safety, and conviction, not by the need to stay fully invested. "
                    "The portfolio evolved dynamically as market conditions changed, moving from a defensive posture into concentrated equity positions as opportunities emerged."
                ),
            },
            {
                "heading": "Project Motivation",
                "body": (
                    "When the simulation began, markets across both the U.S. and India felt stretched. Valuations were high, sentiment was optimistic, and I did not see many opportunities that met my criteria for downside protection.\n"
                    "Instead of forcing capital into expensive equities, I made a conscious decision to start defensively. My goal was not to maximize activity, but to preserve flexibility.\n"
                    "This project was about answering one question:\n"
                    "Can discipline and patience outperform forced participation over a short and volatile window?"
                ),
            },
            {
                "heading": "Decision-Making Process",
                "body": (
                    "Starting defensively\n"
                    "I began the portfolio fully allocated to a Gold ETF (IAUM), as a strategic placeholder while I waited for equity valuations to reset.\n\n"
                    "First rotation into equities: Polycab\n"
                    "On March 3, Polycab dropped sharply on news of a potential new entrant. I viewed this as a market overreaction and initiated a 10% position.\n\n"
                    "Conviction under pressure: NVDA and Phinia\n"
                    "On March 13, I initiated positions in NVDA and Phinia following sharp declines. I averaged down in NVDA as markets sold off further, allowing the position to grow to roughly 20%.\n\n"
                    "Expanding into India during volatility\n"
                    "Between March 25 and April 8, I added Mahindra & Mahindra, Goldiam, and HDFC Bank as global markets weakened.\n"
                ),
            },
            {
                "heading": "Risk Framework and Portfolio Construction",
                "body": (
                    "I ran the portfolio with a concentrated, high-conviction mindset. When valuation offered downside protection, I was comfortable allocating 10% or more to a single idea.\n"
                    "The portfolio evolved from 100% gold to roughly 80% equities and 20% gold by the end of the period. Exposure was split across U.S. and Indian markets."
                ),
            },
            {
                "heading": "Results and Impact",
                "body": (
                    "By rotating capital from gold into undervalued equities during periods of fear, the portfolio outperformed the S&P 500 over the simulation window.\n"
                    "More importantly, performance came from process, not prediction. I waited, acted selectively, and sized positions based on conviction rather than noise."
                ),
            },
            {
                "heading": "Reflections and Lessons",
                "body": (
                    "Patience beats activity. Starting defensively created flexibility to buy when others were forced sellers.\n"
                    "Conviction must be earned. NVDA reminded me how narrative can overpower discipline.\n"
                    "Geographic familiarity matters. My deeper understanding of Indian companies improved decisions under stress.\n"
                    "Short windows test discipline, not brilliance."
                ),
            },
            {
                "heading": "Conclusion",
                "body": (
                    "This simulation was less about outperforming a benchmark and more about understanding myself as a portfolio manager.\n"
                    "Markets fall faster than they rise, fear creates opportunity, and patience is not passive, it is preparatory."
                ),
            },
        ],
    },

    {
        "slug": "ametek-equity-research-ame",
        "title": "Group Equity Research – AMETEK Inc. (NYSE: AME)",
        "one_liner": "Full equity research analysis with a Sell recommendation, later validated by a post-recommendation price decline.",
        "summary": (
            "A group research project where I focused on valuation discipline, separating business quality from investment attractiveness using DCF and comps."
        ),
        "sections": [
            {"heading": "1-line summary", "body": "Conducted a full equity research analysis on AMETEK and recommended a Sell based on DCF and relative valuation, which was later validated by a post-recommendation price decline."},
            {
                "heading": "Overview",
                "body": (
                    "This project was a group-based equity research assignment focused on evaluating the investment attractiveness of AMETEK Inc., a diversified industrial company with a track record of margin expansion and acquisition-led growth.\n"
                    "My role was to assess whether AMETEK’s market valuation was justified by its fundamentals, and whether the price left any margin of safety."
                ),
            },
            {
                "heading": "Project Motivation",
                "body": (
                    "AMETEK is often viewed as a high-quality compounder within industrials. That reputation can lead investors to overlook valuation discipline.\n"
                    "I wanted to test whether expectations around growth, margins, and capital deployment had become too optimistic."
                ),
            },
            {
                "heading": "Technical Details",
                "body": (
                    "I built a DCF model based on AMETEK’s operating segments and ran a comparable company analysis using relevant industrial peers across EV/EBITDA, EV/EBIT, and P/E.\n"
                    "The goal was consistency across methods rather than precision in any single output."
                ),
            },
            {
                "heading": "Key Analytical Challenges",
                "body": (
                    "High-quality businesses distort valuation signals\n"
                    "Strong margins and steady growth can make a stock look safe even when valuation stretches.\n\n"
                    "Acquisition-driven growth assumptions\n"
                    "I avoided assuming that historical acquisition success would continue indefinitely.\n\n"
                    "DCF sensitivity\n"
                    "To reduce terminal value dependence, I focused on the explicit forecast period."
                ),
            },
            {
                "heading": "Results and Impact",
                "body": (
                    "Both the DCF and comparable valuation indicated AMETEK was trading above intrinsic value at the time of analysis.\n"
                    "We issued a Sell recommendation.\n"
                    "Over the subsequent three months, the stock declined by approximately 6%, aligning with the valuation-driven conclusion."
                ),
            },
            {
                "heading": "Conclusion",
                "body": (
                    "This project reinforced a lesson: quality alone does not justify any price.\n"
                    "Strong businesses can still be poor investments when expectations are stretched."
                ),
            },
        ],
    },

    {
        "slug": "gentex-lbo-take-private",
        "title": "Private Equity LBO Simulation – Gentex Take-Private Analysis",
        "one_liner": "Built an LBO, DCF, and comps framework to test leverage capacity and risk-adjusted returns for a take-private.",
        "summary": (
            "A private equity style underwriting exercise focused on cash flow durability, leverage discipline, and exit realism rather than market narratives."
        ),
        "sections": [
            {
                "heading": "Overview",
                "body": (
                    "This project was a simulated private equity investment analysis focused on Gentex Corporation, a global leader in auto-dimming mirrors and advanced automotive electronics.\n"
                    "The objective was to determine whether Gentex could be an attractive take-private candidate for a long-term, value-oriented sponsor."
                ),
            },
            {
                "heading": "Project Motivation",
                "body": (
                    "Gentex stood out because it sits at the intersection of stability and optionality.\n"
                    "The core automotive business generates recurring cash flows, while newer initiatives add long-term upside without requiring aggressive assumptions."
                ),
            },
            {
                "heading": "Investment Thesis",
                "body": (
                    "Defensive core cash flows\n"
                    "Embedded growth optionality\n"
                    "Balance sheet capacity"
                ),
            },
            {
                "heading": "Technical Details",
                "body": (
                    "I built three parallel valuation frameworks: a DCF, comps, and a full LBO model.\n"
                    "Base case assumptions:\n"
                    "Entry multiple ~13x LTM EBITDA\n"
                    "About 58% debt financing at entry\n"
                    "5-year hold period\n"
                    "Exit at the same multiple"
                ),
            },
            {
                "heading": "Key Decision Points and Trade-Offs",
                "body": (
                    "Leverage discipline\n"
                    "Exit assumptions\n"
                    "Growth realism"
                ),
            },
            {
                "heading": "Results and Impact",
                "body": (
                    "Under base-case assumptions, the transaction generated approximately 2.4x MOIC and an IRR of ~18%, with stable returns across reasonable scenarios.\n"
                    "The analysis clarified where returns were resilient and where they deteriorated."
                ),
            },
            {
                "heading": "Conclusion",
                "body": (
                    "In a buyout, valuation alone is not enough. Cash flow durability, capital structure discipline, and exit realism matter far more than optimistic growth narratives."
                ),
            },
        ],
    },

    {
        "slug": "markowitz-meets-ben-graham",
        "title": "When Markowitz Meets Ben Graham – Portfolio Optimization vs Judgment",
        "one_liner": "Compared a real manager’s portfolio to a Markowitz optimizer to see where math ends and judgment begins.",
        "summary": (
            "A practical study on why skilled investors hold portfolios that look mathematically suboptimal, and where blending discipline and conviction works best."
        ),
        "sections": [
            {"heading": "1-line summary", "body": "I compared a real equity manager’s portfolio to a Markowitz-optimized allocation to understand where quantitative efficiency ends and fundamental judgment begins."},
            {
                "heading": "Overview",
                "body": (
                    "I analyzed an actively managed equity portfolio and compared its weights to those generated by Markowitz mean–variance optimization.\n"
                    "The goal was to understand why investors override models, and where optimization fails to capture business quality, valuation, and downside protection."
                ),
            },
            {
                "heading": "Project Motivation",
                "body": (
                    "In theory, Markowitz gives the best portfolio for a given risk. In practice, investors override it.\n"
                    "I wanted to understand why, and where models provide discipline versus where they miss what matters in long-term investing."
                ),
            },
            {
                "heading": "Technical Details",
                "body": (
                    "I reconstructed the manager’s portfolio, computed returns, vol, and correlations, built an efficient frontier, and generated a max-Sharpe portfolio.\n"
                    "Then I compared optimized weights stock by stock against the manager’s weights and evaluated the business reasons behind differences."
                ),
            },
            {
                "heading": "Results and Insights",
                "body": (
                    "The optimized portfolio delivered higher theoretical efficiency, but removed positions a long-term investor might intentionally hold.\n"
                    "Markowitz optimizes based on the past, investors allocate based on how businesses evolve.\n"
                    "The strongest outcome came from blending both approaches."
                ),
            },
            {
                "heading": "Conclusion",
                "body": (
                    "Quant models are powerful tools for risk control, but they are not decision-makers.\n"
                    "Real investing sits between mathematical discipline and fundamental conviction."
                ),
            },
        ],
    },

    {
        "slug": "earnings-call-signal-extraction-fdap",
        "title": "Earnings Call Signal Extraction (FDAP)",
        "one_liner": "Built a Python workflow to scan earnings transcripts and surface tone and narrative shifts that matter.",
        "summary": (
            "A repeatable system to extract investment-relevant changes from unstructured earnings call text, complementing traditional modeling."
        ),
        "sections": [
            {"heading": "1-line summary", "body": "I built a Python-based system to scan earnings call transcripts and surface changes in management tone and messaging that matter for investment decisions."},
            {
                "heading": "Overview",
                "body": (
                    "This project focused on extracting useful signals from earnings calls where critical information is buried in unstructured text.\n"
                    "I built a Python workflow that processes transcripts and highlights changes in tone, emphasis, and narrative over time."
                ),
            },
            {
                "heading": "Project Motivation",
                "body": (
                    "Earnings calls are subjective. Two analysts can read the same transcript and reach different conclusions.\n"
                    "I wanted to make earnings analysis more consistent by identifying what changed quarter to quarter, rather than relying on intuition."
                ),
            },
            {
                "heading": "Technical Details",
                "body": (
                    "I cleaned and structured transcripts, split content into prepared remarks and Q&A, grouped commentary by themes, and tracked changes over time.\n"
                    "I checked whether tone and emphasis shifts aligned with subsequent performance and market reactions."
                ),
            },
            {
                "heading": "Results and Impact",
                "body": (
                    "The framework made it faster to identify early warning signs and changes in management confidence, and helped guide where deeper fundamental work was needed."
                ),
            },
            {
                "heading": "Conclusion",
                "body": (
                    "Qualitative information can be handled with the same discipline as financial data. Data tools can sharpen judgment rather than replace it."
                ),
            },
        ],
    },
]
