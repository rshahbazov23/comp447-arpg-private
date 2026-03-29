#!/usr/bin/env python3
"""Generate the Feasibility Report PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register fonts that support Turkish characters
import os, subprocess

# Try to register system fonts with Turkish glyph support
_FONT_REGISTERED = False
for font_path in [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]:
    if os.path.exists(font_path):
        try:
            if font_path.endswith(".ttc"):
                from reportlab.pdfbase.ttfonts import TTFont
                pdfmetrics.registerFont(TTFont("ArialUni", font_path, subfontIndex=0))
            else:
                pdfmetrics.registerFont(TTFont("ArialUni", font_path))
            _FONT_REGISTERED = True
            break
        except Exception:
            continue

# Fallback: use DejaVu if available
if not _FONT_REGISTERED:
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("ArialUni", font_path))
                _FONT_REGISTERED = True
                break
            except Exception:
                continue

OUTPUT_PATH = "/Users/riko/Documents/Spring 2026/COMP447/Project/ARPG-main/Feasibility_Report.pdf"

def build_styles():
    ss = getSampleStyleSheet()

    styles = {}
    styles["title"] = ParagraphStyle(
        "Title2", parent=ss["Title"], fontSize=18, leading=22,
        spaceAfter=4, alignment=TA_CENTER,
    )
    styles["subtitle"] = ParagraphStyle(
        "Subtitle", parent=ss["Normal"], fontSize=11, leading=14,
        alignment=TA_CENTER, textColor=HexColor("#444444"), spaceAfter=2,
    )
    styles["date"] = ParagraphStyle(
        "Date", parent=ss["Normal"], fontSize=10, leading=12,
        alignment=TA_CENTER, textColor=HexColor("#666666"), spaceAfter=20,
    )
    styles["h1"] = ParagraphStyle(
        "H1", parent=ss["Heading1"], fontSize=15, leading=18,
        spaceBefore=18, spaceAfter=8, textColor=HexColor("#1a1a2e"),
    )
    styles["h2"] = ParagraphStyle(
        "H2", parent=ss["Heading2"], fontSize=12, leading=15,
        spaceBefore=12, spaceAfter=6, textColor=HexColor("#16213e"),
    )
    styles["body"] = ParagraphStyle(
        "Body2", parent=ss["Normal"], fontSize=9.5, leading=13,
        spaceAfter=6, alignment=TA_JUSTIFY,
    )
    styles["body_bold"] = ParagraphStyle(
        "BodyBold", parent=styles["body"], fontName="Helvetica-Bold",
    )
    styles["bullet"] = ParagraphStyle(
        "Bullet2", parent=styles["body"], leftIndent=18,
        bulletIndent=6, spaceBefore=1, spaceAfter=1,
    )
    styles["code"] = ParagraphStyle(
        "Code2", parent=ss["Code"], fontSize=8, leading=10,
        leftIndent=12, spaceAfter=6, backColor=HexColor("#f5f5f5"),
        borderColor=HexColor("#cccccc"), borderWidth=0.5,
        borderPadding=4, fontName="Courier",
    )
    styles["verdict_green"] = ParagraphStyle(
        "VGreen", parent=styles["body"], fontSize=10, leading=13,
        textColor=HexColor("#2e7d32"), fontName="Helvetica-Bold",
    )
    styles["verdict_yellow"] = ParagraphStyle(
        "VYellow", parent=styles["body"], fontSize=10, leading=13,
        textColor=HexColor("#f57f17"), fontName="Helvetica-Bold",
    )
    return styles


_cell_style = ParagraphStyle("CellStyle", fontSize=7.5, leading=10, fontName="Helvetica")
_cell_style_bold = ParagraphStyle("CellStyleBold", fontSize=7.5, leading=10, fontName="Helvetica-Bold")
_header_style = ParagraphStyle("HeaderStyle", fontSize=8, leading=10, fontName="Helvetica-Bold", textColor=white)


def _wrap_cells(data, is_header=False):
    """Wrap all string cells in Paragraphs so text wraps inside columns."""
    wrapped = []
    for row in data:
        new_row = []
        for cell in row:
            if isinstance(cell, str):
                style = _header_style if is_header else _cell_style
                new_row.append(Paragraph(cell.replace("\n", "<br/>"), style))
            else:
                new_row.append(cell)
        wrapped.append(new_row)
    return wrapped


def make_table(header, rows, col_widths=None):
    wrapped_header = _wrap_cells([header], is_header=True)
    wrapped_rows = _wrap_cells(rows)
    data = wrapped_header + wrapped_rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def build_story(S):
    story = []
    add = story.append

    # ── Title page ──
    add(Spacer(1, 1.5 * inch))
    add(Paragraph("ARPG Confidence-Guided Token Rejection", S["title"]))
    add(Paragraph("Code Feasibility Report", S["title"]))
    add(Spacer(1, 12))
    add(HRFlowable(width="60%", thickness=1, color=HexColor("#1a1a2e"),
                    spaceAfter=12, spaceBefore=4))
    if _FONT_REGISTERED:
        author_str = ('Riad Shahbazov &nbsp;&middot;&nbsp; '
                      '<font face="ArialUni">\u00d6mer Mara\u015f</font> '
                      '&nbsp;&middot;&nbsp; Mohamad Alomar')
    else:
        author_str = ('Riad Shahbazov &nbsp;&middot;&nbsp; '
                      'Omer Maras &nbsp;&middot;&nbsp; Mohamad Alomar')
    add(Paragraph(author_str, S["subtitle"]))
    add(Paragraph("March 29, 2026", S["date"]))
    add(Spacer(1, 30))

    # Executive summary
    add(Paragraph("Executive Summary", S["h1"]))
    add(Paragraph(
        "We audited the ARPG codebase to determine whether our proposed inference-time "
        "modification \u2014 confidence-guided token rejection and a refinement-pass fallback "
        "\u2014 is implementable without retraining or architectural changes. "
        "<b>The answer is yes.</b> The refinement pass is low-risk and can be implemented in "
        "~30 lines of code. Full in-loop rejection is also feasible but requires more careful "
        "engineering. The codebase is clean, modular, pure PyTorch, with no custom CUDA kernels "
        "or opaque abstractions blocking us.", S["body"]))
    add(PageBreak())

    # ── Section 1 ──
    add(Paragraph("1. Codebase Structure (What Matters)", S["h1"]))
    add(make_table(
        ["File / Location", "What It Does"],
        [
            ["models/arpg.py\nTransformer.generate() (L453\u2013542)",
             "The entire inference loop. This is where all our modifications go."],
            ["models/arpg.py\nTransformer.forward_shared() (L418\u2013428)",
             "Runs Pass-1 (content KV) + Pass-2 (cross-attention prediction) and returns logits."],
            ["models/arpg.py\nDecoder_Decoder.forward() (L323\u2013345)",
             "Executes both decoder passes. Pass-1 self-attention builds KV cache; "
             "shared K,V projection feeds Pass-2 cross-attention."],
            ["models/arpg.py\nAttention.update_kv_cache() (L141\u2013152)",
             "Simple torch.cat append for Pass-1 self-attention caches."],
            ["models/arpg.py\nDecoder_Decoder.update_kv_cache() (L308\u2013321)",
             "Simple torch.cat append for shared K,V (what Pass-2 attends to). "
             "This is the cache our proposal targets."],
            ["models/arpg.py\nprecompute_freqs_cis_2d() (L556\u2013569)",
             "Precomputes 2D RoPE frequencies. Indexed by spatial position \u2014 not sequential."],
            ["sample_c2i_ddp.py\nmain()",
             "Calls generate(), decodes tokens to pixels via VQ model, saves PNGs."],
        ],
        col_widths=[2.2 * inch, 4.3 * inch],
    ))
    add(Paragraph(
        "Everything else (VQ model, datasets, training code) is irrelevant to our modification.",
        S["body"]))

    # ── Section 2 ──
    add(Paragraph("2. Where Logits Are Produced \u2014 The Interception Point", S["h1"]))
    add(Paragraph(
        "Inside <font face='Courier' size=9>generate()</font>, each decoding step produces logits through this sequence:",
        S["body"]))
    add(Paragraph(
        "# Line 521: Forward pass through both decoders<br/>"
        "logits = self.forward_shared(input_ids, freqs_cis, num_pred)<br/><br/>"
        "# Lines 522\u2013523: Classifier-free guidance<br/>"
        "cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]<br/>"
        "logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale<br/><br/>"
        "# Line 528: Temperature scaling<br/>"
        "logits = logits[:, -num_pred:] / max(temperature, 1e-5)<br/><br/>"
        "# Line 533: Softmax \u2192 full probability distribution &lt;\u2014 INTERCEPTION POINT<br/>"
        "probs = F.softmax(logits, dim=-1)<br/><br/>"
        "# Line 534: Sample tokens<br/>"
        "sampled = torch.multinomial(probs.flatten(0, 1), num_samples=1)",
        S["code"]))
    add(Paragraph(
        "<b>Right after line 533</b>, we have the full probability distribution per token. "
        "Computing all three confidence metrics is trivial:", S["body"]))
    add(Paragraph(
        "# probs shape: (num_samples, num_pred, vocab_size)<br/><br/>"
        "# 1. Max softmax probability<br/>"
        "max_prob, _ = probs.max(dim=-1)<br/><br/>"
        "# 2. Entropy<br/>"
        "entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)<br/><br/>"
        "# 3. Top-1 / top-2 margin<br/>"
        "top2 = probs.topk(2, dim=-1).values<br/>"
        "margin = top2[:, :, 0] - top2[:, :, 1]",
        S["code"]))
    add(Paragraph("<b>No obstacles here. This is plug-and-play.</b>", S["body_bold"]))

    # ── Section 3 ──
    add(Paragraph("3. KV Cache Architecture \u2014 Can We Intercept It?", S["h1"]))
    add(Paragraph("There are two active cache levels during inference:", S["body"]))
    add(make_table(
        ["Cache", "Location", "What It Stores", "Update Method"],
        [
            ["Pass-1 self-attn", "Attention.update_kv_cache()\nper layer",
             "K,V for causal self-attention\nover content tokens", "torch.cat append"],
            ["Shared K,V\n(Pass-1 \u2192 Pass-2)", "Decoder_Decoder\n.update_kv_cache()",
             "Projected representations that\nPass-2 queries attend to", "torch.cat append"],
        ],
        col_widths=[1.0 * inch, 1.6 * inch, 2.2 * inch, 1.2 * inch],
    ))
    add(Paragraph(
        "Both use simple <font face='Courier' size=9>torch.cat</font> concatenation. "
        "No paged attention, no compiled CUDA kernels, no complex memory management.", S["body"]))
    add(Paragraph(
        "<b>Critical finding:</b> Pass-2 cross-attention layers do NOT have their own caches at "
        "inference (the kv_cache flag is never set to True in setup_kv_cache()). They receive the "
        "full accumulated shared K,V as direct arguments each step.", S["body"]))
    add(Paragraph("<b>Verdict: Caches are fully interceptable.</b>", S["body_bold"]))

    # ── Section 4 ──
    add(Paragraph("4. Key Architectural Insight \u2014 Timing Works In Our Favor", S["h1"]))
    add(Paragraph(
        "<b>Predicted tokens do NOT immediately enter the cache.</b> Here is why:", S["body"]))
    add(Paragraph(
        "\u2022 At step <i>t</i>, the model takes sequences[-1] (tokens from step <i>t\u22121</i>) as input.",
        S["bullet"]))
    add(Paragraph(
        "\u2022 Pass-1 processes these and adds them to the cache.", S["bullet"]))
    add(Paragraph(
        "\u2022 Pass-2 predicts new tokens for step <i>t</i>.", S["bullet"]))
    add(Paragraph(
        "\u2022 These new predictions are appended to sequences but do not enter the cache until step <i>t+1</i>.",
        S["bullet"]))
    add(Spacer(1, 4))
    add(Paragraph(
        "This means <b>we can evaluate confidence on newly predicted tokens and decide whether to "
        "accept or reject them BEFORE they contaminate the cache.</b> This is the single most "
        "important structural finding \u2014 it makes both approaches viable.", S["body"]))

    # ── Section 5 ──
    add(Paragraph("5. Feasibility of Full In-Loop Defer-and-Redecode", S["h1"]))
    add(Paragraph("Verdict: FEASIBLE but moderately invasive (~60\u201380 lines modified)",
                  S["verdict_yellow"]))
    add(Spacer(1, 4))

    add(Paragraph("What works naturally:", S["h2"]))
    add(Paragraph(
        "\u2022 Confidence evaluation happens before cache contamination (see Section 4).",
        S["bullet"]))
    add(Paragraph(
        "\u2022 RoPE positions are <b>explicitly indexed</b> \u2014 freqs_cis_[:, next_range, ...] uses "
        "arbitrary position indices, not sequential offsets. Deferred tokens naturally keep their "
        "original spatial positions.", S["bullet"]))
    add(Paragraph(
        "\u2022 Cache updates are simple appends \u2014 feeding fewer accepted tokens just means a smaller append.",
        S["bullet"]))
    add(Paragraph(
        "\u2022 CFG combines logits before rejection \u2014 both batch halves get the same accept/reject decisions.",
        S["bullet"]))

    add(Paragraph("What requires work:", S["h2"]))
    add(make_table(
        ["Challenge", "Difficulty", "Description"],
        [
            ["Dynamic schedule", "Medium",
             "Must replace the simple last_pos counter with a dynamic pool of "
             "remaining + deferred positions. Rewrite ~15 lines of schedule logic."],
            ["Selective cache admission", "Medium",
             "Feed only accepted tokens as next step's input; last_range becomes the "
             "accepted subset, not the full next_range."],
            ["Per-sample variable rejection", "Medium",
             "Different samples may reject different tokens. Either fix rejection count "
             "per step (simpler) or use padding/masking (more complex)."],
            ["torch.compile incompatibility", "Low",
             "Dynamic control flow breaks fullgraph=True compilation. Must disable. Trivial fix."],
            ["Bookkeeping", "Medium",
             "Track deferred positions, merge them back into future steps, handle forced "
             "acceptance at the final step."],
        ],
        col_widths=[1.4 * inch, 0.7 * inch, 3.9 * inch],
    ))
    add(Spacer(1, 4))

    add(Paragraph("Required state to maintain:", S["h2"]))
    add(Paragraph(
        "\u2022 <font face='Courier' size=8>remaining_positions</font>: tensor of undecoded spatial indices "
        "(starts as full orders, shrinks)", S["bullet"]))
    add(Paragraph(
        "\u2022 <font face='Courier' size=8>deferred_positions</font>: positions rejected this step, "
        "prepended back to remaining_positions", S["bullet"]))
    add(Paragraph(
        "\u2022 <font face='Courier' size=8>accepted_mask</font>: per-step boolean mask over predicted tokens",
        S["bullet"]))
    add(Paragraph(
        "\u2022 Current schedule position recalculated from len(remaining_positions) instead of last_pos",
        S["bullet"]))

    # ── Section 6 ──
    add(Paragraph("6. Feasibility of Refinement-Pass Fallback", S["h1"]))
    add(Paragraph("Verdict: FEASIBLE and straightforward (~30 lines of new code)",
                  S["verdict_green"]))
    add(Spacer(1, 4))

    add(Paragraph("Why it is simpler:", S["h2"]))
    add(Paragraph(
        "\u2022 The main generation loop runs <b>completely unmodified</b>.", S["bullet"]))
    add(Paragraph(
        "\u2022 At the end of generation, the KV cache is full (all 256 tokens represented).",
        S["bullet"]))
    add(Paragraph(
        "\u2022 We just add a post-loop step: identify worst tokens, re-decode them against the full cache.",
        S["bullet"]))
    add(Paragraph(
        "\u2022 This is architecturally identical to ARPG's own zero-shot inpainting (paper Fig. 5b) "
        "\u2014 the model already supports querying specific positions against a pre-filled cache.",
        S["bullet"]))

    add(Paragraph("Implementation sketch:", S["h2"]))
    add(Paragraph(
        "<b>Step 1</b> \u2014 During normal generation, store per-step confidence (add 3 lines inside the loop):",
        S["body"]))
    add(Paragraph(
        "# After: probs = F.softmax(logits, dim=-1)<br/>"
        "step_conf = probs.max(dim=-1).values<br/>"
        "all_confidences.append(step_conf)<br/>"
        "all_positions.append(next_range)",
        S["code"]))
    add(Paragraph(
        "<b>Step 2</b> \u2014 After the loop, before setup_kv_cache(enable=False), add refinement:",
        S["body"]))
    add(Paragraph(
        "# Identify K% worst tokens<br/>"
        "all_conf = torch.cat(all_confidences, dim=-1)<br/>"
        "num_refine = int(seq_len * refine_ratio)<br/>"
        "_, worst_idx = all_conf.topk(num_refine, largest=False, dim=-1)<br/>"
        "worst_positions = all_pos[worst_idx[0]]<br/><br/>"
        "# Create [MASK] queries for those positions<br/>"
        "queries = self.embeddings(<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;torch.full((num_samples, num_refine),<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;self.mask_token_id, device=device))<br/><br/>"
        "# Run Pass-2 cross-attention against FULL cached K,V<br/>"
        "refine_freqs = freqs_cis_[:, worst_positions, ...]<br/>"
        "q = queries<br/>"
        "for layer in self.layers.cross_dec:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;q = layer(x=q, k=self.layers.k_cache,<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v=self.layers.v_cache, freqs_cis=refine_freqs)<br/><br/>"
        "refine_logits = self.head(self.norm(q)).float()<br/>"
        "# Apply temperature, top-k/top-p, sample, replace tokens in output",
        S["code"]))

    add(Paragraph("Why RoPE works correctly:", S["h2"]))
    add(Paragraph(
        "\u2022 Cached keys already have RoPE applied during generation (Decoder_Decoder.forward(), L337).",
        S["bullet"]))
    add(Paragraph(
        "\u2022 CrossAttention.forward() applies RoPE only to the query (line 229).",
        S["bullet"]))
    add(Paragraph(
        "\u2022 Attention score = q_rot(target_pos) \u00b7 k_rot(content_pos) \u2192 correct relative position encoding.",
        S["bullet"]))
    add(Paragraph(
        "\u2022 We just index freqs_cis_ with the correct spatial positions for our refinement queries.",
        S["bullet"]))

    # ── Section 7 ──
    add(Paragraph("7. Engineering Risks (Ranked)", S["h1"]))
    add(make_table(
        ["Risk", "Severity", "Details", "Mitigation"],
        [
            ["torch.compile with\nfullgraph=True", "HIGH",
             "Default is ON in sample_c2i_ddp.py L164. "
             "Dynamic control flow will break compilation.",
             "Set default=False or pass --no-compile. "
             "For refinement-only, can keep compile for main loop."],
            ["Per-sample variable\nrejection counts", "MEDIUM",
             "In-loop rejection only. Different batch items "
             "may reject different numbers of tokens.",
             "Fix rejection count per step (reject bottom K% "
             "always = same count across batch)."],
            ["Schedule rewrite for\ndynamic pool", "MEDIUM",
             "In-loop rejection only. Must replace ~15 lines "
             "of schedule logic.",
             "Start with refinement pass (no schedule changes needed)."],
            ["Stale confidence\nafter CFG", "LOW",
             "We compute confidence on CFG-combined logits, "
             "which don't correspond to either distribution exactly.",
             "Acceptable \u2014 CFG-combined logits reflect the "
             "actual sampling distribution."],
            ["No custom CUDA /\nno paged attention", "NONE",
             "Pure PyTorch, simple torch.cat caches, "
             "standard F.scaled_dot_product_attention.",
             "This is the best possible news. Nothing "
             "blocking us at the systems level."],
        ],
        col_widths=[1.1 * inch, 0.65 * inch, 2.25 * inch, 2.25 * inch],
    ))

    # ── Section 8 ──
    add(Paragraph("8. Recommended Implementation Plan", S["h1"]))

    add(Paragraph("Phase 1: Refinement Pass (Week 1)", S["h2"]))
    add(Paragraph(
        "1. Modify generate() to store per-step probs and position indices. "
        "<b>Zero risk to existing behavior.</b>", S["bullet"]))
    add(Paragraph(
        "2. Add post-loop refinement method. <b>Modular, independently testable.</b>",
        S["bullet"]))
    add(Paragraph(
        "3. Validate: refine_ratio=0.0 must reproduce vanilla ARPG exactly.", S["bullet"]))
    add(Paragraph(
        "4. Validate: refine_ratio=1.0 (re-decode everything) as a sanity check.", S["bullet"]))
    add(Paragraph(
        "5. Disable torch.compile for initial development.", S["bullet"]))

    add(Paragraph("Phase 2: Pilot Sweep (Week 2)", S["h2"]))
    add(Paragraph(
        "1. Sweep refine_ratio in {0.05, 0.10, 0.20} x all 3 confidence metrics.", S["bullet"]))
    add(Paragraph("2. Evaluate with FID-10K (fast).", S["bullet"]))
    add(Paragraph("3. Generate spatial heatmaps of rejected token locations.", S["bullet"]))
    add(Paragraph(
        "4. If FID improves: confidence-guided hypothesis validated \u2192 proceed to Phase 3.",
        S["bullet"]))
    add(Paragraph(
        "5. If FID does not improve: the full in-loop version is unlikely to help. "
        "Write up the negative result with analysis.", S["bullet"]))

    add(Paragraph("Phase 3: In-Loop Rejection (Weeks 3\u20134, only if Phase 2 shows signal)",
                  S["h2"]))
    add(Paragraph(
        "1. Rewrite schedule loop with dynamic remaining-position pool.", S["bullet"]))
    add(Paragraph("2. Implement selective cache admission.", S["bullet"]))
    add(Paragraph(
        "3. Fix rejection count per step to avoid per-sample variability.", S["bullet"]))
    add(Paragraph(
        "4. Compare against refinement pass and vanilla under matched step budgets.",
        S["bullet"]))

    # ── Section 9 ──
    add(Paragraph("9. Final Verdict", S["h1"]))
    add(make_table(
        ["Approach", "Status", "Effort", "Risk"],
        [
            ["Refinement pass", "GREEN", "~30 lines, ~3\u20134 days", "Low"],
            ["In-loop defer-and-redecode", "YELLOW", "~70 lines, ~1\u20132 weeks", "Medium"],
        ],
        col_widths=[2.0 * inch, 0.8 * inch, 1.8 * inch, 1.0 * inch],
    ))
    add(Spacer(1, 10))
    add(Paragraph(
        "The codebase is unusually well-structured for an academic repo. No CUDA kernels, no compiled "
        "graphs we cannot disable, no distributed entanglement in the generation loop, and explicit "
        "position indexing throughout. The timing of cache updates works in our favor \u2014 predicted "
        "tokens do not enter the cache until the next step, giving us a clean interception window.",
        S["body"]))
    add(Spacer(1, 6))
    add(Paragraph(
        "<b>Start with the refinement pass.</b> It is our safety net and our controlled ablation. "
        "If it shows signal, invest in the full in-loop version. If it does not, we still have a "
        "complete, well-motivated negative result to report. The architecture of this codebase is "
        "unusually clean and modular \u2014 no custom CUDA, no opaque abstractions, pure PyTorch. "
        "That is the best news in this audit.", S["body"]))

    return story


def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="ARPG Confidence-Guided Token Rejection \u2014 Code Feasibility Report",
        author="Riad Shahbazov, \u00d6mer Mara\u015f, Mohamad Alomar",
    )
    S = build_styles()
    story = build_story(S)
    doc.build(story)
    print(f"PDF saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
