# AI智能体设计方案模板执行约定

## Reference

- Retained reference: `E:\赛题四_智能体\data\raw\AI智能体设计方案.docx`
- SHA-256: `8E8148DB47A1FD7050C11027105F85299C72596FFAEFD4C9A700824AFC9877D0`
- Reference pages: 1
- Sections: 1
- Read-only render: `E:\赛题四_智能体\qa\template-reference-render\page-1.png`
- Audit evidence: `E:\赛题四_智能体\qa\template-style-evidence.json`

## Page System

- A4 portrait, 8.27 x 11.69 inches.
- Margins: left 1.25 inches, right 1.25 inches, top 1.00 inch, bottom 1.00 inch.
- One section beginning with a new page; no columns, headers, footers, page-number fields, odd/even variants, drawings, or content controls.
- Content may paginate naturally, but page size, orientation, margins, and section count must remain unchanged.

## Typography and Rhythm

- All source paragraphs use the `Normal` style with direct formatting and black text.
- Title at `word/document.xml` body paragraph 1: centered Arial 26 pt bold; 24 pt before and after; 1.2 line spacing. Preserve the title text and visual treatment.
- Metadata at body paragraphs 2-4: Arial bold, left aligned, 6 pt before and after, 1.2 line spacing. Replace only the underscore values.
- Major section headings at body paragraphs 6, 8, 10, 12, and 14: Arial 16 pt bold, left aligned; 16 pt before and 6 pt after; 1.2 line spacing. Preserve wording, numbering, and order.
- Evaluation subheads at body paragraphs 15-16: Arial body size, left aligned, 6 pt before and after; 1.2 line spacing. Preserve numbering and wording except the source terminal semicolon may be normalized.
- Added body paragraphs use black Arial 10.5 pt, 1.3 line spacing, 0 pt before and 6 pt after. No decorative rules, callout cards, text boxes, or floating shapes.

## Tables

- Tables may be added only for architecture/module comparisons, validation metrics, and elapsed-time summaries.
- Fit within the 5.77-inch text width. Use content-weighted columns, vertically centered cells, light gray `D9D9D9` borders, dark navy `1D2330` header fill with white bold text, and alternating white / pale blue body rows.
- Use Arial 9 pt inside tables, allow automatic row expansion, and repeat headers if a table spans pages.

## Content Flow and Slot Map

- Paragraph 1: preserve the official title `AI智能体设计方案`.
- Paragraph 2: fill the team name only when supplied; otherwise preserve the underscore placeholder.
- Paragraph 3: replace the work-name underscore with `城市路桥隧边坡结构病害智能巡检与分级评定智能体`.
- Paragraph 4: replace the date underscore with the applicable completion date.
- Paragraph 6 blank/body slot after section one: add the problem, dataset scope, technical constraints, and design conclusion.
- Paragraph 8 blank/body slot after section two: add the end-to-end architecture narrative and architecture table.
- Paragraph 10 blank/body slot after section three: add data, routing, visual inference, structured-output, checkpoint, validation, and packaging module descriptions.
- Paragraph 12 blank/body slot after section four: add numbered innovation items grounded in implemented behavior.
- Paragraphs 15-16: add calibration/validation metrics and actual execution-time tables after their respective subheads.
- Existing blank spacer paragraphs may be reused or removed only to avoid empty-page gaps. Do not change source heading order.

## Package Preservation

- Preserve all package parts except `word/document.xml`, `docProps/core.xml`, and `docProps/app.xml`, which may change as required by content and metadata updates.
- Preserve `[Content_Types].xml`, `_rels/.rels`, `word/_rels/document.xml.rels`, `word/footnotes.xml`, `word/endnotes.xml`, `word/theme/theme1.xml`, `word/settings.xml`, `word/styles.xml`, `word/webSettings.xml`, `word/fontTable.xml`, and `docProps/custom.xml` unless python-docx performs a necessary deterministic relationship update for inserted tables.
- The retained reference remains byte-for-byte unchanged at its recorded path.

## Fidelity Gates

- Recompute the reference SHA-256 before authoring and before delivery.
- Render the final document to page PNGs, inspect every page at 100 percent, and compare against the source for page geometry, title, metadata, heading hierarchy, and recurring spacing.
- Rerun section, style, heading, image, field, and content-control audits. No unexpected sections, headers, footers, drawings, fields, controls, clipped text, broken tables, or missing Chinese glyphs are allowed.
