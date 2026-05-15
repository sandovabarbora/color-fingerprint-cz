# Product

## Register

brand

## Users

A creative director, brand strategist, or hiring lead at a Czech advertising
agency. They've been forwarded a link to this single-page report by Barbora as
part of a portfolio review. They're skimming on a laptop between meetings,
deciding within the first 15 seconds whether to read past the headline.

They look at design work every day. The thing that earns their second minute
is not "I made a thing" but "this person sees what I see, and gives me a tool".

## Product Purpose

A portfolio artifact that bridges data engineering and creative direction. It
exists to demonstrate three things simultaneously: technical craft (a real,
reproducible pipeline), analytical rigor (numerical findings, not just
adjectives), and visual taste (the page itself reads like an editorial spread,
not a dashboard).

Success = the reader finishes the page, mentally categorizes the author as
"someone we should talk to", and forwards the link to one more colleague.

## Brand Personality

Editorial film-still. Serious without being academic. Beautiful without being
decorative. Confident enough to use negative space. Closer to a magazine
feature about color in advertising than to a SaaS dashboard about advertising.

Three-word personality: precise, considered, cinematic.

## Anti-references

- **Generic SaaS portfolio templates.** The Vercel/Notion-template look:
  card grids, hero metric boxes with big colored numbers, three-column feature
  rows, identical iconography. AI-slop adjacent. The current draft sits closer
  to this lane than the personality wants.
- **Mood-board collages.** A bunch of pretty images stitched together with no
  argument. The whole project's thesis is "stop using mood-boards; use
  numbers": the page itself must not be a mood-board.
- **Pure tech-demo aesthetic.** Dark mode, mono everywhere, log-style text.
  Wrong audience.

## Design Principles

1. **The page is the proof.** A report about color and creative discipline
   must itself demonstrate color and creative discipline. The form of the
   page is the argument; failing it is a credibility failure.

2. **Numbers, treated like film stills.** Findings get framed with care:
   pulled out, given air, allowed to be the focal point of a section.
   No densely-packed paragraphs of statistics.

3. **Negative space is a deliberate choice, not a default.** Editorial-style
   layouts vary. Some sections are tight and full of detail; some breathe.
   Same-padding-everywhere is the SaaS reflex to avoid.

4. **Show the color, don't just talk about it.** The page renders actual
   color from the corpus: strips, anchors, gaps, palette swatches.
   Decorative use of color (gradients, glow) is forbidden; analytical use
   of color (the strips, the bar fills, the scatter dots) is the whole job.

5. **Every word earns its place.** Copy is editorial-tight. No restated
   headings, no transitional throat-clearing, no em dashes (commas, colons,
   semicolons, periods, parentheses do the work). One verb per sentence
   wherever possible.

## Accessibility & Inclusion

- WCAG AA color contrast on body text.
- Functional without JavaScript (already the case, no JS anywhere).
- Reasonable reading at 320px width (responsive without separate mobile design).
- Color-blind safe for the chart axes (labels carry the meaning, not just
  the dot colors, anchor color is a redundant encoder, not the primary one).
- No motion that can't be disabled / doesn't trigger prefers-reduced-motion.
