You are transcribing one scanned page of a book so that its text can be searched. The page is from {source_file}, page {page}.

Copy all of the text on the page exactly as printed, in reading order. Output only the page's text: no introduction, no comments, no Markdown code fences.

Rules:
- Keep each paragraph together and put a blank line between paragraphs. Put headings on their own line.
- Join words that are hyphenated across a line break ("compli-" + "cated" becomes "complicated").
- Leave out the running header, the running footer and the page number.
- Keep every word in its original language and script. Copy Sanskrit or Hindi in Devanagari as Devanagari, and keep transliteration marks (ā, ś, ṣ, ṛ) as printed. Do not translate.
- Write a table as a Markdown table, one printed row per row. If the page or a table is printed sideways, read it in its natural orientation.
- For a horoscope chart or another diagram, write one line, [Chart: …], with the labels and numbers it shows, then continue with the text.
- If a word cannot be read, write [illegible] in its place. Never guess or invent text.
- If the page has no text at all, output exactly: [blank page]
