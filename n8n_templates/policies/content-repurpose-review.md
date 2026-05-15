# Policy: Content Repurpose Review

Purpose:
Turn one source article, transcript, or outline into review-ready social content.

Audience:
Traditional Chinese readers in Taiwan. Solo founders, consultants, coaches, creators, and small business operators.

Fixed Rules:
- Keep the main idea accurate.
- Do not invent sources, dates, prices, or legal claims.
- No external links in the final Facebook body unless explicitly requested.
- If a source is needed, include source name only.
- Write in natural Traditional Chinese.
- Avoid corporate buzzwords.
- Create versions for Facebook, Threads, Instagram caption, and email.
- Every output must be human-reviewed before publishing.

Output JSON:

```json
{
  "summary": "one sentence",
  "facebook": "string",
  "threads": ["post 1", "post 2", "post 3"],
  "instagram": "string",
  "email_subject": "string",
  "email_body": "string",
  "image_prompt": "optional visual idea",
  "risk_flags": ["missing source", "claim needs verification"]
}
```

