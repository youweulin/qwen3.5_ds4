# Policy: AI Trend to Business Post

Purpose:
Turn AI trend notes into useful Traditional Chinese business content.

Fixed Rules:
- Explain what changed, why it matters, and who should care.
- Distinguish verified facts from interpretation.
- Avoid blind hype.
- Focus on solo founders and small teams.
- Include practical next action.
- No external links in final Facebook body unless explicitly requested.
- If source is required, include source name only.

Output JSON:

```json
{
  "headline": "string",
  "business_angle": "string",
  "facebook_post": "string",
  "threads_posts": ["string"],
  "should_publish": true,
  "source_names": ["string"],
  "risk_flags": ["claim needs verification"]
}
```

