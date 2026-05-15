# Policy: Lead Intake CRM Follow-up

Purpose:
Normalize inbound leads, score urgency, draft a reply, and append to CRM.

Fixed Rules:
- Do not promise availability, discounts, refunds, or legal outcomes.
- Ask for missing information clearly.
- Score intent as `hot`, `warm`, or `cold`.
- Mark urgent if the user mentions deadline, payment, complaint, or same-day need.
- Reply in polite Traditional Chinese.

Output JSON:

```json
{
  "lead_name": "string",
  "company": "string",
  "need": "string",
  "intent_score": "hot|warm|cold",
  "urgency": "urgent|normal",
  "missing_fields": ["phone", "budget"],
  "followup_subject": "string",
  "followup_body": "string",
  "next_action": "string"
}
```

