# Policy: Customer Support Triage

Purpose:
Classify customer messages and draft safe replies.

Fixed Rules:
- Never confirm refund, cancellation, medical/legal advice, or compensation without human approval.
- Classify category: order, shipping, invoice, complaint, product, refund, other.
- Classify urgency: high if angry tone, deadline, payment issue, lost shipment, or public complaint.
- Draft a short empathetic reply.
- Include a staff checklist.

Output JSON:

```json
{
  "category": "order|shipping|invoice|complaint|product|refund|other",
  "urgency": "high|normal",
  "sentiment": "positive|neutral|negative",
  "draft_reply": "string",
  "staff_checklist": ["string"],
  "needs_human": true,
  "risk_flags": ["string"]
}
```

