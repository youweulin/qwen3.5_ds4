# Policy: Invoice Order Digest

Purpose:
Summarize order, invoice, and shipment rows into a daily admin checklist.

Fixed Rules:
- Do not modify orders.
- Do not mark anything shipped unless source data says shipped.
- Identify missing invoice number, missing tracking number, duplicated order ID, and abnormal amount.
- Produce a concise daily summary and action list.

Output JSON:

```json
{
  "daily_summary": "string",
  "total_orders": 0,
  "total_amount": 0,
  "needs_invoice": ["order_id"],
  "needs_shipment": ["order_id"],
  "exceptions": [
    {
      "order_id": "string",
      "issue": "string",
      "suggested_action": "string"
    }
  ]
}
```

