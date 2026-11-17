# Acme Pay — API error runbook (fictional demo corpus)

Twelve chunks. The error codes are deliberately adjacent (E-4012, E-4021, E-4102,
E-4120, E-4201, E-4202, E-4210) and so are the API versions (v2.3, v2.4).
Embedding search is weak at telling such identifiers apart, which is the failure
this graph is built to catch. Each `## kb-NN` heading starts one chunk.

## kb-01
E-4012 · Billing sync failed: currency mismatch. The billing sync rejected an invoice
because its currency differs from the customer's account currency. Fix: change the
invoice currency or enable multi-currency on the account, then re-run the billing sync.
The billing sync retries automatically every 10 minutes for up to 6 hours.

## kb-02
E-4021 · Ledger export rejected: duplicate journal reference. The nightly ledger export
to your accounting system was refused because a journal entry reused a reference that
was already posted. Fix: void the duplicate entry, then start a manual export from
Settings → Ledger → Export now. A single export is capped at 5,000 journal lines.

## kb-03
E-4102 · Billing sync paused: payment method expired. The billing sync stopped charging
a customer whose card on file has expired. Fix: ask the customer to update their payment
method; the billing sync resumes on the next billing cycle. Acme Pay sends the customer
3 reminder emails, 7 days apart.

## kb-04
E-4120 · Billing sync throttled: too many invoice updates. More than 300 invoice updates
were sent in one minute, so the billing sync throttled the account. Fix: batch invoice
updates and retry after the time given in the Retry-After header. Throttling clears
after 60 seconds.

## kb-05
E-4201 · Refund rejected: amount exceeds captured total. A refund was requested for more
than the captured amount of the original charge. Fix: refund at most the captured amount;
for partial captures, refund each capture separately. Refunds can be issued up to
180 days after capture.

## kb-06
E-4202 · Refund blocked: dispute open. A refund cannot be issued while a chargeback
dispute is open on the charge. Fix: respond to the dispute in the Disputes tab within
7 days. If the dispute closes in your favour, you can then refund the charge.

## kb-07
E-4210 · Refund delayed: card network timeout. The card network did not confirm the
refund in time. No action is needed: Acme Pay retries the refund automatically for up to
72 hours and the refund status stays "pending" until the network confirms. Do not issue
a second refund, because the customer could be refunded twice.

## kb-08
Webhook signatures. Webhooks on API v2.3 are signed with HMAC-SHA256 using the endpoint
secret. API v2.4 also signs a timestamp and rejects any event older than 300 seconds,
so servers on v2.4 must keep their clocks in sync.

## kb-09
Rate limits. Live-mode API keys allow 100 requests per second and test-mode keys allow
25 requests per second. Requests over the limit return HTTP 429 with a Retry-After header.

## kb-10
Idempotency keys. POST requests accept an Idempotency-Key header. Keys are stored for
24 hours; repeating a request with the same key returns the original response instead
of creating a second charge or refund.

## kb-11
Payouts. Payouts to bank accounts run daily at 02:00 UTC. The minimum payout is 1.00 in
the account currency. A failed payout is retried 2 times before the balance is returned
to the account.

## kb-12
Support escalation. When contacting support, include the error code, the request ID and
the time of the failure. P1 incidents are answered within 1 hour and P2 incidents within
8 hours.
