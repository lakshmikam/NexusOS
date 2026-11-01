# OrbitDesk billing policy

Policy ID: billing-v1
Version: 1
Effective from: 2026-10-01
Product: fictional OrbitDesk demo

## Price and money units

OrbitDesk Pro costs INR 999 per month in this simulation. Amounts are stored in paise: 99900 represents INR 999.00. Taxes, real payment rails and exchange rates are outside this demo.

## Invoices and payment attempts

An invoice records the amount owed. A payment records an individual payment attempt. Only a payment with status `succeeded` is a confirmed charge. A `pending` or `failed` attempt is not evidence of a successful charge.

## Duplicate-payment investigation

Two distinct successful payment records for the same invoice require duplicate-payment investigation. Do not count retries with the same provider reference as separate payments. Do not promise or execute a refund based only on a customer's message.

## Refund eligibility and approval

A proposed duplicate-charge refund requires verified payment records, no prior refund of that charge, backend eligibility checks, and approval by an authorized operator. Refund execution is simulated and will be implemented in a later phase.
