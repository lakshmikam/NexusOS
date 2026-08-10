# NexusOS SFT — Failure Modes & Debugging Guide

This document records common failure modes encountered during
QLoRA / SFT training and inference.

---

## 1. Loss Does Not Decrease

### Symptom

Training runs successfully but loss remains almost constant.

Example:

```text
Step 100   loss = 2.91
Step 200   loss = 2.90
Step 300   loss = 2.91