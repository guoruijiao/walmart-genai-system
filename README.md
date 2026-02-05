# Walmart GenAI System

A production-style GenAI API that demonstrates how to build **grounded, auditable LLM systems** for retail use cases using Retrieval-Augmented Generation (RAG) and schema-validated outputs.

## Overview

This system answers retail questions (e.g. return policy, delivery, store info) by:
1. Retrieving relevant domain documents
2. Generating responses grounded in retrieved context
3. Enforcing a strict JSON contract with citations, intent, confidence, and next actions

The goal is to demonstrate **system-level GenAI design**, not a chatbot demo.

## Architecture

Client → FastAPI → Core Generation  
→ Retrieval Layer (RAG) → LLM (OpenAI Responses API)

LLMs are treated as **reasoning engines**, while factual correctness comes from retrieval.

## Output Contract

All responses follow a structured schema including:
- answer
- citations
- intent
- entities
- confidence
- next_action

This enables downstream automation and evaluation.

## Project Structure 
src/walmart_genai/
api/     # FastAPI endpoints
core/    # LLM interface, schema, generation logic
rag/     # Retrieval components
scripts/   # Local testing utilities
data/docs/ # Domain documents 

## Why this project
This repo focuses on:
- grounding and auditability over free-form text
- explicit interfaces over prompt-only logic
- extensibility to agents, evaluation, and open-source models