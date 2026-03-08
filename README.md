# PDF Summarizer (Next.js for Vercel)

This repository has been migrated to Next.js (App Router) for Vercel deployment.
The service keeps the same Korean/English behavior and preserves route coverage while
using serverless API routes for PDF summarization.

## Features

- PDF text extraction with `pdf-parse`
- OpenAI summarization (`gpt-4.1-mini`)
- Korean/English UI (auto-detected from `Accept-Language`, override with `?lang=ko|en`)
- In-memory IP/day rate limiting (20 requests/day)
- Health check endpoint at `/api/health`
- OCR intentionally disabled in v1 (as requested)

## Tech Stack

- Frontend/Backend: Next.js 14 (App Router)
- AI: OpenAI SDK
- PDF processing: pdf-parse
- Language: TypeScript

## Environment Variables

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Running

```bash
npm install
npm run dev
```

## API

- `POST /api/summarize`
  - `multipart/form-data` fields:
    - `pdf` (required)
    - `level` (`short | medium | long`)
    - `lang` (`ko | en`)
  - max upload size: 20MB
  - returns: `{ summary, remaining }`
- `GET /api/health`

## Notes

- Legacy Flask implementation files (`app.py`, templates) remain in the repo for reference.
- `uploads/` directory is no longer used by the Next.js implementation.

