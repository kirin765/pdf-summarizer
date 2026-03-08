import { NextRequest, NextResponse } from "next/server";
import { detectLanguage, MAX_FILE_SIZE_BYTES, type Lang, uiText } from "@/lib/locales";
import { checkRateLimit, extractTextFromPdf, summarizeText } from "@/lib/summarizer";

export const runtime = "nodejs";
export const maxDuration = 60;

type ApiErrorBody = {
  error: string;
};

function getClientIp(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for");
  if (forwarded) {
    return forwarded.split(",")[0].trim();
  }
  return headers.get("x-real-ip") || "unknown";
}

function errorResponse(message: string, status: number) {
  return NextResponse.json<ApiErrorBody>({ error: message }, { status });
}

function normalizeLevel(level: string | null | undefined): "short" | "medium" | "long" {
  if (level === "short" || level === "long") return level;
  return "medium";
}

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const lang = detectLanguage(formData.get("lang") as string | null, request.headers.get("accept-language"));

  const ip = getClientIp(request.headers);
  const { limited, remaining } = checkRateLimit(ip);

  if (limited) {
    return errorResponse(uiText[lang].errors.rateLimit, 429);
  }

  const file = formData.get("pdf");
  if (!(file instanceof File)) {
    return errorResponse(uiText[lang].errors.noFile, 400);
  }

  if (!file.name || !file.name.toLowerCase().endsWith(".pdf")) {
    return errorResponse(uiText[lang].errors.invalidType, 415);
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return errorResponse(uiText[lang].errors.fileTooLarge, 413);
  }

  let buffer: Buffer;
  try {
    const arrayBuffer = await file.arrayBuffer();
    buffer = Buffer.from(arrayBuffer);
  } catch {
    return errorResponse(uiText[lang].errors.processing, 400);
  }

  const level = normalizeLevel(formData.get("level") as string | null);

  try {
    const text = await extractTextFromPdf(buffer);
    if (!text) {
      return errorResponse(uiText[lang].errors.noText, 400);
    }

    const summary = await summarizeText(text, level, lang);

    return NextResponse.json({
      summary,
      lang,
      level,
      remaining,
      rateLimitPerDay: 20,
    });
  } catch (error: unknown) {
    const message = (error as Error)?.message || "";
    if (message === "missing_openai_key") {
      return errorResponse(uiText[lang].errors.missingApiKey, 500);
    }
    if ((error as { name?: string })?.name === "APITimeoutError") {
      return errorResponse(uiText[lang].errors.timeout, 504);
    }
    if ((error as { name?: string })?.name === "RateLimitError") {
      return errorResponse(uiText[lang].errors.apiRateLimit, 429);
    }
    return errorResponse(uiText[lang].errors.apiFailure, 500);
  }
}
