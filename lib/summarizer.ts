import OpenAI from "openai";
import pdfParse from "pdf-parse";
import { Lang } from "@/lib/locales";

const parsePdf = ((pdfParse as unknown) as (buffer: Buffer) => Promise<{ text: string }>);

export const RATE_LIMIT_PER_DAY = 20;
const DAY_TO_ISO = () => new Date().toISOString().slice(0, 10);

const _rateLimitStore = new Map<string, { day: string; count: number }>();

export function checkRateLimit(ip: string): { limited: boolean; remaining: number } {
  const today = DAY_TO_ISO();
  const existing = _rateLimitStore.get(ip);

  if (!existing || existing.day !== today) {
    _rateLimitStore.set(ip, { day: today, count: 1 });
    return { limited: false, remaining: RATE_LIMIT_PER_DAY - 1 };
  }

  if (existing.count >= RATE_LIMIT_PER_DAY) {
    return { limited: true, remaining: 0 };
  }

  existing.count += 1;
  return { limited: false, remaining: RATE_LIMIT_PER_DAY - existing.count };
}

function splitTextIntoChunks(
  text: string,
  maxChars: number = 6000,
  overlap: number = 500
): string[] {
  const chunks: string[] = [];
  let start = 0;
  const length = text.length;

  while (start < length) {
    const end = start + maxChars;
    const chunk = text.slice(start, end);
    chunks.push(chunk);
    if (end >= length) break;
    start = Math.max(0, end - overlap);
  }

  return chunks;
}

function buildInstruction(level: string, lang: Lang): string {
  if (lang === "en") {
    if (level === "short") {
      return "Summarize the following PDF in 3-4 concise sentences in English.";
    }
    if (level === "long") {
      return "Summarize the following PDF in about one A4 page in English, with headings and bullet points.";
    }
    return "Summarize the following PDF into 3-5 key bullet points in English.";
  }

  if (level === "short") {
    return "다음 PDF를 3~4개의 간결한 문장으로 요약해 주세요.";
  }
  if (level === "long") {
    return "다음 PDF를 한 페이지 분량으로 제목과 불릿 포인트를 포함해서 자세히 요약해 주세요.";
  }
  return "다음 PDF를 3~5개의 핵심 불릿 포인트로 요약해 주세요.";
}

export async function extractTextFromPdf(fileBuffer: Buffer): Promise<string> {
  const parsed = await parsePdf(fileBuffer);
  return (parsed.text || "").trim();
}

export async function summarizeText(
  text: string,
  level: "short" | "medium" | "long" = "medium",
  lang: Lang = "ko"
): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("missing_openai_key");
  }

  const client = new OpenAI({ apiKey });
  const languageLabel = lang === "en" ? "English" : "Korean";
  const baseInstruction = buildInstruction(level, lang);
  const cleanText = (text || "").trim();

  if (cleanText.length <= 9000) {
    const completion = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: "You are a professional document summarizer." },
        { role: "user", content: `${baseInstruction}\n\n${cleanText}` },
      ]
    });
    return (completion.choices[0].message?.content || "").trim();
  }

  const chunks = splitTextIntoChunks(cleanText, 6000, 500);
  const selectedChunks = chunks.slice(0, 6);
  const partialSummaries: string[] = [];

  for (let i = 0; i < selectedChunks.length; i += 1) {
    const chunkInstruction = `You are summarizing part ${i + 1} of a long PDF document. Focus on key ideas and ignore minor details. Write the summary in ${languageLabel}.\n\n${baseInstruction}`;
    const completion = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: "You are a professional document summarizer." },
        {
          role: "user",
          content: `${chunkInstruction}\n\n${selectedChunks[i]}`,
        },
      ]
    });

    const partial = (completion.choices[0].message?.content || "").trim();
    partialSummaries.push(`[Part ${i + 1}]\n${partial}`);
  }

  let finalInstruction =
    "Below are summaries of each part of a long PDF document. Combine them into ONE final summary in " +
    languageLabel +
    ". ";

  if (level === "short") {
    finalInstruction += "Make it 3-4 sentences.";
  } else if (level === "long") {
    finalInstruction += "Make it about one A4 page with clear structure (headings and bullet points).";
  } else {
    finalInstruction += "Make it 3-7 key bullet points.";
  }

    const completion = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: "You are a professional document summarizer." },
        {
          role: "user",
          content: `${finalInstruction}\n\n${partialSummaries.join("\n\n")}`,
        },
      ]
    });

  return (completion.choices[0].message?.content || "").trim();
}
