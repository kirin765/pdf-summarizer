export type Lang = "ko" | "en";

export const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;

export function detectLanguage(
  queryLang?: string | null,
  acceptLanguage?: string | null
): Lang {
  if (queryLang === "en" || queryLang === "ko") {
    return queryLang;
  }

  const header = (acceptLanguage || "").toLowerCase();

  if (header.startsWith("en") || header.includes(" en-") || header.includes("en;")) {
    return "en";
  }
  return "ko";
}

export const uiText: Record<
  Lang,
  {
    title: string;
    subtitle: string;
    uploadLabel: string;
    levelLabel: string;
    summarizeButton: string;
    levelOptions: { short: string; medium: string; long: string };
    ocrNotice: string;
    resultLabel: string;
    remainingLabel: string;
    errors: {
      noFile: string;
      invalidType: string;
      fileTooLarge: string;
      noText: string;
      rateLimit: string;
      processing: string;
      unknown: string;
      missingApiKey: string;
      timeout: string;
      apiRateLimit: string;
      apiFailure: string;
    };
  }
> = {
  ko: {
    title: "QuickPDFSum – PDF 자동 요약 서비스",
    subtitle:
      "PDF를 업로드하면 핵심 내용만 빠르게 요약해 드립니다. OCR(스캔 PDF)은 v1에서 제외되어 디지털 PDF 위주로 동작합니다.",
    uploadLabel: "PDF 파일 선택",
    levelLabel: "요약 길이",
    summarizeButton: "요약하기",
    levelOptions: {
      short: "짧게 (3~4문장)",
      medium: "보통 (3~5 불릿)",
      long: "자세히 (A4 1페이지)",
    },
    ocrNotice: "주의: 스캔 PDF는 OCR 미지원(v1)이라 텍스트 추출이 제한될 수 있습니다.",
    resultLabel: "요약 결과",
    remainingLabel: "오늘 남은 무료 횟수",
    errors: {
      noFile: "PDF 파일이 업로드되지 않았습니다.",
      invalidType: "PDF 파일만 업로드해 주세요.",
      fileTooLarge: "업로드 한도는 20MB입니다.",
      noText: "PDF에서 텍스트를 추출할 수 없습니다.",
      rateLimit: "오늘 무료 사용 횟수(20회)를 모두 사용했습니다. 내일 다시 이용해 주세요.",
      processing: "문서 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
      unknown: "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
      missingApiKey: "OPENAI_API_KEY가 설정되지 않았습니다.",
      timeout: "AI 요약 요청 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
      apiRateLimit: "AI 요약 요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
      apiFailure: "AI 요약 처리 중 서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    },
  },
  en: {
    title: "QuickPDFSum – AI PDF Summarizer",
    subtitle:
      "Upload a PDF and get a concise summary quickly. OCR (scanned PDFs) is excluded in v1, so digital PDFs are supported first.",
    uploadLabel: "Choose PDF file",
    levelLabel: "Summary length",
    summarizeButton: "Summarize",
    levelOptions: {
      short: "Short (3-4 sentences)",
      medium: "Medium (3-5 bullets)",
      long: "Long (about 1 A4)",
    },
    ocrNotice: "Note: OCR is disabled in v1, so scanned PDFs may not be processed.",
    resultLabel: "Summary",
    remainingLabel: "Remaining free requests today",
    errors: {
      noFile: "No PDF file was uploaded.",
      invalidType: "Please upload a PDF file only.",
      fileTooLarge: "The upload limit is 20MB.",
      noText: "Could not extract any text from the PDF.",
      rateLimit: "You have used all 20 free requests today. Please try again tomorrow.",
      processing: "An error occurred while processing the document. Please try again.",
      unknown: "An unknown error occurred. Please try again later.",
      missingApiKey: "OPENAI_API_KEY is not configured.",
      timeout: "AI summarization request timed out. Please try again later.",
      apiRateLimit: "AI summarization requests are temporarily too many. Please try again later.",
      apiFailure: "An AI summarization error occurred. Please try again later.",
    },
  },
};
