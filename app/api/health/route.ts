import { NextResponse } from "next/server";
import { RATE_LIMIT_PER_DAY } from "@/lib/summarizer";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json({
    status: "healthy",
    rate_limit_per_day: RATE_LIMIT_PER_DAY,
    version: "1.0.0",
  });
}
