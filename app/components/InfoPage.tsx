import Link from "next/link";
import type { ReactNode } from "react";

interface InfoPageProps {
  lang: "ko" | "en";
  title: string;
  children: ReactNode;
}

export function InfoPage({ lang, title, children }: InfoPageProps) {
  return (
    <div className="shell">
      <main className="card" style={{ maxWidth: 840, margin: "0 auto" }}>
        <h1>{title}</h1>
        <div style={{ color: "var(--text-sub)", marginBottom: 14 }}>
          <Link href="/">← {lang === "ko" ? "메인 페이지로" : "Back to main"}</Link>
        </div>
        <div style={{ fontSize: 15, lineHeight: 1.7, color: "#d1d5db" }}>{children}</div>
      </main>
    </div>
  );
}
