"use client";

import { FormEvent, useRef, useState } from "react";
import { type Lang, uiText } from "@/lib/locales";

interface MainPageProps {
  initialLang: Lang;
}

export function MainPage({ initialLang }: MainPageProps) {
  const [lang, setLang] = useState<Lang>(initialLang);
  const [level, setLevel] = useState<"short" | "medium" | "long">("medium");
  const [summary, setSummary] = useState("");
  const [error, setError] = useState("");
  const [remaining, setRemaining] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const copy = uiText[lang];

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setSummary("");
    setError("");

    const file = fileRef.current?.files?.[0] || null;

    if (!file) {
      setError(copy.errors.noFile);
      setLoading(false);
      return;
    }

    const form = new FormData();
    form.append("pdf", file);
    form.append("level", level);
    form.append("lang", lang);

    const response = await fetch("/api/summarize", {
      method: "POST",
      body: form,
    });

    const payload = await response.json();
    if (!response.ok) {
      const message = payload?.error || copy.errors.unknown;
      setError(message);
      setLoading(false);
      return;
    }

    setSummary(payload.summary || "");
    setRemaining(typeof payload.remaining === "number" ? payload.remaining : null);
    setLoading(false);
  };

  return (
    <div className="shell">
      <div className="card">
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 24 }}>{copy.title}</div>
            <div style={{ color: "var(--text-sub)", marginTop: 4 }}>{copy.subtitle}</div>
          </div>
          <div style={{ fontSize: 12 }}>
            <a href="/?lang=ko" style={{ marginRight: 10 }} aria-label="Korean version">
              KR
            </a>
            |
            <a href="/?lang=en" style={{ marginLeft: 10 }} aria-label="English version">
              EN
            </a>
          </div>
        </header>

        <form onSubmit={onSubmit}>
          <div className="section">
            <label style={{ display: "block", marginBottom: 8 }}>{copy.uploadLabel}</label>
            <input ref={fileRef} type="file" accept="application/pdf" />
            <div style={{ marginTop: 12 }}>
              <label htmlFor="summaryLevel" style={{ display: "block", marginBottom: 6 }}>
                {copy.levelLabel}
              </label>
              <select
                id="summaryLevel"
                value={level}
                onChange={(event) =>
                  setLevel(event.currentTarget.value as "short" | "medium" | "long")
                }
              >
                <option value="short">{copy.levelOptions.short}</option>
                <option value="medium">{copy.levelOptions.medium}</option>
                <option value="long">{copy.levelOptions.long}</option>
              </select>
            </div>
            <button
              type="submit"
              disabled={loading}
              style={{ marginTop: 12, width: "100%", padding: "10px 14px", borderRadius: 999 }}
            >
              {loading ? "..." : copy.summarizeButton}
            </button>
            <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-sub)" }}>
              {copy.ocrNotice}
            </div>
          </div>
        </form>

        {error ? <div className="section" style={{ borderColor: "rgba(248, 113, 113, 0.6)", color: "var(--danger)" }}>{error}</div> : null}

        {typeof remaining === "number" ? (
          <div className="section" style={{ color: "var(--text-sub)", fontSize: 13 }}>
            {copy.remainingLabel}: {remaining}
          </div>
        ) : null}

        {summary ? (
          <div className="section">
            <h3 style={{ margin: 0, marginBottom: 8 }}>{copy.resultLabel}</h3>
            <pre style={{ whiteSpace: "pre-wrap", fontSize: 14, margin: 0 }}>{summary}</pre>
          </div>
        ) : null}
      </div>
    </div>
  );
}
