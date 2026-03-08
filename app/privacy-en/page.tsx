import { InfoPage } from "@/app/components/InfoPage";

export default function PrivacyEn() {
  return (
    <InfoPage lang="en" title="Privacy Policy">
      <p>QuickPDFSum values user privacy and processes temporary data only to provide summary responses.</p>
      <h2>Collected Information</h2>
      <ul>
        <li>IP address and request metadata</li>
        <li>Uploaded PDF files and extracted text during processing</li>
      </ul>
      <h2>Retention</h2>
      <ul>
        <li>Uploaded PDFs are deleted immediately after summarization.</li>
        <li>Rate-limit information is reset daily.</li>
      </ul>
      <h2>Third Parties</h2>
      <p>We process document text using OpenAI API for summarization only and do not use it for profiling or advertising.</p>
    </InfoPage>
  );
}
