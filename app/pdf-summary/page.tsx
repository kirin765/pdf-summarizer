import { InfoPage } from "@/app/components/InfoPage";

export default function PdfSummary() {
  return (
    <InfoPage lang="en" title="PDF Summary">
      <p>
        QuickPDFSum summarizes long papers, reports, invoices, and contracts in a readable format.
        Upload a PDF, choose summary depth, and get concise outcomes quickly.
      </p>
      <h2>What you can do</h2>
      <ul>
        <li>Summarize PDFs into short sentences or bullet points</li>
        <li>Use for reports, contracts, research papers, and notes</li>
        <li>Use output for writing, review, and sharing</li>
      </ul>
      <h2>How to use</h2>
      <ol>
        <li>Upload a PDF file</li>
        <li>Pick summary length</li>
        <li>Receive AI-generated summary</li>
      </ol>
    </InfoPage>
  );
}
