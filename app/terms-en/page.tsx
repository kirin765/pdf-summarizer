import { InfoPage } from "@/app/components/InfoPage";

export default function TermsEn() {
  return (
    <InfoPage lang="en" title="Terms of Service">
      <p>These Terms govern use of QuickPDFSum, an AI PDF summarization service.</p>
      <h2>Service</h2>
      <p>
        This service may be improved, modified, or temporarily suspended without prior notice for
        stability and security reasons.
      </p>
      <h2>Usage Limits</h2>
      <ul>
        <li>Daily quota, file size, and abuse controls may be applied.</li>
        <li>IPs can be blocked for repeated misuse.</li>
      </ul>
      <h2>Disclaimer</h2>
      <p>Summaries are for reference only and should be validated against original documents.</p>
    </InfoPage>
  );
}
