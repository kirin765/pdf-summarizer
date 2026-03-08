import { InfoPage } from "@/app/components/InfoPage";

export default function ExtractText() {
  return (
    <InfoPage lang="en" title="Text Extraction">
      <p>
        The service extracts text from uploaded PDF pages and sends it to the summarization model.
        This helps you generate short summaries for long documents quickly.
      </p>
      <h2>Note</h2>
      <p>
        OCR is not enabled in v1, so scanned images without selectable text may produce limited results.
      </p>
    </InfoPage>
  );
}
