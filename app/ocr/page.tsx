import { InfoPage } from "@/app/components/InfoPage";

export default function OcrPage() {
  return (
    <InfoPage lang="en" title="OCR Support">
      <p>
        This V1 release does not include OCR execution on Vercel. The PDF summarizer supports
        text PDFs first, and scanned/image-based PDFs may need OCR in a dedicated pipeline.
      </p>
      <p>For scanned documents in this version, please convert to text-based PDF for best results.</p>
    </InfoPage>
  );
}
