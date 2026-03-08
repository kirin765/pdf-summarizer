import { InfoPage } from "@/app/components/InfoPage";

export default function OcrKr() {
  return (
    <InfoPage lang="ko" title="OCR 지원 안내">
      <p>
        현재 v1 버전은 Vercel 환경 제약으로 OCR(이미지 기반 스캔 PDF 처리)을 별도 파이프라인으로 분리했습니다.
      </p>
      <p>스캔 PDF는 텍스트 인식이 제한될 수 있으므로 텍스트 기반 PDF로 변환 후 사용해 주세요.</p>
    </InfoPage>
  );
}
