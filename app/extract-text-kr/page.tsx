import { InfoPage } from "@/app/components/InfoPage";

export default function ExtractTextKr() {
  return (
    <InfoPage lang="ko" title="텍스트 추출">
      <p>
        업로드한 PDF에서 텍스트를 추출해 요약 모델에 전달합니다. 긴 문서도 여러 청크로 분리해 처리합니다.
      </p>
      <h2>제한</h2>
      <p>스캔 PDF는 OCR이 없으면 텍스트 추출이 어려울 수 있습니다.</p>
    </InfoPage>
  );
}
