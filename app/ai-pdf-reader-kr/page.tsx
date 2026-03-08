import { InfoPage } from "@/app/components/InfoPage";

export default function AiPdfReaderKr() {
  return (
    <InfoPage lang="ko" title="AI PDF Reader">
      <p>
        AI PDF Reader는 문서의 핵심 문장과 포인트를 빠르게 정리해줍니다.
      </p>
      <h2>특징</h2>
      <ul>
        <li>문서 텍스트 추출 + 요약</li>
        <li>한글/영문 결과 지원</li>
        <li>긴 문서도 핵심 위주로 정리</li>
      </ul>
    </InfoPage>
  );
}
