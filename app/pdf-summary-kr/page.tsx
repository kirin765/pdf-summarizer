import { InfoPage } from "@/app/components/InfoPage";

export default function PdfSummaryKr() {
  return (
    <InfoPage lang="ko" title="PDF 요약기 – AI PDF 자동 요약 도구">
      <p>
        QuickPDFSum은 PDF 문서를 AI로 자동 분석해 핵심 내용만 빠르게 정리해 줍니다.
      </p>
      <h2>주요 기능</h2>
      <ul>
        <li>논문, 리포트, 공문서 요약</li>
        <li>짧은 문장 또는 핵심 포인트 형태 정리</li>
        <li>처리 후 파일은 즉시 삭제</li>
      </ul>
      <h2>사용 방법</h2>
      <ol>
        <li>PDF 업로드</li>
        <li>요약 길이 선택</li>
        <li>요약 결과 확인</li>
      </ol>
    </InfoPage>
  );
}
